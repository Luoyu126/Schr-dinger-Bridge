"""DIFUSCO's MIS network (gated GCN, node-feature-only sparse mode) in plain PyTorch.

Port of GNNEncoder(node_feature_only=True, sparse=True) from Edward-Sun/DIFUSCO
(difusco/models/gnn_encoder.py), the architecture DIFUSCO uses for MIS (12 layers, hidden 256,
sum aggregation, layer norm). Per layer, with node states h and edge states e:

    e'_ij = A h_j + B h_i + C e_ij,   gate_ij = sigmoid(e'_ij)     (as in DIFUSCO's code,
                                                                  which swaps A/B vs its doc)
    h'_i  = U h_i + sum_j gate_ij * V h_j                    (edge j -> i, sum aggregation)
    h, e  = relu(LN(h')), relu(LN(e'))                        ("direct" mode)
    h     = h_in + h + W_t time_emb                           (time added to nodes)
    e     = e_in + zero_init_MLP(e)

Input: the node scalar x_i through DIFUSCO's ScalarEmbeddingSine1D and a linear layer; time
through the sinusoidal timestep embedding (t in [0, 1] is scaled by 1000 to DIFUSCO's step
range) and a 2-layer MLP. Edge states start at zero.

Like DIFUSCO's MISDataset, every node gets a self-loop in the GNN's edge list
(`self_loops=True`): a gated self-message with its own edge state. The energy never sees
these loops (they are added inside the network).

Deviations, all required by ASBS and kept minimal:
  * torch_sparse is replaced by index_add_ (same sum aggregation);
  * several independent samples (each with its own graph and time) are batched as one
    disjoint union; the time embedding is gathered per node;
  * DIFUSCO's output head normalises with GroupNorm over all nodes of the batch, which
    would couple independent samples; it is replaced by a per-node LayerNorm;
  * the last linear layer is zero-initialised, so the controller starts at u = 0 (as the
    ASBS / AS networks do).
Like DIFUSCO on ER graphs (--use_activation_checkpoint), each layer can be checkpointed
during training (`activation_checkpoint=True`): same outputs and gradients, edge
activations are recomputed in the backward pass instead of stored.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32,
                                                           device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def scalar_embedding_sine_1d(x: torch.Tensor, dim: int, temperature: float = 10000.0):
    """DIFUSCO's ScalarEmbeddingSine1D (normalize=False)."""
    d = torch.arange(dim, dtype=torch.float32, device=x.device)
    d = temperature ** (2 * torch.div(d, 2, rounding_mode="trunc") / dim)
    p = x[:, None] / d
    return torch.stack((p[:, 0::2].sin(), p[:, 1::2].cos()), dim=2).flatten(1)


class GatedGCNLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.U, self.V, self.A, self.B, self.C = (nn.Linear(hidden, hidden) for _ in range(5))
        self.norm_h = nn.LayerNorm(hidden)
        self.norm_e = nn.LayerNorm(hidden)

    def forward(self, h, e, src, dst):
        """Edge (src -> dst): DIFUSCO's edge_index[1] is the neighbour j, edge_index[0] the
        receiving node i. Each undirected edge appears in both directions."""
        Vh = self.V(h)[src]
        e_new = self.A(h)[src] + self.B(h)[dst] + self.C(e)
        gates = torch.sigmoid(e_new)
        msg = gates * Vh
        agg = torch.zeros(h.shape, dtype=msg.dtype, device=h.device).index_add_(0, dst, msg)
        h_new = self.U(h) + agg
        return torch.relu(self.norm_h(h_new)), torch.relu(self.norm_e(e_new))


class DifuscoGNN(nn.Module):
    """Graph-conditioned network v(t, x; G) -> one scalar per node.

    forward(t, x, batch) with x (N_nodes,) node scalars of all samples, t (S,) one time per
    sample and `batch` a GraphBatch-like object with .src, .dst (directed edges, both ways)
    and .node2sample (N_nodes,). Returns (N_nodes,).
    """

    def __init__(self, hidden: int = 256, n_layers: int = 12, time_scale: float = 1000.0,
                 activation_checkpoint: bool = False, self_loops: bool = True):
        super().__init__()
        self.hidden, self.time_scale = hidden, time_scale
        self.activation_checkpoint = activation_checkpoint
        self.self_loops = self_loops
        t_dim = hidden // 2
        self.node_embed = nn.Linear(hidden, hidden)
        self.time_embed = nn.Sequential(nn.Linear(hidden, t_dim), nn.ReLU(), nn.Linear(t_dim, t_dim))
        self.layers = nn.ModuleList(GatedGCNLayer(hidden) for _ in range(n_layers))
        self.time_layers = nn.ModuleList(nn.Sequential(nn.ReLU(), nn.Linear(t_dim, hidden))
                                         for _ in range(n_layers))
        self.edge_out = nn.ModuleList()
        for _ in range(n_layers):
            lin = nn.Linear(hidden, hidden)
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.edge_out.append(nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), lin))
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, t: torch.Tensor, x: torch.Tensor, batch) -> torch.Tensor:
        src, dst, n2s = batch.src, batch.dst, batch.node2sample
        if self.self_loops:  # DIFUSCO's MISDataset appends (i, i) for every node
            loops = torch.arange(x.shape[0], device=x.device)
            src, dst = torch.cat([src, loops]), torch.cat([dst, loops])
        wd = self.node_embed.weight.dtype  # embeddings follow the parameter dtype
        h = self.node_embed(scalar_embedding_sine_1d(x, self.hidden).to(wd))
        e = torch.zeros(src.shape[0], self.hidden, device=x.device, dtype=h.dtype)
        temb = self.time_embed(
            timestep_embedding(t.reshape(-1) * self.time_scale, self.hidden).to(wd))
        ckpt = self.activation_checkpoint and torch.is_grad_enabled()
        for layer, tl, eo in zip(self.layers, self.time_layers, self.edge_out):
            if ckpt:
                h, e = checkpoint(self._block, layer, tl, eo, h, e, temb, src, dst, n2s,
                                  use_reentrant=False)
            else:
                h, e = self._block(layer, tl, eo, h, e, temb, src, dst, n2s)
        return self.out(torch.relu(self.out_norm(h))).squeeze(-1)

    @staticmethod
    def _block(layer, tl, eo, h, e, temb, src, dst, n2s):
        h_new, e_new = layer(h, e, src, dst)
        return h + h_new + tl(temb)[n2s], e + eo(e_new)
