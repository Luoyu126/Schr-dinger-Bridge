"""Controller / corrector networks: forward(t, x) -> (B, n).

FourierMLP mirrors the official ASBS FourierMLP (zero-initialised output).
GraphNet is a message-passing GNN for a single fixed graph (plan §3.6):
node inputs are x_i plus static degree features; each layer aggregates
neighbour states with the dense adjacency (sum and mean) plus a global mean,
is modulated by the time embedding (FiLM), and is residual + LayerNorm.
Dense A @ h is cheap for the graph sizes used here (n <= a few hundred).
"""
from __future__ import annotations

import math

import torch
from torch import nn


class TimeEmbed(nn.Module):
    def __init__(self, dim_out: int, n_freq: int = 16, channels: int = 64):
        super().__init__()
        self.register_buffer("freqs", torch.arange(1, n_freq + 1).float() * math.pi,
                             persistent=False)
        self.net = nn.Sequential(nn.Linear(2 * n_freq, channels), nn.SiLU(),
                                 nn.Linear(channels, dim_out))

    def forward(self, t):
        t = t.reshape(-1, 1).float()
        arg = t * self.freqs[None]
        return self.net(torch.cat([arg.sin(), arg.cos()], -1))


class FourierMLP(nn.Module):
    def __init__(self, dim: int, channels: int = 256, num_layers: int = 4, **_):
        super().__init__()
        self.input_embed = nn.Linear(dim, channels)
        self.time_embed = TimeEmbed(channels, channels=channels)
        self.hidden = nn.ModuleList(nn.Linear(channels, channels) for _ in range(num_layers - 2))
        self.out = nn.Linear(channels, dim)
        self.act = nn.GELU()
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, t, x):
        h = self.input_embed(x) + self.time_embed(t.expand(x.shape[0], 1) if t.numel() == 1 else t)
        for layer in self.hidden:
            h = layer(self.act(h))
        return self.out(self.act(h))


class GNNLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden))
        self.film = nn.Linear(hidden, 2 * hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, te, A, inv_deg, inv_avg_deg):
        s = torch.matmul(A, h)                      # (B, n, H) neighbour sum
        g = h.mean(1, keepdim=True).expand_as(h)    # global context
        m = self.mlp(torch.cat([h, s * inv_avg_deg, s * inv_deg, g], -1))
        scale, shift = self.film(te).unsqueeze(1).chunk(2, -1)
        return self.norm(h + m * (1 + scale) + shift)


class GraphNet(nn.Module):
    def __init__(self, adj, hidden: int = 64, num_layers: int = 6, **_):
        super().__init__()
        A = torch.as_tensor(adj, dtype=torch.float32)
        n = A.shape[0]
        deg = A.sum(1)
        self.register_buffer("A", A, persistent=False)
        self.register_buffer("inv_deg", (1.0 / deg.clamp_min(1.0))[:, None], persistent=False)
        self.inv_avg_deg = 1.0 / max(deg.mean().item(), 1.0)
        static = torch.stack([deg / deg.max().clamp_min(1.0), 1.0 / (1.0 + deg)], -1)
        self.register_buffer("static", static, persistent=False)  # (n, 2)
        self.n = n
        self.node_in = nn.Linear(1 + static.shape[1], hidden)
        self.time_embed = TimeEmbed(hidden, channels=hidden)
        self.layers = nn.ModuleList(GNNLayer(hidden) for _ in range(num_layers))
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, t, x):
        B = x.shape[0]
        t = t.expand(B, 1) if t.numel() == 1 else t
        feats = torch.cat([x.unsqueeze(-1), self.static.expand(B, -1, -1)], -1)
        h = self.node_in(feats)
        te = self.time_embed(t)
        for layer in self.layers:
            h = layer(h, te, self.A, self.inv_deg, self.inv_avg_deg)
        return self.out(torch.nn.functional.silu(h)).squeeze(-1)


def build_net(kind: str, adj, **kw) -> nn.Module:
    if kind == "gnn":
        return GraphNet(adj, hidden=kw.get("hidden", 64), num_layers=kw.get("layers", 6))
    if kind == "mlp":
        return FourierMLP(adj.shape[0], channels=kw.get("hidden", 256),
                          num_layers=kw.get("layers", 4))
    raise ValueError(kind)
