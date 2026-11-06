"""Decoding and metrics (plan §3.7, §7.1).

decode: hard decision p_i = 1[x_i > 0]
repair: visit vertices by graph degree (high -> low); drop a vertex if it
        still has a selected neighbour  -> independent set
extend: visit vertices by degree (low -> high); add if no selected neighbour
        -> maximal independent set
Raw (pre-repair) metrics are always reported next to post-repair ones.
"""
from __future__ import annotations

import numpy as np
import torch


class Decoder:
    def __init__(self, adj, device="cuda"):
        A = torch.as_tensor(np.asarray(adj), dtype=torch.float32, device=device)
        self.A = A
        deg = A.sum(1)
        self.n = A.shape[0]
        self.n_edges = int(A.sum().item() // 2)
        # stable orders (ties by index)
        self.order_desc = sorted(range(self.n), key=lambda i: (-deg[i].item(), i))
        self.order_asc = sorted(range(self.n), key=lambda i: (deg[i].item(), i))

    def hard(self, x):
        return (x > 0).float()

    def conflicts(self, s):
        """Number of violated edges per sample."""
        return 0.5 * ((s @ self.A) * s).sum(-1)

    def repair(self, s):
        s = s.clone()
        for v in self.order_desc:
            bad = (s[:, v] > 0) & ((s @ self.A[:, v]) > 0)
            s[bad, v] = 0.0
        return s

    def extend(self, s):
        s = s.clone()
        for v in self.order_asc:
            free = (s[:, v] == 0) & ((s @ self.A[:, v]) == 0)
            s[free, v] = 1.0
        return s

    def decode(self, x):
        return self.extend(self.repair(self.hard(x)))


def diversity(sols: torch.Tensor) -> dict:
    """sols: (K, n) {0,1}. Distinct count and mean pairwise Hamming of distinct ones."""
    uniq = torch.unique(sols, dim=0)
    k = uniq.shape[0]
    if k < 2:
        return {"n_distinct": k, "mean_hamming": 0.0}
    D = torch.cdist(uniq, uniq, p=1)
    return {"n_distinct": k, "mean_hamming": (D.sum() / (k * (k - 1))).item()}


@torch.no_grad()
def mis_metrics(x: torch.Tensor, energy, dec: Decoder, k_best=(1, 16), k_div=64,
                opt_size: int | None = None) -> dict:
    """x: (N, n) terminal samples. N should be a multiple of max(k_best)."""
    s_raw = dec.hard(x)
    conf = dec.conflicts(s_raw)
    sol = dec.extend(dec.repair(s_raw))
    size = sol.sum(-1)
    out = {
        "E_tgt_mean": energy.e_tgt(x).mean().item(),
        "raw_size_mean": s_raw.sum(-1).mean().item(),
        "raw_conflict_edge_frac": (conf.mean() / max(dec.n_edges, 1)).item(),
        "raw_feasible_frac": (conf == 0).float().mean().item(),
        "repaired_only_size_mean": dec.repair(s_raw).sum(-1).mean().item(),
        "mis_size_mean": size.mean().item(),
        "mis_size_max": size.max().item(),
    }
    for K in k_best:
        m = (size.shape[0] // K) * K
        out[f"best_of_{K}"] = size[:m].view(-1, K).max(1).values.mean().item()
    out.update({f"div_{k}": v for k, v in diversity(sol[:k_div]).items()})
    if opt_size:
        out["gap_mean"] = 1.0 - out["mis_size_mean"] / opt_size
        out[f"gap_best_of_{max(k_best)}"] = 1.0 - out[f"best_of_{max(k_best)}"] / opt_size
    if not torch.isfinite(x).all():
        out["nonfinite"] = True
    return out


# --------------------------------------------------------------- U1 (n <= 16)
def codes(s: torch.Tensor) -> torch.Tensor:
    """{0,1}^n -> int64 code."""
    n = s.shape[1]
    w = (2 ** torch.arange(n, device=s.device, dtype=torch.int64))
    return (s.long() * w).sum(-1)


def code_hist(c: torch.Tensor, n: int) -> torch.Tensor:
    return torch.bincount(c, minlength=2**n).double()


def entropy(p: torch.Tensor) -> float:
    q = p[p > 0]
    return float(-(q * q.log()).sum())


@torch.no_grad()
def discrete_metrics(x: torch.Tensor, ref_p: torch.Tensor, mis_codes: torch.Tensor,
                     feasible_mask: torch.Tensor) -> dict:
    """Distribution of the raw hard-decision pattern vs the reference."""
    n = x.shape[1]
    h = code_hist(codes((x > 0).float()), n)
    p = h / h.sum()
    ref_p = ref_p.to(p)
    return {
        "tv": 0.5 * (p - ref_p).abs().sum().item(),
        "entropy": entropy(p),
        "p_exact_mis": p[mis_codes].sum().item(),
        "feasible_frac": p[feasible_mask].sum().item(),
        "n_samples": int(h.sum().item()),
    }


def tv_noise_floor(ref_p: torch.Tensor, n_samples: int, reps: int = 5, seed: int = 0) -> float:
    """Expected TV between ref_p and an empirical distribution of n_samples draws from it."""
    g = torch.Generator().manual_seed(seed)
    p = ref_p.double().cpu()
    vals = []
    for _ in range(reps):
        idx = torch.multinomial(p, n_samples, replacement=True, generator=g)
        e = torch.bincount(idx, minlength=p.numel()).double() / n_samples
        vals.append(0.5 * (e - p).abs().sum().item())
    return float(np.mean(vals))


def group_tv_noise_floor(ref_p: torch.Tensor, n_per_group: int, n_groups: int,
                         reps: int = 5, seed: int = 0) -> float:
    """Expected max pairwise TV between n_groups iid empirical histograms of n_per_group draws.

    This is the iid baseline for the PT-MALA convergence statistic `group_tv_max`;
    (floor / observed)^2 estimates the effective-sample fraction of the correlated draws.
    """
    g = torch.Generator().manual_seed(seed)
    p = ref_p.double().cpu()
    vals = []
    for _ in range(reps):
        hs = []
        for _ in range(n_groups):
            idx = torch.multinomial(p, n_per_group, replacement=True, generator=g)
            hs.append(torch.bincount(idx, minlength=p.numel()).double() / n_per_group)
        vals.append(max(0.5 * (hs[a] - hs[b]).abs().sum().item()
                        for a in range(n_groups) for b in range(a + 1, n_groups)))
    return float(np.mean(vals))


def feasible_mask_all(adj: np.ndarray) -> torch.Tensor:
    """Boolean over all 2^n codes: is the pattern an independent set."""
    n = adj.shape[0]
    assert n <= 20
    c = torch.arange(2**n, dtype=torch.int64)
    bits = ((c[:, None] >> torch.arange(n)) & 1).double()
    A = torch.as_tensor(adj, dtype=torch.float64)
    return ((bits @ A) * bits).sum(-1) == 0

