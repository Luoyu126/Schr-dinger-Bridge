"""pCQO-MIS (Alkhouri et al., ICML 2025, arXiv 2406.19532), batched over graphs.

Follows the official C++ solver (github.com/ledenmat/pCQO-mis-benchmark, cpp_impl/pcqo_mis.cpp,
main()); the defaults are its ER-[700-800] settings (the commented "ER settings" block there, and the
same values in benchmark.py). On x in [0, 1]^n it minimises
    f(x) = -sum_v x_v + gamma sum_{(u,v) in E} x_u x_v - gamma' sum_{(u,v) in E'} x_u x_v
(E' = edges of the complement graph) by momentum gradient descent projected onto the box:
    g = -1 + gamma X A - gamma' X A',  v <- momentum v + lr g,  X <- clamp(X - v, 0, 1)
(A' = 1 - A - I; the velocity is never reset). Every `steps_per_batch` steps, each of the `batch`
vectors is thresholded at 0.5, a threshold mask that is a maximal independent set may replace the
best set, and every vector is resampled as X ~ N(X, std^2) (std 2.25, not clamped until the next
step). The first X is drawn from N(1 - deg/max deg, std^2). The C++ code runs in float16 on the GPU;
dtype=torch.float16 does the same.

The C++ maximal-set test, clamp(m - 0.1(-1 + m(nA - A')), 0, 1) == m, holds exactly when m is
independent and every node outside m has a neighbour in m (node v in m: (n+1) k_v <= |m| forces
k_v = 0; v outside m: (n+1) k_v >= |m| + 1 forces k_v >= 1, with k_v = set neighbours of v);
common.is_maximal_independent tests that directly.

Differences from the C++ run: random numbers (torch.manual_seed(seed) here, seed 113 there), and all
graphs of the batch run together on one GPU (padded; padded nodes are held at 0), so the wall time
is for the whole batch, not per graph.
"""
from __future__ import annotations

import time

import torch

from .common import is_maximal_independent, pad_graphs


@torch.no_grad()
def pcqo(adjs, *, lr=9e-6, momentum=0.9, steps=225_000, steps_per_batch=450, gamma=350.0,
         gamma_prime=7.0, batch=256, std=2.25, dtype=torch.float16, device="cuda", seed=113,
         log_every=0):
    """Run pCQO on each graph in `adjs` (list of n_g x n_g 0/1 arrays).

    Returns a dict:
        best_size  (G,) size of the largest maximal independent set found
        best_set   list of (n_g,) bool tensors (all False if none was found)
        curve      list of dicts per check: step, time, best_size (G,), frac_maximal (G,),
                   mean_maximal_size (G,) (mean size of the maximal sets among the batch, nan if none)
        time       wall time of the whole run in seconds
    """
    torch.manual_seed(seed)
    G = len(adjs)
    A, valid = pad_graphs(adjs, device=device, dtype=torch.float32)
    n = A.shape[1]
    eye = torch.eye(n, device=device)
    vv = valid.unsqueeze(2)*valid.unsqueeze(1)
    Ac = (1 - A - eye)*vv
    deg = A.sum(1)
    mean = (1 - deg/deg.max(1, keepdim=True).values)*valid
    A_h, Ac_h = A.to(dtype), Ac.to(dtype)
    mask = valid.unsqueeze(1).to(dtype)                                      # G x 1 x n
    X = (mean.unsqueeze(1).expand(G, batch, n) + std*torch.randn(G, batch, n, device=device)).to(dtype)*mask
    vel = torch.zeros(G, batch, n, device=device, dtype=dtype)
    best = torch.zeros(G, device=device)
    best_set = torch.zeros(G, n, device=device, dtype=torch.bool)
    curve = []
    if device != "cpu":
        torch.cuda.synchronize()
    t0 = time.time()
    for it in range(steps):
        grad = -1 + gamma*torch.bmm(X, A_h) - gamma_prime*torch.bmm(X, Ac_h)
        vel = momentum*vel + lr*grad
        X = (X - vel).clamp(0, 1)*mask
        if (it + 1) % steps_per_batch == 0:
            m = (X > 0.5).float()
            ok = is_maximal_independent(A, m, valid)                         # G x B
            size = torch.where(ok, m.sum(-1), torch.full_like(ok, -1, dtype=torch.float32))
            top, idx = size.max(1)
            better = top > best
            best = torch.where(better, top, best)
            best_set = torch.where(better.unsqueeze(1), m[torch.arange(G, device=device), idx] > 0, best_set)
            nok = ok.sum(1)
            mean_size = torch.where(nok > 0, (m.sum(-1)*ok).sum(1)/nok.clamp(min=1),
                                    torch.full_like(best, float("nan")))
            if device != "cpu":
                torch.cuda.synchronize()
            curve.append({"step": it + 1, "time": time.time() - t0, "best_size": best.cpu().tolist(),
                          "frac_maximal": (nok.float()/batch).cpu().tolist(),
                          "mean_maximal_size": mean_size.cpu().tolist()})
            if log_every and (it + 1) % (log_every*steps_per_batch) == 0:
                print(f"step {it + 1}  {time.time() - t0:.0f}s  mean best {best.mean().item():.3f}  "
                      f"frac maximal {(nok.float()/batch).mean().item():.3f}", flush=True)
            if it + 1 != steps:
                X = (X.float() + std*torch.randn_like(X, dtype=torch.float32)).to(dtype)*mask
    if device != "cpu":
        torch.cuda.synchronize()
    sizes = [a.shape[0] for a in adjs]
    return {"best_size": best.cpu(), "best_set": [best_set[g, :sizes[g]].cpu() for g in range(G)],
            "curve": curve, "time": time.time() - t0}
