"""PQQA, parallel quasi-quantum annealing (Ichikawa & Arai, ICLR 2025, arXiv 2409.02135), batched over graphs.

Follows qqa.optimize() of the first public release (github.com/Yuma-Ichikawa/QQA4CO, commit b8721ae,
src/qqa.py) and the paper (Sec. 4-5, App. C). S parallel runs x^(s) in [0, 1]^n (sigma(w) = Clamp(w),
App. C.1, not a sigmoid) start from U(0, 1) and take AdamW steps (weight decay 0.01, torch's default)
on
    sum_s [ l(x^(s)) + gamma_t s(x^(s)) ] (1 - alpha)  -  alpha S sum_i STD_s[x_i^(s)]
(Eq. 10; the (1 - alpha) factor is the code's), followed by Langevin noise sqrt(2 lr T) xi and a clamp
to [0, 1]. MIS loss (App. D.1, the code's QUBO): l(x) = x^T Q x, Q_ii = -1, Q_uv = Q_vu = penalty on
edges, i.e. -sum_i x_i^2 + 2 penalty sum_E x_u x_v. alpha-entropy s(x) = sum_i 1 - (1 - 2x_i)^4.
gamma_t rises linearly from -2 to 0.1 over the steps. After every step all runs are rounded and the
rounded vector with the lowest l is kept (as the code does).

Defaults: lr 1 and temp 1e-3 (the code's MIS notebooks; paper App. C.3: lr from {1, 0.1, 0.01},
T = 0.001), alpha = 0.1 (div_param default of that release), penalty 3 (mis_er_small in the next
commit, 558a7ce, src/instance.py). The paper's ER-[700-800] runs use S in {100, 1000} and 3,000 or
30,000 steps (Sec. 5.1).

Reported here beyond the code: the largest independent rounded vector seen in any run at any step
(best_feasible), so that a lowest-loss vector with a conflict cannot inflate the set size.
Differences from the original: random numbers; all graphs run together (padded, padded nodes held
at 0), so the wall time is for the whole batch.
"""
from __future__ import annotations

import time

import torch

from .common import pad_graphs


def pqqa(adjs, *, runs=100, steps=3000, lr=1.0, temp=1e-3, min_bg=-2.0, max_bg=0.1, curve_rate=4,
         div_param=0.1, penalty=3.0, weight_decay=0.01, device="cuda", seed=0, record_every=100,
         log_every=0):
    """Run PQQA on each graph in `adjs` (list of n_g x n_g 0/1 arrays).

    Returns a dict:
        best_loss_size, best_loss_conflicts  (G,) size and number of conflicting edges of the
                                             lowest-loss rounded vector (what the code returns)
        best_feasible  (G,) largest independent rounded vector seen;  best_set: list of (n_g,) bool
        curve          list of dicts every `record_every` steps: step, time, gamma, best_feasible (G,),
                       frac_feasible (G,) (independent rounded runs at this step),
                       mean_size (G,) (mean rounded size over runs), mean_std (G,) (mean over nodes
                       of the across-run STD of x), binary_frac (G,) (entries within 0.01 of 0 or 1)
        time           wall time of the whole run in seconds
    """
    torch.manual_seed(seed)
    G = len(adjs)
    A, valid = pad_graphs(adjs, device=device, dtype=torch.float32)
    n = A.shape[1]
    mask = valid.unsqueeze(1)                                                  # G x 1 x n
    x = (torch.rand(G, runs, n, device=device)*mask).requires_grad_(True)
    opt = torch.optim.AdamW([x], lr=lr, weight_decay=weight_decay)
    best_loss = torch.full((G,), float("inf"), device=device)
    bl_size = torch.zeros(G, device=device)
    bl_conf = torch.zeros(G, device=device)
    best_feas = torch.zeros(G, device=device)
    best_set = torch.zeros(G, n, device=device, dtype=torch.bool)
    ar = torch.arange(G, device=device)
    curve = []

    def check(step, bg, xr):
        """Update the bests from the rounded runs xr (G x S x n); record the curve."""
        nonlocal best_loss, bl_size, bl_conf, best_feas, best_set
        k = torch.bmm(xr, A)
        quad = (xr*k).sum(-1)                                                  # 2 * conflicts
        size = xr.sum(-1)
        loss = -size + penalty*quad
        lmin, li = loss.min(1)
        better = lmin < best_loss
        best_loss = torch.where(better, lmin, best_loss)
        bl_size = torch.where(better, size[ar, li], bl_size)
        bl_conf = torch.where(better, quad[ar, li]/2, bl_conf)
        feas = quad == 0
        fsize = torch.where(feas, size, torch.full_like(size, -1))
        top, fi = fsize.max(1)
        fb = top > best_feas
        best_feas = torch.where(fb, top, best_feas)
        best_set = torch.where(fb.unsqueeze(1), xr[ar, fi] > 0, best_set)
        if record_every and (step % record_every == 0 or step == steps):
            if device != "cpu":
                torch.cuda.synchronize()
            xd = x.detach()
            nv = valid.sum(1)
            near = (((xd < 0.01) | (xd > 0.99)).float()*mask).sum((1, 2))/(nv*runs)
            curve.append({"step": step, "time": time.time() - t0, "gamma": bg,
                          "best_feasible": best_feas.cpu().tolist(),
                          "frac_feasible": feas.float().mean(1).cpu().tolist(),
                          "mean_size": size.mean(1).cpu().tolist(),
                          "mean_std": ((xd.std(1)*valid).sum(1)/nv).cpu().tolist(),
                          "binary_frac": near.cpu().tolist()})
        if log_every and step % log_every == 0:
            print(f"step {step}  {time.time() - t0:.0f}s  gamma {bg:.3f}  best feasible "
                  f"{best_feas.mean().item():.3f}  frac feasible {feas.float().mean().item():.3f}",
                  flush=True)

    if device != "cpu":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        check(0, min_bg, x.round())
    for epoch in range(steps):
        opt.zero_grad()
        bg = min_bg + (max_bg - min_bg)*epoch/steps
        losses = -(x*x).sum(-1) + penalty*(x*torch.bmm(x, A)).sum(-1)          # G x S
        pens = (1 - (1 - 2*x)**curve_rate).mul(mask).sum(-1)
        div = -(x.std(1)*valid).sum()*runs
        total = (losses + pens*bg).sum()*(1 - div_param) + div*div_param
        total.backward()
        opt.step()
        with torch.no_grad():
            x.add_(torch.randn_like(x)*(2*lr*temp)**0.5).clamp_(0, 1).mul_(mask)
            check(epoch + 1, bg, x.round())
    if device != "cpu":
        torch.cuda.synchronize()
    sizes = [a.shape[0] for a in adjs]
    return {"best_loss_size": bl_size.cpu(), "best_loss_conflicts": bl_conf.cpu(),
            "best_feasible": best_feas.cpu(), "best_set": [best_set[g, :sizes[g]].cpu() for g in range(G)],
            "curve": curve, "time": time.time() - t0}
