"""Reference distribution of nu ∝ exp(-E_tgt / T) for small graphs (plan U1).

Multi-chain MALA with parallel tempering on a geometric temperature ladder
T = T_0 < T_1 < ... < T_K. Step sizes are adapted per level during burn-in
(target acceptance ~0.57) and frozen afterwards. Histograms of the raw
hard-decision pattern sign(x) are collected at the target level.

Convergence check: chains are split into `n_groups` independent groups; the
max pairwise TV between group histograms is compared to the TV expected from
sampling noise alone (same number of effective draws).
"""
from __future__ import annotations

import torch

from sbmis.evaluate import code_hist, codes


@torch.no_grad()
def pt_mala(energy, n_chains=4096, n_levels=10, t_max_factor=8.0, burn=4000, n_steps=20000,
            thin=10, swap_every=5, n_groups=4, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    n = energy.n
    T0 = energy.temp
    temps = T0 * t_max_factor ** (torch.arange(n_levels, device=device, dtype=torch.float64)
                                  / max(n_levels - 1, 1))
    inv_T = (1.0 / temps).float()[:, None]                     # (K, 1)
    K, C = n_levels, n_chains
    x = torch.randn(K, C, n, device=device, generator=g)
    h = torch.full((K, 1), 0.05, device=device)               # step sizes

    def U(x):  # (K, C) potential per level
        return energy.e_tgt(x) * inv_T

    def gU(x):
        return energy.grad_tgt(x) * inv_T[..., None]

    u, gu = U(x), gU(x)
    acc_sum = torch.zeros(K, device=device)
    hist = torch.zeros(n_groups, 2**n, dtype=torch.float64, device=device)
    swap_acc = torch.zeros(K - 1, device=device)
    swap_n = 0
    grp = torch.arange(C, device=device) % n_groups
    for it in range(burn + n_steps):
        hs = h[..., None]                                       # (K,1,1)
        xi = torch.randn(x.shape, device=device, generator=g)
        y = x - hs * gu + torch.sqrt(2 * hs) * xi
        uy, gy = U(y), gU(y)
        # log q(x|y) - log q(y|x)
        lq_fwd = -((y - x + hs * gu) ** 2).sum(-1) / (4 * h)
        lq_bwd = -((x - y + hs * gy) ** 2).sum(-1) / (4 * h)
        log_a = (u - uy) + lq_bwd - lq_fwd
        accept = torch.log(torch.rand(K, C, device=device, generator=g)) < log_a
        x = torch.where(accept[..., None], y, x)
        u = torch.where(accept, uy, u)
        gu = torch.where(accept[..., None], gy, gu)
        a_rate = accept.float().mean(1)
        if it < burn:
            h = h * torch.exp(0.05 * (a_rate - 0.57))[:, None]
        else:
            acc_sum += a_rate
        # replica exchange between neighbouring levels (alternating parity)
        if it % swap_every == 0 and K > 1:
            par = (it // swap_every) % 2
            for k in range(par, K - 1, 2):
                e_k, e_k1 = energy.e_tgt(x[k]), energy.e_tgt(x[k + 1])
                la = (e_k - e_k1) * (inv_T[k] - inv_T[k + 1])
                sw = torch.log(torch.rand(C, device=device, generator=g)) < la
                xk, xk1 = x[k].clone(), x[k + 1].clone()
                x[k] = torch.where(sw[:, None], xk1, xk)
                x[k + 1] = torch.where(sw[:, None], xk, xk1)
                if it >= burn:
                    swap_acc[k] += sw.float().mean()
            u, gu = U(x), gU(x)
            if it >= burn:
                swap_n += 1
        if it >= burn and (it - burn) % thin == 0:
            c = codes((x[0] > 0).float())
            for q in range(n_groups):
                hist[q] += code_hist(c[grp == q], n).to(hist)
    p_groups = hist / hist.sum(1, keepdim=True)
    p = hist.sum(0) / hist.sum()
    tvs = [0.5 * (p_groups[a] - p_groups[b]).abs().sum().item()
           for a in range(n_groups) for b in range(a + 1, n_groups)]
    return {
        "p": p.cpu(),
        "temps": temps.tolist(),
        "step_sizes": h.squeeze(1).tolist(),
        "mala_accept": (acc_sum / n_steps).tolist(),
        "swap_accept": (swap_acc / max(swap_n / 2, 1)).tolist(),
        "group_tv_max": max(tvs) if tvs else 0.0,
        "n_draws": int(hist.sum().item()),
        "n_draws_per_group": int(hist[0].sum().item()),
    }


def summarize_reference(p: torch.Tensor, mis_codes: torch.Tensor, feasible: torch.Tensor) -> dict:
    from sbmis.evaluate import entropy
    mode = int(torch.argmax(p))
    return {
        "p_exact_mis": p[mis_codes].sum().item(),
        "feasible_frac": p[feasible].sum().item(),
        "entropy": entropy(p),
        "mode_is_mis": bool((mis_codes == mode).any()),
        "mode_prob": p[mode].item(),
        "n_support_99": int((torch.cumsum(torch.sort(p, descending=True).values, 0) < 0.99).sum()) + 1,
    }

