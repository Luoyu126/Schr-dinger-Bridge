"""Parallel-tempering reference of the MIS target along a temperature ladder.

Question (AS pilots, round 4): when the annealed AS samplers reach the all-off corner at
T ~ 0.1-0.8, is the target nu_T ∝ exp(-E_tgt / T) itself near-empty there, or has the sampler
lost it? For each validation graph, MALA chains with replica exchange run on a geometric ladder
T_min .. T_max with the samplers' energy (sbmis.energies.mis.MISEnergy). Every `snap_every`
steps after burn-in the chains of every level are decoded and scored like the samplers'
evaluation (sbmis.amortized.decode_metrics, groups of K = 16 samples): raw_size_mean,
raw_feasible_frac, raw_empty_frac, n_distinct, mis_size_mean, best_of_16, E_tgt_mean.
Two independent ensembles check convergence: N(0, I) starts and the degree-greedy maximal
independent set (x = +-1); replicas are exchanged only within an ensemble.

python scripts/pt_ladder.py --graph_ids 0 1 2 3 --out runs/pilots/pt_ladder/g0-3.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbmis.datasets import load_opt, load_split  # noqa: E402
from sbmis.energies.mis import MISEnergy  # noqa: E402
from sbmis.evaluate import Decoder  # noqa: E402

STARTS = ("rand", "greedy")


@torch.no_grad()
def pt_graph(en, temps, C, burn, steps, snap_every, swap_every, seed, dev, greedy):
    """MALA + replica exchange. Returns snapshots (S, L, 2C, n) and diagnostics.

    Chains [0, C) of every level start from N(0, I), chains [C, 2C) from `greedy` (in {0,1}^n).
    The state keeps E_tgt and grad E_tgt per chain, so a swap only permutes them.
    """
    g = torch.Generator(device=dev).manual_seed(seed)
    n, L = en.n, len(temps)
    beta = (1.0 / temps)[:, None]                                         # (L, 1)
    x = torch.randn(L, 2 * C, n, device=dev, generator=g)
    x[:, C:] = 2.0 * greedy - 1.0
    E, G = en.e_tgt(x), en.grad_tgt(x)
    h = torch.full((L, 1), 1e-3, device=dev) * temps[:, None]             # step size per level
    acc = torch.zeros(L, device=dev)
    sw_acc, sw_n = torch.zeros(L - 1, device=dev), torch.zeros(L - 1, device=dev)
    snaps = []
    for it in range(burn + steps):
        hs = h[..., None]
        y = x - hs * beta[..., None] * G + torch.sqrt(2 * hs) * torch.randn(x.shape, device=dev,
                                                                              generator=g)
        Ey, Gy = en.e_tgt(y), en.grad_tgt(y)
        lq_f = ((y - x + hs * beta[..., None] * G) ** 2).sum(-1)
        lq_b = ((x - y + hs * beta[..., None] * Gy) ** 2).sum(-1)
        log_a = beta * (E - Ey) + (lq_f - lq_b) / (4 * h)
        a = torch.log(torch.rand(log_a.shape, device=dev, generator=g)) < log_a
        x = torch.where(a[..., None], y, x)
        E = torch.where(a, Ey, E)
        G = torch.where(a[..., None], Gy, G)
        rate = a.float().mean(1)
        if it < burn:
            h = h * torch.exp(0.05 * (rate - 0.57))[:, None]
        else:
            acc += rate
        if it % swap_every == 0:
            k = torch.arange((it // swap_every) % 2, L - 1, 2, device=dev)
            s = torch.log(torch.rand(len(k), 2 * C, device=dev, generator=g)) < \
                (E[k] - E[k + 1]) * (beta[k] - beta[k + 1])
            for t in (x, E, G):
                lo, hi = t[k].clone(), t[k + 1].clone()
                m = s.view(*s.shape, *([1] * (t.dim() - 2)))
                t[k], t[k + 1] = torch.where(m, hi, lo), torch.where(m, lo, hi)
            if it >= burn:
                sw_acc[k] += s.float().mean(1)
                sw_n[k] += 1
        if it >= burn and (it - burn + 1) % snap_every == 0:
            snaps.append(x.clone())
    return torch.stack(snaps), {"mala_accept": (acc / steps).tolist(),
                                "swap_accept": (sw_acc / sw_n.clamp_min(1)).tolist(),
                                "step_sizes": h.squeeze(1).tolist()}


@torch.no_grad()
def score(snaps, en, dec, C, K):
    """Sampler-evaluation metrics per (snapshot, level, start): dict of arrays (S, L, 2)."""
    S, L, _, n = snaps.shape
    x = snaps.reshape(-1, n)
    s_raw = dec.hard(x)
    size_raw = s_raw.sum(-1)
    sol = dec.extend(dec.repair(s_raw))
    size = sol.sum(-1)
    shp = (S, L, 2, C)
    m = {"E_tgt_mean": en.e_tgt(x).view(shp).mean(-1),
         "raw_size_mean": size_raw.view(shp).mean(-1),
         "raw_feasible_frac": (dec.conflicts(s_raw) == 0).float().view(shp).mean(-1),
         "raw_empty_frac": (size_raw == 0).float().view(shp).mean(-1),
         "mis_size_mean": size.view(shp).mean(-1),
         "best_of_16": size.view(S, L, 2, C // K, K).max(-1).values.mean(-1)}
    sol = sol.view(S, L, 2, C // K, K, n)
    nd = torch.zeros(S, L, 2, C // K)
    for idx in np.ndindex(S, L, 2, C // K):
        nd[idx] = torch.unique(sol[idx], dim=0).shape[0]
    m["n_distinct"] = nd.mean(-1)
    return {k: v.cpu().numpy() for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="er_200_300")
    ap.add_argument("--graph_ids", type=int, nargs="+", default=list(range(16)))
    ap.add_argument("--lambda_p", type=float, default=2.0)
    ap.add_argument("--kappa_w", type=float, default=1.0)
    ap.add_argument("--pmap", default="smooth")
    ap.add_argument("--well", default="full")
    ap.add_argument("--t_min", type=float, default=0.075)
    ap.add_argument("--t_max", type=float, default=24.0)
    ap.add_argument("--levels", type=int, default=60)
    ap.add_argument("--chains", type=int, default=64, help="per level and start (multiple of 16)")
    ap.add_argument("--burn", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--snap_every", type=int, default=2000)
    ap.add_argument("--swap_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev, K = "cuda", 16
    assert a.chains % K == 0 and a.steps % a.snap_every == 0
    val, opt = load_split(a.dataset, "val"), load_opt(a.dataset, "val")
    temps = a.t_min * (a.t_max / a.t_min) ** (torch.arange(a.levels, device=dev, dtype=torch.float32)
                                              / (a.levels - 1))
    res = {"args": vars(a), "temps": temps.tolist(), "graphs": {}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for gi in a.graph_ids:
        t0 = time.time()
        adj = val.adj(gi)
        en = MISEnergy(adj, a.lambda_p, a.kappa_w, a.t_min, device=dev, pmap=a.pmap, well=a.well)
        dec = Decoder(adj, device=dev)
        greedy = dec.extend(torch.zeros(1, en.n, device=dev))[0]
        snaps, diag = pt_graph(en, temps, a.chains, a.burn, a.steps, a.snap_every, a.swap_every,
                               a.seed + gi, dev, greedy)
        t_pt = time.time() - t0
        m = score(snaps, en, dec, a.chains, K)
        # per level and start: mean over snapshots, plus the first and last half (drift check)
        S = snaps.shape[0]
        levels = []
        for lv in range(a.levels):
            row = {"T": float(temps[lv])}
            for j, st in enumerate(STARTS):
                row[st] = {k: float(v[:, lv, j].mean()) for k, v in m.items()}
                row[st]["raw_size_half"] = [float(m["raw_size_mean"][:S // 2, lv, j].mean()),
                                            float(m["raw_size_mean"][S // 2:, lv, j].mean())]
                row[st]["E_half"] = [float(m["E_tgt_mean"][:S // 2, lv, j].mean()),
                                     float(m["E_tgt_mean"][S // 2:, lv, j].mean())]
            levels.append(row)
        res["graphs"][gi] = {"n": en.n, "opt": opt[gi]["size"], "greedy_size": float(greedy.sum()),
                             "levels": levels, **diag, "wall_pt": t_pt, "wall": time.time() - t0}
        out.write_text(json.dumps(res))
        lo = levels[0]
        print(f"graph {gi} n={en.n} opt={opt[gi]['size']} greedy={int(greedy.sum())} "
              f"pt {t_pt:.0f}s total {time.time() - t0:.0f}s | T={lo['T']:.3f} best16 "
              f"rand {lo['rand']['best_of_16']:.2f} greedy {lo['greedy']['best_of_16']:.2f} | "
              f"swap acc min {min(diag['swap_accept']):.3f} | mala acc "
              f"{min(diag['mala_accept']):.2f}-{max(diag['mala_accept']):.2f}", flush=True)


if __name__ == "__main__":
    main()
