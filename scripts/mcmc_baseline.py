"""U8: non-learned baseline at the same energy-gradient budget (plan §7.2 U8).

C parallel MALA chains start from the source mu = N(0, I) and target nu ∝ exp(-E_tgt / T);
one step costs C gradient calls. At every budget b the C current states (one sample per
chain) are scored with the same decoder and metrics as the trained samplers, so
scripts/summarize.py can put both on one table. Step sizes adapt online to an acceptance
rate of 0.57. Optional annealing: the temperature decays geometrically from T_hi to T over
the first `anneal_frac` of the final budget and is held at T afterwards (annealed Langevin
with a Metropolis correction).

python scripts/mcmc_baseline.py --graph g2 --ref runs/u1_ref/g2_k2_T0.5.pt --chains 20000 \
    --budgets 76800 153600 307200 614400 --out runs/main_ref/g2/MALA_c20000_s0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbmis.energies.mis import MISEnergy  # noqa: E402
from sbmis.evaluate import Decoder, discrete_metrics, mis_metrics  # noqa: E402
from sbmis.graphs import best_known_mis, get_graph  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="g2")
    ap.add_argument("--lambda_p", type=float, default=2.0)
    ap.add_argument("--kappa_w", type=float, default=2.0)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--chains", type=int, default=1024)
    ap.add_argument("--budgets", type=int, nargs="+", required=True)
    ap.add_argument("--anneal_frac", type=float, default=0.0, help="0 = plain MALA")
    ap.add_argument("--t_hi_factor", type=float, default=8.0)
    ap.add_argument("--h0", type=float, default=0.01)
    ap.add_argument("--adapt_rate", type=float, default=0.2,
                    help="log step size += rate * (acceptance - 0.57) every step")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("log.jsonl", "summary.json"):
        (out / f).unlink(missing_ok=True)
    adj = get_graph(a.graph)
    en = MISEnergy(adj, a.lambda_p, a.kappa_w, a.temp, device=dev)
    dec = Decoder(adj, device=dev)
    ref = torch.load(a.ref, weights_only=False) if a.ref else None
    if ref is not None:
        s = ref["summary"]
        assert s["graph"] == a.graph and s["kappa_w"] == a.kappa_w and s["temp"] == a.temp
        opt = s["mis_size"]
    else:
        opt = best_known_mis(a.graph, adj)["size"]
    C = a.chains
    budgets = sorted(a.budgets)
    steps_total = budgets[-1] // C
    k_anneal = int(a.anneal_frac * steps_total)
    meta = {"args": {**vars(a), "method": "mala"}, "n": en.n, "opt_size": opt,
            "steps_total": steps_total}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    def temp_at(k):
        if k >= k_anneal:
            return a.temp
        return a.temp * a.t_hi_factor ** (1 - k / k_anneal)

    x = torch.randn(C, en.n, device=dev)
    h = torch.full((C, 1), a.h0, device=dev)
    calls, t0, bi = 0, time.time(), 0
    with torch.no_grad():
        T = temp_at(0)
        u, gu = en.e_tgt(x) / T, en.grad_tgt(x) / T
        calls += C
        for k in range(steps_total):
            T_new = temp_at(k)
            if T_new != T:  # re-scale the cached potential to the new temperature
                u, gu, T = u * T / T_new, gu * T / T_new, T_new
            y = x - h * gu + torch.sqrt(2 * h) * torch.randn_like(x)
            uy, gy = en.e_tgt(y) / T, en.grad_tgt(y) / T
            calls += C
            lq_fwd = -((y - x + h * gu) ** 2).sum(-1) / (4 * h[:, 0])
            lq_bwd = -((x - y + h * gy) ** 2).sum(-1) / (4 * h[:, 0])
            acc = torch.log(torch.rand(C, device=dev)) < (u - uy) + lq_bwd - lq_fwd
            x = torch.where(acc[:, None], y, x)
            u = torch.where(acc, uy, u)
            gu = torch.where(acc[:, None], gy, gu)
            h = h * float(np.exp(a.adapt_rate * (acc.float().mean().item() - 0.57)))
            while bi < len(budgets) and calls >= budgets[bi]:
                rec = {"stage": k + 1, "grad_calls": calls, "wall": time.time() - t0,
                       "temp_now": T, "step_size": float(h.mean()), "accept": float(acc.float().mean())}
                rec.update(mis_metrics(x[: (C // 16) * 16], en, dec, opt_size=opt))
                if ref is not None:
                    rec.update(discrete_metrics(x, ref["p"], ref["mis_codes"].to(dev),
                                                ref["feasible"].to(dev)))
                with open(out / "log.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(json.dumps({k_: (round(v, 4) if isinstance(v, float) else v)
                                  for k_, v in rec.items()}), flush=True)
                bi += 1
    (out / "summary.json").write_text(json.dumps({"status": "ok", "grad_calls": calls,
                                                  "wall": time.time() - t0}))


if __name__ == "__main__":
    main()
