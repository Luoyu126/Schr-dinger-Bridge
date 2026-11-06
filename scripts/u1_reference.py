"""U1: reference distributions of nu on small graphs + (kappa_w, T) calibration.

python scripts/u1_reference.py --graphs g0 g1 g2 --kappas 1 2 4 --temps 1 0.5 0.25
Saves runs/u1_ref/<graph>_k<kappa>_T<temp>.pt (probabilities over all 2^n
hard-decision patterns) and prints a summary table row per config.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbmis.energies.mis import MISEnergy  # noqa: E402
from sbmis.evaluate import codes, feasible_mask_all, tv_noise_floor  # noqa: E402
from sbmis.graphs import all_maximum_independent_sets, get_graph  # noqa: E402
from sbmis.reference import pt_mala, summarize_reference  # noqa: E402


def ref_path(out, graph, kappa, temp):
    return Path(out) / f"{graph}_k{kappa:g}_T{temp:g}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="+", default=["g0", "g1", "g2"])
    ap.add_argument("--kappas", nargs="+", type=float, default=[1, 2, 4])
    ap.add_argument("--temps", nargs="+", type=float, default=[1, 0.5, 0.25])
    ap.add_argument("--lambda_p", type=float, default=2.0)
    ap.add_argument("--chains", type=int, default=4096)
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--burn", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out", default="runs/u1_ref")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for gname in a.graphs:
        adj = get_graph(gname)
        mis = all_maximum_independent_sets(adj)
        mis_codes = codes(torch.tensor(mis, dtype=torch.float32))
        feas = feasible_mask_all(adj)
        for kappa in a.kappas:
            for temp in a.temps:
                t0 = time.time()
                en = MISEnergy(adj, a.lambda_p, kappa, temp, device="cuda")
                r = pt_mala(en, n_chains=a.chains, n_levels=a.levels, burn=a.burn,
                            n_steps=a.steps)
                p = r.pop("p")
                s = summarize_reference(p, mis_codes, feas)
                # TV between groups expected from iid noise alone (lower bound for correlated draws)
                s["group_tv_iid_floor"] = tv_noise_floor(p, r["n_draws_per_group"])
                s.update({k: r[k] for k in ("group_tv_max", "n_draws")})
                s.update(graph=gname, n=adj.shape[0], n_edges=int(adj.sum() // 2),
                         mis_size=int(mis[0].sum()), n_mis=len(mis), kappa_w=kappa, temp=temp,
                         mala_accept_T0=r["mala_accept"][0], swap_accept_min=min(r["swap_accept"]),
                         secs=round(time.time() - t0, 1))
                torch.save({"p": p, "summary": s, "diag": r, "mis_codes": mis_codes,
                            "feasible": feas}, ref_path(a.out, gname, kappa, temp))
                print(json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
