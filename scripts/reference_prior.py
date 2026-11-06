"""How informative is each reference process on its own? (u = 0 diagnostic)

For every group of the reference-process experiment: draw X_0 from the source, run the
uncontrolled base SDE to t = 1 and score the hard decisions of X_1, exactly as a trained
sampler is scored. On U1 graphs this is the TV to the PT-MALA reference; on larger graphs
the raw (pre-repair) conflict rate, raw set size and mean E_tgt.

python scripts/reference_prior.py --graphs g0 g1 g2 er:100:0.15:0 --out runs/reference_prior.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbmis.base.linear_energy_base import build_base  # noqa: E402
from sbmis.energies.mis import MISEnergy  # noqa: E402
from sbmis.evaluate import Decoder, discrete_metrics, mis_metrics  # noqa: E402
from sbmis.graphs import get_graph  # noqa: E402
from sbmis.sources import build_source  # noqa: E402

# name -> (base kind, base kwargs, source kind, source kwargs)
GROUPS = {
    "B0": ("zero", {}, "gauss", {}),
    "B0-S": ("ve_geometric", {"sigma_min": 0.001}, "gauss", {}),
    "AS": ("ve_geometric", {"sigma_min": 0.001}, "delta", {}),
    "VP": ("vp", {}, "gauss", {}),
    "iso-OU@4": ("iso_ou", {"iso_match_rho": 4.0}, "gauss", {}),
    "rot-B1+@4": ("linear_energy_shift_rot", {"rho": 4.0}, "gauss", {}),
    "B0+prior2": ("zero", {}, "energy_prior", {"gamma_rel": 2.0}),
    "B0+prior4": ("zero", {}, "energy_prior", {"gamma_rel": 4.0}),
    "B0+prior8": ("zero", {}, "energy_prior", {"gamma_rel": 8.0}),
    "B1+@4+prior4": ("linear_energy_shift", {"rho": 4.0}, "energy_prior", {"gamma_rel": 4.0}),
    **{f"B1 rho={r:g}": ("linear_energy", {"rho": r}, "gauss", {}) for r in (-2, 1, 2, 4)},
    **{f"B1+ rho={r:g}": ("linear_energy_shift", {"rho": r}, "gauss", {}) for r in (2, 4, 8, 16)},
}


@torch.no_grad()
def terminal(base, source, N, steps=200, device="cpu"):
    x = source.sample(N, device)
    for j in range(steps):
        x = base.step(x, j / steps, 1 / steps)
    return x


def lean_adjoint_amp(base, n):
    """max over directions of |a_0| / |a_1| (how much the AM targets are amplified)."""
    eye = torch.eye(n)
    a0 = base.lean_adjoint(eye, torch.zeros(n, 1))
    return float(torch.linalg.matrix_norm(a0, ord=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="+", default=["g0", "g1", "g2", "er:100:0.15:0"])
    ap.add_argument("--groups", nargs="+", default=list(GROUPS))
    ap.add_argument("--kappa_w", type=float, default=2.0)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--out", default="runs/reference_prior.jsonl")
    a = ap.parse_args()
    torch.manual_seed(0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        for g in a.graphs:
            adj = get_graph(g)
            en = MISEnergy(adj, 2.0, a.kappa_w, a.temp, device="cpu")
            dec = Decoder(adj, device="cpu")
            ref_file = Path(f"runs/u1_ref/{g}_k{a.kappa_w:g}_T{a.temp:g}.pt")
            ref = torch.load(ref_file, weights_only=False) if ref_file.exists() else None
            for name in a.groups:
                kind, bkw, skind, skw = GROUPS[name]
                base = build_base(kind, en, sigma=1.0, **{"rho": 0.0, **bkw})
                src = build_source(skind, en, **skw)
                x = terminal(base, src, a.samples)
                row = {"graph": g, "group": name, "amp": round(lean_adjoint_amp(base, en.n), 3),
                       "x1_std": round(float(x.std()), 3),
                       "E_tgt_mean": round(float(en.e_tgt(x).mean()), 2)}
                if ref is not None:
                    d = discrete_metrics(x, ref["p"], ref["mis_codes"], ref["feasible"])
                    row.update({k: round(d[k], 3) for k in
                                ("tv", "p_exact_mis", "entropy", "feasible_frac")})
                else:
                    m = mis_metrics(x[:4096], en, dec)
                    row.update({k: round(float(m[k]), 3) for k in
                                ("raw_conflict_edge_frac", "raw_size_mean", "mis_size_mean")})
                row["base_info"] = base.info
                print(json.dumps(row), flush=True)
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
