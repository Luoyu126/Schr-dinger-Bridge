"""Continuous arms of the three-arm study (docs/plan.md §0.8), fast-validation protocol.

AS arm   : Adjoint Sampling (Dirac source, analytic corrector, AM only)
             AS            f = 0
             AS_B1p_rho<r> f = B1+ energy drift
ASBS arm : ASBS (Gaussian source, learned corrector), AM first as in the official code
             B0            f = 0
             B1p_rho<r>    f = B1+ energy drift
             iso<r>m2      isotropic OU matched on E|X_1|^2 / n to B1+ at the same rho
             B0+prior<g>   f = 0, energy-prior source (ASBS-style: energy information in the source)

Shared (tier-1 alignment with the AS / ASBS papers, plan §0.8): fixed low temperature with the
bounded smooth p(x) encoding, per-node clipping of grad E / T at 150, fp32. Training cadence
(batch, reuse, steps, AM:CM) is kept from the first batch (tier 2 deferred). A checkpoint is
saved every stage so any evaluation protocol can be applied offline later.

python scripts/sweeps/amortized_v2.py --rho 2 --gamma 4 --seeds 0
    -> runs/amortized/er_200_300_v2/jobs_s0.tsv
"""
import argparse
from pathlib import Path

PY = "/usr0/home/shengyuf/yunyi/envs/sb/bin/python"
CALLS_PER_STAGE = 32 * 8 * 30


def groups(rho, gamma):
    r, g = f"{rho:g}", f"{gamma:g}"
    asbs = "--init am_first"
    return {
        "AS": "--base zero --source delta --analytic_corrector",
        f"AS_B1p_rho{r}": f"--base linear_energy_shift --rho {r} --source delta --analytic_corrector",
        "B0": f"--base zero {asbs}",
        f"B1p_rho{r}": f"--base linear_energy_shift --rho {r} {asbs}",
        f"iso{r}m2": f"--base iso_ou --iso_match_rho {r} --iso_match m2 {asbs}",
        f"B0+prior{g}": f"--base zero --source energy_prior --prior_gamma {g} {asbs}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="er_200_300")
    ap.add_argument("--tag", default="v2", help="output root runs/amortized/<dataset>_<tag>")
    ap.add_argument("--stages", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.075)
    ap.add_argument("--kappa_w", type=float, default=1.0)
    ap.add_argument("--pmap", default="smooth")
    ap.add_argument("--alpha_max", type=float, default=150.0)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--groups", nargs="+", default=None)
    a = ap.parse_args()
    G = groups(a.rho, a.gamma)
    names = a.groups or list(G)
    root = Path("runs/amortized") / f"{a.dataset}_{a.tag}"
    root.mkdir(parents=True, exist_ok=True)
    save_at = " ".join(str(CALLS_PER_STAGE * s) for s in range(1, a.stages + 1))
    common = (f"--dataset {a.dataset} --stages {a.stages} --eval_every 1 --eval_k 16 "
              f"--n_infer_steps 25 --temp {a.temp:g} --kappa_w {a.kappa_w:g} --pmap {a.pmap} "
              f"--clip_mode node --alpha_max {a.alpha_max:g} --no_amp --save_at {save_at}")
    lines = []
    for seed in a.seeds:
        for name in names:
            out = root / f"{name}_s{seed}"
            lines.append(f"{a.dataset}_{a.tag}__{name}__s{seed}\t{PY} scripts/train_amortized.py "
                         f"{common} {G[name]} --seed {seed} --out {out}")
    f = root / f"jobs_s{'_'.join(map(str, a.seeds))}.tsv"
    if f.exists():
        raise SystemExit(f"{f} exists")
    f.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs -> {f}")


if __name__ == "__main__":
    main()
