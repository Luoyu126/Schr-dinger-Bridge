"""Amortized reference-process study on ER graphs (plan U7).

One DIFUSCO gated GCN (8 layers x 128, AMP, activation checkpointing) for u and h, trained
on the whole train split; everything except the reference process / source is shared:
32 graphs x 8 samples per epoch, 25 simulation steps, AM:CM = 30:30 epochs, 50 iterations
of batch 32 per epoch, lr 3e-4, alpha_max 100 * sqrt(n / 100). One stage = 7680 energy
gradient calls for every group (AS runs AM only, at the same calls per stage).

Groups (picked from the single-instance study, runs/main_ref):
  B0          f = 0, sigma = 1 (ASBS)
  AS          Dirac source + analytic corrector, AM only (Adjoint Sampling)
  VP          official VPSDE reference
  B0+prior4   f = 0, energy-prior source, gamma_rel = 4
  iso4m2      isotropic OU matched on E|X_1|^2 / n to B1+ at rho = 4
  B1p_rho2/4  ours: B1+ (M - lambda_min I) at rho = 2 / 4
  AS_B1p_rho2/4  Adjoint Sampling with the B1+ reference instead of f = 0: Dirac source, so
                 any reference is memoryless and the corrector grad log p_1^base is analytic
                 (no CM); differs from AS only in the reference process
rot is left out: a per-graph Haar rotation makes u, h non-functions of the graph.

Validation during training: the 32 val graphs, 16 samples each, N = 25, every 2 stages.
Checkpoints every 4 stages for the test-split evaluation (scripts/eval_amortized.py).

python scripts/sweeps/amortized_er.py --dataset er_200_300 --stages 16 --seeds 0
    -> runs/amortized/er_200_300/jobs_s0.tsv
"""
import argparse
from pathlib import Path

PY = "/usr0/home/shengyuf/yunyi/envs/sb/bin/python"
CALLS_PER_STAGE = 32 * 8 * 30

GROUPS = {
    "B0": "--base zero",
    "AS": "--base zero --source delta --analytic_corrector",
    "VP": "--base vp",
    "B0+prior4": "--base zero --source energy_prior --prior_gamma 4",
    "iso4m2": "--base iso_ou --iso_match_rho 4 --iso_match m2",
    "B1p_rho2": "--base linear_energy_shift --rho 2",
    "B1p_rho4": "--base linear_energy_shift --rho 4",
    "AS_B1p_rho2": "--base linear_energy_shift --rho 2 --source delta --analytic_corrector",
    "AS_B1p_rho4": "--base linear_energy_shift --rho 4 --source delta --analytic_corrector",
}
DEFAULT_GROUPS = ["B0", "AS", "VP", "B0+prior4", "iso4m2", "B1p_rho2", "B1p_rho4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="er_200_300")
    ap.add_argument("--stages", type=int, default=16)
    ap.add_argument("--ckpt_every", type=int, default=4, help="stages between checkpoints")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS, choices=list(GROUPS))
    ap.add_argument("--tsv", default=None, help="job-table file name (default jobs_s<seeds>.tsv)")
    a = ap.parse_args()
    root = Path("runs/amortized") / a.dataset
    root.mkdir(parents=True, exist_ok=True)
    save_at = " ".join(str(CALLS_PER_STAGE * s)
                       for s in range(a.ckpt_every, a.stages + 1, a.ckpt_every))
    common = (f"--dataset {a.dataset} --stages {a.stages} --eval_every 2 --eval_k 16 "
              f"--n_infer_steps 25 --save_at {save_at}")
    lines = []
    for seed in a.seeds:
        for g in a.groups:
            out = root / f"{g}_s{seed}"
            lines.append(f"{a.dataset}__{g}__s{seed}\t{PY} scripts/train_amortized.py {common} "
                         f"{GROUPS[g]} --seed {seed} --out {out}")
    f = root / (a.tsv or f"jobs_s{'_'.join(map(str, a.seeds))}.tsv")
    if f.exists():
        raise SystemExit(f"{f} exists; pass a new --tsv name")
    f.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs -> {f}")


if __name__ == "__main__":
    main()
