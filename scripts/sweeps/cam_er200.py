"""CAM arm of the three-arm study on ER-[200-300] (docs/plan.md §0.8), fast validation.

  CAM_tau0.1        CAM, paper defaults (tau annealed 0.1 -> 0 over training)
  CAM_tau0.05       CAM, the paper's other tau0 (equal tuning budget; also the "lower tau" control)
  CAMPrior_p<t>     CAM with the RLSA energy-flip prior + learned residual, prior tau0 = t
                    (inner RLSA schedule along the steps, outer scale tied to tau_train)
All: official ER hyperparameters of cam/scripts/mis/mis_er200_cam.sh, single GPU, a shortened
budget of --epochs E with the temperature annealed over all of it, weights saved every epoch.

python scripts/sweeps/cam_er200.py --epochs 50 --seeds 0 -> runs/cam/mis_er200/jobs_s0.tsv
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--prior_tau0", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--project", default="mis_er200")
    a = ap.parse_args()
    groups = {"CAM_tau0.1": "--tau0 0.1", "CAM_tau0.05": "--tau0 0.05"}
    for t in a.prior_tau0:
        groups[f"CAMPrior_p{t:g}"] = f"--method CAMPrior --prior_tau0 {t:g}"
    root = Path("runs/cam") / a.project
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for seed in a.seeds:
        for name, extra in groups.items():
            exp = f"{name}_s{seed}"
            lines.append(f"{a.project}__{exp}\tPROJECT={a.project} EXP={exp} bash "
                         f"cam/scripts/mis/mis_er200_cam.sh --epochs {a.epochs} --N_anneal {a.epochs} "
                         f"--save_every_epoch --seed {seed} {extra}")
    f = root / f"jobs_s{'_'.join(map(str, a.seeds))}.tsv"
    if f.exists():
        raise SystemExit(f"{f} exists")
    f.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs -> {f}")


if __name__ == "__main__":
    main()
