"""CAM arm on ER-[700-800] (CAM's main benchmark; docs/plan.md §0.8).

Same five groups as scripts/sweeps/cam_er200.py (CAM tau0 0.1 / 0.05, CAMPrior prior tau0
0.01 / 0.1 / 1.0), official ER hyperparameters (cam/scripts/mis/mis_er700_cam.sh), one GPU per
run, --epochs E with the temperature annealing and the cosine learning-rate schedule spanning
all E epochs, validation and a checkpoint every epoch. mini_batch 512 needs ~10.2 GB on a
2080 Ti, so the allocator runs with expandable segments.

python scripts/sweeps/cam_er700.py --epochs 100 --seeds 0 -> runs/cam/mis_er700/jobs_e100_s0.tsv
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--prior_tau0", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--project", default="mis_er700")
    a = ap.parse_args()
    groups = {"CAM_tau0.1": "--tau0 0.1", "CAM_tau0.05": "--tau0 0.05"}
    for t in a.prior_tau0:
        groups[f"CAMPrior_p{t:g}"] = f"--method CAMPrior --prior_tau0 {t:g}"
    root = Path("runs/cam") / a.project
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for seed in a.seeds:
        for name, extra in groups.items():
            exp = f"{name}_e{a.epochs}_s{seed}"
            lines.append(f"{a.project}__{exp}\tPYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                         f"PROJECT={a.project} EXP={exp} bash cam/scripts/mis/mis_er700_cam.sh "
                         f"--epochs {a.epochs} --N_anneal {a.epochs} --save_every_epoch "
                         f"--seed {seed} {extra}")
    f = root / f"jobs_e{a.epochs}_s{'_'.join(map(str, a.seeds))}.tsv"
    if f.exists():
        raise SystemExit(f"{f} exists")
    f.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs -> {f}")


if __name__ == "__main__":
    main()
