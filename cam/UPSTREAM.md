# Vendored CAM code

- Source: https://github.com/Shengyu-Feng/CAM (official code of "Unsupervised Diffusion Solver
  for Combinatorial Optimization via Combinatorial Adjoint Matching", Feng, Suresh and Yang,
  ICML 2026).
- Upstream commit: `b8c6167add0b77b54c3013252f0c93a6c298cb9a` ("FM schedule", 2026-10-27).
- Vendored on 2026-11-06 from a read-only clone of that commit, all files except `.git/`.
- License: the upstream repository has no LICENSE file. Used with the author's permission.

## Local modifications

Everything else is byte-identical to the upstream commit (`diff -r <upstream clone> cam`
shows only the items below).

1. `cam.png` (README figure) not copied, so the image link in `README.md` is broken.
2. `main.py`: when `--do_train` and `--save_dir` already holds `*.pt`, upstream asks for
   confirmation with `input()`, which hangs unattended runs. It now exits at once with an error
   unless `CAM_ALLOW_OVERWRITE=1` is set.
3. `solvers/meta_solver.py`, `MetaSolver.evaluate`: also records the dataset index of each
   evaluated graph, and when a log file is given (test mode) writes
   `<log_file>.per_graph.csv` with columns `idx,file,obj,time` (objective = -MIS size, time in
   seconds, same values that upstream averages). Averages, validation and training are unchanged.
4. Added `scripts/env.sh`: paths to the existing conda env `~/anaconda3/envs/CAM` (used
   read-only), data/run dirs, and every cache/config/temp dir pointed under
   `/usr0/home/shengyuf/yunyi/.cache/cam`, wandb disabled, no bytecode files.
5. Added `scripts/mis/mis_er200_cam.sh`: `scripts/mis/mis_er_cam.sh` with the same
   hyperparameters, run as one process (no SLURM) on our ER-[200-300] data in `data/cam/` and
   saving to `runs/cam/<PROJECT>/<EXP>`.

6. Added `solvers/cam_prior.py`: `CAMPrior` (`--method CAMPrior`), CAM with an RLSA energy prior
   as the base process. Flip logit = RLSA's flip logit (same gradient, top-d threshold and inner
   temperature schedule as `solvers/rlsa.py`) + the network output as a residual, with the output
   head zero-initialised so that the untrained sampler is RLSA. The prior temperature is also
   scaled by tau_train/tau_train(0) (`--prior_fixed` turns this off), and CAM's entropy term
   becomes tau * KL(Ber(p) || Ber(p_prior)). The module docstring has the formulas and the choice
   of kappa. Each checkpoint gets a `<ckpt>.prior.json` sidecar with the prior scale it was
   trained with, which test mode reads back. New flags: `--prior_*`, `--residual_init`.
7. `main.py`: registers `--method CAMPrior` and its flags (three added lines). All flag names are
   new, and every flag used in `scripts/` still resolves as before under argparse prefix matching.
8. Added `tests/test_cam_prior.py` (CPU, run with `$CAM_PYTHON cam/tests/test_cam_prior.py`).
9. `solvers/meta_solver.py` (unsupervised training loop) and `main.py`: `--save_every_epoch` also
   saves `epoch_XXXX.pt` every epoch (CAMPrior writes its `.prior.json` sidecar next to each), so
   training curves can be re-evaluated offline. Default off; `best.pt` handling is unchanged.

`scripts/config.sh` and the other upstream scripts are unchanged and not used by our runner.
`CAMPrior` runs through the same runner by appending flags, e.g.
`CUDA_VISIBLE_DEVICES=<gpu> EXP=camprior bash cam/scripts/mis/mis_er200_cam.sh --method CAMPrior`.

## Data

`scripts/export_cam_dataset.py` (repo root) converts `data/datasets/<name>/{train,val,test}.npz`
into CAM's gpickle layout under `data/cam/<name>/ER_{train,valid,test}/` (gitignored).
