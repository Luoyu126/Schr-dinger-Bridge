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
10. `solvers/meta_solver.py`: with `--save_every_epoch`, the untrained weights are also saved as
    `epoch_init.pt` before the first epoch (CAMPrior adds its `.prior.json` with ratio 1).
11. `scripts/env.sh`: `cam_require_gpu` refuses any GPU that runs another user's process
    (previously GPU 1 was refused unconditionally).
12. Added `scripts/mis/test_curve_er700.sh` (test-set curve, CPU) and
    `scripts/mis/large_curve_er700.sh` (ER-[9000-11000] generalisation curve: untrained weights
    and every 10th epoch, 200 steps x 20 samples, several evaluations at a time on one GPU).
    It reads the data from `$CAM_ER700_DIR` and accepts any `<group>_e<E>_s<seed>` run name; the
    untrained weights come from `mis_er700_init2/` if present, else from the run directory.
13. `scripts/env.sh`: the cache directory can be moved with `CAM_CACHE` (for other machines).
14. Added `scripts/mis/seeds_er700.sh` (more seeds of CAM tau0=0.1 and CAMPrior prior tau0=1.0 on
    ER-[700-800], same settings as the seed-0 runs) and `scripts/requirements_env.txt` (pip freeze
    of the env used for all runs). With `SET=ablations` it starts the four CAMPrior ablations.
15. `solvers/cam_prior.py`: ablation switches `--prior_policy {on,off}` (prior in the flip logits
    or not) and `--prior_kl_ref {rlsa,uniform}` (KL to the prior or CAM's entropy term); both
    off reproduce CAM (tests/test_cam_prior.py, test 5). Defaults keep the original behaviour.
16. `solvers/cam_prior.py`: at test time, prior settings that differ from the checkpoint's
    `.prior.json` are an error (previously a warning, so a missing `--prior_d` silently evaluated
    with the default); `--prior_allow_mismatch` keeps the old behaviour on purpose.
17. Added `scripts/mis/mis_rb_cam.sh`: CAM on RB-[200-300] / RB-[800-1200] (`SCALE=small|large`)
    with the flags of the public `mis_rb_small_cam.sh` / `mis_rb_large_cam.sh`, data from
    `$CAM_RB_DIR` (default `/data/shengyuf/MIS`, byte-identical to the HF release), one process per
    GPU named in `CUDA_VISIBLE_DEVICES`, and `MODE=test DEVICE=cpu` for CPU evaluation.
18. Added `scripts/mis/rb_runs.sh` (launches CAM / CAMPrior RB runs with the ER protocol: one GPU
    per run, 1/10 of the official epochs; CAMPrior group names carry the prior's tau0 and d) and
    `scripts/mis/rb_curve.sh` (RB test-set curve on GPU or CPU, retries a failed point at most
    twice and waits between passes instead of retrying at once).
19. Added `solvers/rlsa_distill.py` (`RLSADistill`) and the entry point `distill_rlsa.py`: stage 1
    of the distillation warm start. CAM's network (CAM's own initialisation, flip logits = network
    alone) is fitted to RLSA's per-step flip probabilities, with the forward KL
    KL(Ber(RLSA) || Ber(network)) on RLSA rollouts; the teacher is the CAMPrior prior with
    `--prior_*` (outer factor fixed at 1). Validation is CAM's own inference, and checkpoints are
    plain CAM state_dicts, so stage 2 is `main.py --method CAM ... --checkpoint <best.pt>`.
    `distill_rlsa.py` takes main.py's command line with `--method CAMPrior` and swaps in
    `RLSADistill`; `main.py` itself is unchanged.
20. Added `scripts/mis/mis_er700_distill.sh` (stage-1 launcher: `mis_er700_cam.sh`'s launch block
    with `main.py` replaced by `distill_rlsa.py`) and `scripts/mis/distill_warmstart_er700.sh`
    (both stages per seed on one GPU; stage 2 is the CAM tau0=0.1 command of `seeds_er700.sh`
    plus `--checkpoint`).
21. Added `tests/test_rlsa_distill.py` (CPU, run with `$CAM_PYTHON cam/tests/test_rlsa_distill.py`).

`scripts/config.sh` and the other upstream scripts are unchanged and not used by our runner.
`CAMPrior` runs through the same runner by appending flags, e.g.
`CUDA_VISIBLE_DEVICES=<gpu> EXP=camprior bash cam/scripts/mis/mis_er200_cam.sh --method CAMPrior`.

## Data

`scripts/export_cam_dataset.py` (repo root) converts `data/datasets/<name>/{train,val,test}.npz`
into CAM's gpickle layout under `data/cam/<name>/ER_{train,valid,test}/` (gitignored).
