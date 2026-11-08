#!/bin/bash
# Round 5 of the AS pilots: MALA refinement of the AM buffer (--mcmc_steps). The PT reference
# of the target (scripts/pt_ladder.py, 16 val graphs) shows the target is never near-empty
# (>= 14 nodes on before repair at every T in 0.075..24), while the round-4 samplers fall into
# the all-off corner between T ~ 2 and 0.4 and never leave it. Here every AM terminal sample
# takes 1000 MALA steps on exp(-E / T_e) before entering the buffer; with the Dirac source
# MALA leaves the target invariant, so the AM fixed point is unchanged. Single-temperature MALA
# leaves the all-off corner within ~1000 steps at T >= 0.28 but not at T = 0.075.
#   M1  T 50 -> 0.075 geometric over 60 AM epochs, f = 0 (Brownian), refinement always
#   M2  as M1, refinement only while T_e >= 0.3 (does the sampler hold on its own after?)
#   M3  as M1 with the energy-informed reference (linear_energy_shift, rho 2)
#   M4  no annealing (T = 0.075 throughout), refinement always
# 10 AM epochs per stage, 10 stages (evaluations at T ~ 18.9, 6.4, 2.2, 0.73, 0.25, 0.084,
# then four stages at 0.075). Everything else is P1 (AS, Dirac source, 100 steps, smooth map,
# kappa_w 1, T 0.075).
#
# Usage (from the repo root):
#   PY=/path/to/venv/bin/python GPUS="6 7 8 9" bash scripts/pilots/as_pilots_r5.sh
# Outputs in runs/pilots/as_v2/ next to rounds 1-4.
set -euo pipefail
PY="${PY:-python}"
GPUS=(${GPUS:-0 1 2 3})
OUT=runs/pilots/as_v2
mkdir -p "$OUT" .cache/tmp
export TMPDIR="$PWD/.cache/tmp" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
test -f data/datasets/er_200_300/train.npz || { echo "missing data/datasets/er_200_300 (copy it)" >&2; exit 1; }
[ ${#GPUS[@]} -ge 4 ] || { echo "round 5 needs 4 GPUs in GPUS" >&2; exit 1; }

COMMON="--dataset er_200_300 --base zero --source delta --analytic_corrector --temp 0.075
  --kappa_w 1 --pmap smooth --clip_mode node --alpha_max 150 --no_amp --lr 3e-4
  --eval_every 1 --eval_k 16 --eval_graphs 16 --seed 0
  --n_train_steps 100 --n_infer_steps 100 200
  --am_epochs 10 --stages 10 --mcmc_steps 1000"
ANNEAL="--temp_start 50 --temp_schedule geometric --anneal_epochs 60"
B1P="--base linear_energy_shift --rho 2"
PER_EPOCH=256

run() {  # gpu name extra...
    local gpu=$1 name=$2 s save=""; shift 2
    for ((s = 1; s <= 10; s++)); do save+=" $((PER_EPOCH * 10 * s))"; done
    if [ -e "$OUT/$name" ]; then echo "skip $name: $OUT/$name exists" >&2; return; fi
    echo "start $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu nohup nice -n 10 "$PY" scripts/train_amortized.py $COMMON \
        --save_at $save "$@" --out "$OUT/$name" > "$OUT/$name.out" 2>&1 &
}

run "${GPUS[0]}" M1_mcmc_all          $ANNEAL
run "${GPUS[1]}" M2_mcmc_hot          $ANNEAL --mcmc_min_temp 0.3
run "${GPUS[2]}" M3_mcmc_all_B1p      $ANNEAL $B1P
run "${GPUS[3]}" M4_mcmc_noanneal
wait
echo "all done"
