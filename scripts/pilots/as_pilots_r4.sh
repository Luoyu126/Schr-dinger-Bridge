#!/bin/bash
# Round 4 of the AS pilots: temperature annealing on the energy's own scale. Round 3 started
# at T = 1, where the per-node conflict force (~ 0.75 (lambda d / 2 - 1) ~ 27 at x = 0) still
# dwarfs the reference's (~ 1), and collapsed while T was still ~ 0.6-1. Annealed Training
# (arXiv 2207.11542) starts at the energy's Lipschitz constant. Here T goes 50 -> 0.075
# geometrically (equal epochs per factor of T), then stays at 0.075:
#   H1  f = 0 (Brownian), anneal over 30 AM epochs, 5 x 9 stages (eval every 5 epochs:
#       T ~ 16.9, 5.6, 1.9, 0.65, 0.22, then 0.075)
#   H2  as H1 with the energy-informed reference (linear_energy_shift, rho 2)
#   H3  as H2, anneal over 60 AM epochs, 15 x 5 stages
#   H4  as H1, anneal over 60 AM epochs, 15 x 5 stages
# Everything else is P1 (AS, Dirac source, 100 steps, smooth map, kappa_w 1, T 0.075).
#
# Usage (from the repo root):
#   PY=/path/to/venv/bin/python GPUS="6 7 8 9" bash scripts/pilots/as_pilots_r4.sh
# Outputs in runs/pilots/as_v2/ next to rounds 1-3.
set -euo pipefail
PY="${PY:-python}"
GPUS=(${GPUS:-0 1 2 3})
OUT=runs/pilots/as_v2
mkdir -p "$OUT" .cache/tmp
export TMPDIR="$PWD/.cache/tmp" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
test -f data/datasets/er_200_300/train.npz || { echo "missing data/datasets/er_200_300 (copy it)" >&2; exit 1; }
[ ${#GPUS[@]} -ge 4 ] || { echo "round 4 needs 4 GPUs in GPUS" >&2; exit 1; }

COMMON="--dataset er_200_300 --base zero --source delta --analytic_corrector --temp 0.075
  --kappa_w 1 --pmap smooth --clip_mode node --alpha_max 150 --no_amp --lr 3e-4
  --eval_every 1 --eval_k 16 --eval_graphs 16 --seed 0
  --n_train_steps 100 --n_infer_steps 100 200
  --temp_start 50 --temp_schedule geometric"
B1P="--base linear_energy_shift --rho 2"
PER_EPOCH=256

run() {  # gpu name am_epochs stages anneal_epochs extra...
    local gpu=$1 name=$2 ame=$3 st=$4 ann=$5 s save=""; shift 5
    for ((s = 1; s <= st; s++)); do save+=" $((PER_EPOCH * ame * s))"; done
    if [ -e "$OUT/$name" ]; then echo "skip $name: $OUT/$name exists" >&2; return; fi
    echo "start $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu nohup nice -n 10 "$PY" scripts/train_amortized.py $COMMON \
        --am_epochs "$ame" --stages "$st" --anneal_epochs "$ann" --save_at $save "$@" \
        --out "$OUT/$name" > "$OUT/$name.out" 2>&1 &
}

run "${GPUS[0]}" H1_Tgeo50            5  9 30
run "${GPUS[1]}" H2_Tgeo50_B1p        5  9 30 $B1P
run "${GPUS[2]}" H3_Tgeo50_B1p_slow  15  5 60 $B1P
run "${GPUS[3]}" H4_Tgeo50_slow      15  5 60
wait
echo "all done"
