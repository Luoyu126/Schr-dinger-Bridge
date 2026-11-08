#!/bin/bash
# Round 6 of the AS pilots: a fresh AM buffer during annealing. In round 5 the buffer (10k
# samples, 256 new per AM epoch) spanned the last 39 AM epochs, i.e. a factor ~68 in T of the
# geometric 50 -> 0.075 schedule over 60 epochs; dense, conflicting x1 drawn at high T, re-targeted
# with the current low-T gradient, push every node off, and the samplers came out much sparser
# than the PT reference of the target (3.5-9 vs 15 nodes on at T = 0.73). The official ASBS
# configs keep 10k samples but add 1000 per epoch with no annealing (~10 of 300 epochs per
# stage); NAAS (arXiv 2506.18165) refreshes the buffer every 200-500 iterations. Here the
# buffer holds 1024 samples (4 AM epochs, a factor ~1.5 in T); each sample is still used
# ~6 times (50 x 32 draws per epoch). Against round 5's M3 (buffer 10k, 1000 MALA steps, B1+):
#   S1  buffer 1024, 3000 MALA steps, energy-informed reference (linear_energy_shift, rho 2)
#   S2  as S1 with f = 0 (Brownian)
#   S3  as S1 without refinement (buffer freshness alone)
#   S4  as S1 with 1000 MALA steps (buffer change alone, vs M3)
# Schedule as round 5: T 50 -> 0.075 geometric over 60 AM epochs, 10 AM epochs per stage,
# 10 stages. Everything else is P1 (AS, Dirac source, 100 steps, smooth map, kappa_w 1, T 0.075).
#
# Usage (from the repo root):
#   PY=/path/to/venv/bin/python GPUS="6 7 8 9" bash scripts/pilots/as_pilots_r6.sh
# Outputs in runs/pilots/as_v2/ next to rounds 1-5.
set -euo pipefail
PY="${PY:-python}"
GPUS=(${GPUS:-0 1 2 3})
OUT=runs/pilots/as_v2
mkdir -p "$OUT" .cache/tmp
export TMPDIR="$PWD/.cache/tmp" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
test -f data/datasets/er_200_300/train.npz || { echo "missing data/datasets/er_200_300 (copy it)" >&2; exit 1; }
[ ${#GPUS[@]} -ge 4 ] || { echo "round 6 needs 4 GPUs in GPUS" >&2; exit 1; }

COMMON="--dataset er_200_300 --source delta --analytic_corrector --temp 0.075
  --kappa_w 1 --pmap smooth --clip_mode node --alpha_max 150 --no_amp --lr 3e-4
  --eval_every 1 --eval_k 16 --eval_graphs 16 --seed 0
  --n_train_steps 100 --n_infer_steps 100 200
  --am_epochs 10 --stages 10 --buffer_size 1024
  --temp_start 50 --temp_schedule geometric --anneal_epochs 60"
B1P="--base linear_energy_shift --rho 2"
B0="--base zero"
PER_EPOCH=256

run() {  # gpu name extra...
    local gpu=$1 name=$2 s save=""; shift 2
    for ((s = 1; s <= 10; s++)); do save+=" $((PER_EPOCH * 10 * s))"; done
    if [ -e "$OUT/$name" ]; then echo "skip $name: $OUT/$name exists" >&2; return; fi
    echo "start $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu nohup nice -n 10 "$PY" scripts/train_amortized.py $COMMON \
        --save_at $save "$@" --out "$OUT/$name" > "$OUT/$name.out" 2>&1 &
}

run "${GPUS[0]}" S1_buf1k_mcmc3k_B1p   $B1P --mcmc_steps 3000
run "${GPUS[1]}" S2_buf1k_mcmc3k       $B0  --mcmc_steps 3000
run "${GPUS[2]}" S3_buf1k_B1p          $B1P
run "${GPUS[3]}" S4_buf1k_mcmc1k_B1p   $B1P --mcmc_steps 1000
wait
echo "all done"
