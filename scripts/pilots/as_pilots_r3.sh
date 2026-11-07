#!/bin/bash
# Round 3 of the AS pilots (docs/handoff_as_pilots.md): rounds 1-2 were stable at 100 SDE
# steps but every pilot collapsed to the all-off corner (empty set) within 15 epochs. Each
# pilot below is P1 (AS, f = 0, Dirac source, 100 steps, smooth map, kappa_w 1, T 0.075)
# with one change aimed at that trap:
#   R1  training temperature annealed 1.0 -> 0.075 over the first 30 AM epochs
#       (DiffUCO / SDDS / BloGDiT anneal T during training)
#   R2  clamp map, double well only outside [-1, 1] (kappa_w 1), plus gamma * sum_i
#       (1 - (2 p_i - 1)^2) with gamma -20 -> 0 over 30 AM epochs (CRA, PQQA)
#   R3  energy-informed reference: linear_energy_shift (B1+), rho 2 (NAAS / SCLD: energy in
#       the dynamics)
#   R4  R1 + R3
# The annealed pilots end on the true target for the last 15 epochs (stage 3); evaluations
# always use the true E_tgt.
#
# Usage (from the repo root):
#   PY=/path/to/venv/bin/python GPUS="6 7 8 9" bash scripts/pilots/as_pilots_r3.sh
# Outputs next to rounds 1-2 in runs/pilots/as_v2/; summarise with
# scripts/pilots/summarize_as_pilots.py runs/pilots/as_v2.
set -euo pipefail
PY="${PY:-python}"
GPUS=(${GPUS:-0 1 2 3})
OUT=runs/pilots/as_v2
mkdir -p "$OUT" .cache/tmp
export TMPDIR="$PWD/.cache/tmp" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
test -f data/datasets/er_200_300/train.npz || { echo "missing data/datasets/er_200_300 (copy it)" >&2; exit 1; }
[ ${#GPUS[@]} -ge 4 ] || { echo "round 3 needs 4 GPUs in GPUS" >&2; exit 1; }

# same as scripts/pilots/as_pilots.sh, plus P1's step counts
COMMON="--dataset er_200_300 --base zero --source delta --analytic_corrector --temp 0.075
  --kappa_w 1 --pmap smooth --clip_mode node --alpha_max 150 --no_amp --lr 3e-4
  --eval_every 1 --eval_k 16 --eval_graphs 16 --seed 0
  --n_train_steps 100 --n_infer_steps 100 200"
AME=15; ST=3; PER_EPOCH=256
SAVE=""; for ((s = 1; s <= ST; s++)); do SAVE+=" $((PER_EPOCH * AME * s))"; done
ANNEAL="--anneal_epochs 30"

run() {  # gpu name extra...
    local gpu=$1 name=$2; shift 2
    if [ -e "$OUT/$name" ]; then echo "skip $name: $OUT/$name exists" >&2; return; fi
    echo "start $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu nohup nice -n 10 "$PY" scripts/train_amortized.py $COMMON \
        --am_epochs $AME --stages $ST --save_at $SAVE "$@" \
        --out "$OUT/$name" > "$OUT/$name.out" 2>&1 &
}

run "${GPUS[0]}" R1_Tanneal      --temp_start 1.0 $ANNEAL
run "${GPUS[1]}" R2_binanneal    --pmap clamp --well out --bin_gamma_start -20 $ANNEAL
run "${GPUS[2]}" R3_B1p          --base linear_energy_shift --rho 2
run "${GPUS[3]}" R4_Tanneal_B1p  --base linear_energy_shift --rho 2 --temp_start 1.0 $ANNEAL
wait
echo "all done"
