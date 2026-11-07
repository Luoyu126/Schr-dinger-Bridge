#!/bin/bash
# Stability pilots for Adjoint Sampling (AS, f = 0) on ER-[200-300] at T = 0.075
# (docs/handoff_as_pilots.md; diagnosis in docs/plan.md §0.8 "连续 v2 批次训崩的诊断").
#
# Round 1 (one GPU each, ~1.5-2 h):
#   P1  100 SDE steps                          (is the time discretisation the cause?)
#   P2  25 steps, per-node clip alpha = 30     (is matching the clip to dt enough?)
#   P3  25 steps, lr 1e-4 + grad-norm clip 1   (is optimiser damping enough?)
#   P4  200 SDE steps
# Round 2 (on the round-1 winner; pass its extra flags in WINNER):
#   P5  winner + --pmap clamp --well out --kappa_w 2   (empty-set trap)
#   P6  winner + --temp 0.25                            (low-T mode seeking)
#
# Usage (from the repo root):
#   PY=/path/to/venv/bin/python GPUS="0 1 2 3" bash scripts/pilots/as_pilots.sh            # round 1
#   PY=... GPUS="0 1" ROUND=2 WINNER="--n_train_steps 100 --n_infer_steps 100 200" \
#       STAGES=3 AM_EPOCHS=15 bash scripts/pilots/as_pilots.sh                              # round 2
# Each run writes runs/pilots/as_v2/<name>/ (log.jsonl, epochs.jsonl, ckpt_*.pt) and
# runs/pilots/as_v2/<name>.out. Summarise with scripts/pilots/summarize_as_pilots.py.
set -euo pipefail
PY="${PY:-python}"
GPUS=(${GPUS:-0 1 2 3})
ROUND="${ROUND:-1}"
OUT=runs/pilots/as_v2
mkdir -p "$OUT" .cache/tmp
export TMPDIR="$PWD/.cache/tmp" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
test -f data/datasets/er_200_300/train.npz || { echo "missing data/datasets/er_200_300 (copy it)" >&2; exit 1; }

COMMON="--dataset er_200_300 --base zero --source delta --analytic_corrector --temp 0.075
  --kappa_w 1 --pmap smooth --clip_mode node --alpha_max 150 --no_amp --lr 3e-4
  --eval_every 1 --eval_k 16 --eval_graphs 16 --seed 0"
PER_EPOCH=256   # grad calls per epoch: 32 graphs x 8 samples

save_at() {  # am_epochs stages -> checkpoint at the end of every stage
    local s out=""; for ((s = 1; s <= $2; s++)); do out+=" $((PER_EPOCH * $1 * s))"; done; echo $out
}

run() {  # gpu name am_epochs stages extra...
    local gpu=$1 name=$2 ame=$3 st=$4; shift 4
    if [ -e "$OUT/$name" ]; then echo "skip $name: $OUT/$name exists" >&2; return; fi
    echo "start $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu nohup nice -n 10 "$PY" scripts/train_amortized.py $COMMON \
        --am_epochs "$ame" --stages "$st" --save_at $(save_at "$ame" "$st") "$@" \
        --out "$OUT/$name" > "$OUT/$name.out" 2>&1 &
}

if [ "$ROUND" = 1 ]; then
    [ ${#GPUS[@]} -ge 4 ] || { echo "round 1 needs 4 GPUs in GPUS" >&2; exit 1; }
    run "${GPUS[0]}" P1_steps100   15 3 --n_train_steps 100 --n_infer_steps 100 200
    run "${GPUS[1]}" P2_alpha30    15 6 --n_train_steps 25 --alpha_max 30 --n_infer_steps 25 200
    run "${GPUS[2]}" P3_lr1e-4_gc1 15 6 --n_train_steps 25 --lr 1e-4 --grad_clip 1 --n_infer_steps 25 200
    run "${GPUS[3]}" P4_steps200   10 3 --n_train_steps 200 --n_infer_steps 200
else
    [ -n "${WINNER:-}" ] || { echo "set WINNER to the round-1 winner's extra flags" >&2; exit 1; }
    [ ${#GPUS[@]} -ge 2 ] || { echo "round 2 needs 2 GPUs in GPUS" >&2; exit 1; }
    ST="${STAGES:-3}"; AME="${AM_EPOCHS:-15}"
    run "${GPUS[0]}" P5_clampout "$AME" "$ST" $WINNER --pmap clamp --well out --kappa_w 2
    run "${GPUS[1]}" P6_T0.25    "$AME" "$ST" $WINNER --temp 0.25
fi
wait
echo "all done"
