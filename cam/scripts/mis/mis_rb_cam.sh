#!/bin/bash
# CAM on RB-[200-300] (SCALE=small) or RB-[800-1200] (SCALE=large) from the local copy of the CAM
# data (local addition, see cam/UPSTREAM.md). Hyperparameters identical to the public scripts
# scripts/mis/mis_rb_small_cam.sh and scripts/mis/mis_rb_large_cam.sh; data read-only from
# $CAM_RB_DIR/RB_<scale>_{train,valid,test} (default /data/shengyuf/MIS); checkpoints under
# runs/cam/<PROJECT>/<EXP>. Later arguments override the defaults below (argparse keeps the last).
#
#   SCALE=small CUDA_VISIBLE_DEVICES=4 PROJECT=p EXP=e bash cam/scripts/mis/mis_rb_cam.sh [extra args]
#   MODE=test ...                                      (evaluate best.pt or --checkpoint X.pt)
#   MODE=test DEVICE=cpu ...                           (evaluate on CPU, OMP_NUM_THREADS threads)
# The number of processes is the number of GPUs in CUDA_VISIBLE_DEVICES: name exactly one GPU per
# run, since more GPUs means data parallel training with a larger batch (a different protocol).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

export PRECISION=no
SCALE="${SCALE:?set SCALE=small or SCALE=large}"
export PROJECT="${PROJECT:-mis_rb_$SCALE}"
export EXP="${EXP:-cam}"
MODE="${MODE:-train}"
DATA="${CAM_RB_DIR:-/data/shengyuf/MIS}"
SAVE_DIR="$CAM_RUNDIR/$PROJECT/$EXP"
if [ "${DEVICE:-gpu}" = cpu ]; then
    [ "$MODE" = test ] || { echo "error: DEVICE=cpu is for MODE=test only" >&2; exit 1; }
    export CUDA_VISIBLE_DEVICES=""; NPROC=1
else
    cam_require_gpu
    NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
fi
PORT=$((29500 + RANDOM % 1000))

case "$SCALE" in
    # scripts/mis/mis_rb_small_cam.sh
    small) SCALE_ARGS=(--samples_per_epoch 4000 --batch_size 4 --mini_batch_size 256 --num_kp 10
                       --num_l 8 --epochs 1000 --N_anneal 1000 --patience 200) ;;
    # scripts/mis/mis_rb_large_cam.sh
    large) SCALE_ARGS=(--samples_per_epoch 1000 --batch_size 1 --mini_batch_size 128 --num_kp 20
                       --num_l 6 --tau0 0.1 --epochs 1500 --N_anneal 1500 --patience 400) ;;
    *) echo "error: SCALE must be small or large" >&2; exit 1 ;;
esac
case "$MODE" in
    train) MODE_ARGS=(--do_train) ;;
    test)  MODE_ARGS=(--log_file "$SAVE_DIR/test_log.txt") ;;
    *) echo "error: MODE must be train or test" >&2; exit 1 ;;
esac

cd "$CAM_ROOT"
nice -n 10 "$CAM_ACCELERATE" launch --mixed_precision=$PRECISION \
    --num_processes=$NPROC --num_machines=1 --dynamo_backend=no --main_process_port $PORT main.py \
    --train_path "$DATA/RB_${SCALE}_train" \
    --valid_path "$DATA/RB_${SCALE}_valid" \
    --test_path "$DATA/RB_${SCALE}_test" \
    --problem mis \
    --valid_sample 64 \
    --num_t 50 \
    --num_k 20 \
    --num_tp 50 \
    --num_h 64 \
    --method CAM \
    --model EncodeProcessDecode \
    --lr 0.0005 \
    "${SCALE_ARGS[@]}" \
    --save_dir "$SAVE_DIR" \
    --wandb_name "$PROJECT" \
    --wandb_run_name "$EXP" \
    "${MODE_ARGS[@]}" "$@"
