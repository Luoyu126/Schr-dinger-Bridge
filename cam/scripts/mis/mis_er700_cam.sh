#!/bin/bash
# CAM on ER-[700-800] from the local copy of the CAM data (local addition, see cam/UPSTREAM.md).
# Hyperparameters identical to scripts/mis/mis_er_cam.sh (public CAM); data read-only from
# $CAM_ER700_DIR (default /data/shengyuf/MIS/data_er: er_train / er_valid / er_test, the same
# files as the HF release); checkpoints under runs/cam/<PROJECT>/<EXP>.
# Data parallel over all GPUs in CUDA_VISIBLE_DEVICES (as the official 8-GPU runs).
#
#   CUDA_VISIBLE_DEVICES=0 PROJECT=p EXP=e bash cam/scripts/mis/mis_er700_cam.sh [extra args]
#   CUDA_VISIBLE_DEVICES=4,5 ...                       (2-process data parallel)
#   MODE=test ...                                      (evaluate best.pt or --checkpoint X.pt)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cam_require_gpu

export PRECISION=no
export PROJECT="${PROJECT:-mis_er700}"
export EXP="${EXP:-cam}"
MODE="${MODE:-train}"
DATA="${CAM_ER700_DIR:-/data/shengyuf/MIS/data_er}"
SAVE_DIR="$CAM_RUNDIR/$PROJECT/$EXP"
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
PORT=$((29500 + RANDOM % 1000))

case "$MODE" in
    train) MODE_ARGS=(--do_train) ;;
    test)  MODE_ARGS=(--log_file "$SAVE_DIR/test_log.txt") ;;
    *) echo "error: MODE must be train or test" >&2; exit 1 ;;
esac

cd "$CAM_ROOT"
nice -n 10 "$CAM_ACCELERATE" launch --mixed_precision=$PRECISION \
    --num_processes=$NPROC --num_machines=1 --dynamo_backend=no --main_process_port $PORT main.py \
    --train_path "$DATA/er_train" \
    --valid_path "$DATA/er_valid" \
    --test_path "$DATA/er_test" \
    --problem mis \
    --samples_per_epoch 1000 \
    --valid_sample 64 \
    --batch_size 1 \
    --mini_batch_size 512 \
    --num_t 50 \
    --num_k 20 \
    --num_tp 50 \
    --num_kp 20 \
    --num_h 64 \
    --num_l 8 \
    --method CAM \
    --model EncodeProcessDecode \
    --epochs 1000 \
    --N_anneal 1000 \
    --patience 200 \
    --lr 0.0005 \
    --save_dir "$SAVE_DIR" \
    --wandb_name "$PROJECT" \
    --wandb_run_name "$EXP" \
    "${MODE_ARGS[@]}" "$@"
