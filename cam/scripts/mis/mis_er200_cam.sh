#!/bin/bash
# CAM on our ER-[200-300] dataset (local addition, see cam/UPSTREAM.md).
# Adapted from scripts/mis/mis_er_cam.sh: identical hyperparameters, but a single process
# (no SLURM, no multi-GPU), data from data/cam/er_200_300 (made by scripts/export_cam_dataset.py)
# and checkpoints under runs/cam/<PROJECT>/<EXP>.
#
# Usage (from anywhere):
#   CUDA_VISIBLE_DEVICES=0 bash cam/scripts/mis/mis_er200_cam.sh [extra main.py args]
#   CUDA_VISIBLE_DEVICES=0 MODE=test bash cam/scripts/mis/mis_er200_cam.sh [--checkpoint X.pt]
# Env: PROJECT (default mis_er200_cam), EXP (default cam), MODE=train|test (default train).
# Extra args are appended last, so they override the defaults above them (argparse keeps the
# last value), e.g. --seed 1, --epochs 200 --N_anneal 200.
# In test mode the log is written to $SAVE_DIR/test_log.txt (+ .per_graph.csv) unless
# --log_file is given.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cam_require_gpu

export PRECISION=no
export PROJECT="${PROJECT:-mis_er200_cam}"
export EXP="${EXP:-cam}"
MODE="${MODE:-train}"
DATA="$CAM_DATADIR/er_200_300"
SAVE_DIR="$CAM_RUNDIR/$PROJECT/$EXP"

case "$MODE" in
    train) MODE_ARGS=(--do_train) ;;
    test)  MODE_ARGS=(--log_file "$SAVE_DIR/test_log.txt") ;;
    *) echo "error: MODE must be train or test" >&2; exit 1 ;;
esac

cd "$CAM_ROOT"
nice -n 10 "$CAM_ACCELERATE" launch --mixed_precision=$PRECISION \
    --num_processes=1 --num_machines=1 --dynamo_backend=no main.py \
    --train_path "$DATA/ER_train" \
    --valid_path "$DATA/ER_valid" \
    --test_path "$DATA/ER_test" \
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
