#!/bin/bash
# CAM / CAMPrior / RLSA on minimum dominating set, BA-[200-300] (SCALE=small) or BA-[800-1200]
# (SCALE=large), from the MDS.zip files of the CAM HF release (local addition, see cam/UPSTREAM.md).
# The public CAM repository has no MDS script; hyperparameters are those of the MaxCut BA script
# (scripts/mcut/mcut_ba_large_cam.sh), the other BA problem. Checkpoints under runs/cam/<PROJECT>/<EXP>.
# Extra arguments are appended (e.g. --epochs 100 --N_anneal 100 --method CAMPrior --prior_tau0 1).
#
#   CUDA_VISIBLE_DEVICES=2 SCALE=small PROJECT=mds_ba_small EXP=e bash cam/scripts/mds/mds_ba_local.sh [args]
#   MODE=test ...                                      (evaluate best.pt or --checkpoint X.pt)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cam_require_gpu

export PRECISION=no
SCALE="${SCALE:-small}"
export PROJECT="${PROJECT:-mds_ba_$SCALE}"
export EXP="${EXP:-cam}"
MODE="${MODE:-train}"
DATA="${CAM_MDS_DIR:-/home/shengyuf/yunyi/data/CAM_data/MDS}"
for s in train valid test; do
    if [ ! -d "$DATA/BA_${SCALE}_$s" ]; then echo "error: $DATA/BA_${SCALE}_$s not found (set CAM_MDS_DIR)" >&2; exit 1; fi
done
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
    --train_path "$DATA/BA_${SCALE}_train" \
    --valid_path "$DATA/BA_${SCALE}_valid" \
    --test_path "$DATA/BA_${SCALE}_test" \
    --problem mds \
    --samples_per_epoch 1000 \
    --valid_sample 64 \
    --batch_size 1 \
    --mini_batch_size 256 \
    --num_t 50 \
    --num_k 20 \
    --num_tp 50 \
    --num_kp 20 \
    --num_h 64 \
    --num_l 6 \
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
