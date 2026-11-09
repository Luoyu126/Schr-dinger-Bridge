#!/bin/bash
# Stage 1 of the distillation warm start on ER-[700-800] (local addition, see cam/UPSTREAM.md):
# trains CAM's network to reproduce RLSA's per-step flip probabilities (solvers/rlsa_distill.py,
# entry distill_rlsa.py). The launch block below is mis_er700_cam.sh's with main.py replaced by
# distill_rlsa.py (tests/test_rlsa_distill.py checks this), so data, network and optimiser
# settings are CAM's; pass --method CAMPrior and the teacher's --prior_* flags as extra args.
# Training only. Stage 2 is mis_er700_cam.sh with --checkpoint <this run's save_dir>/best.pt.
#
#   CUDA_VISIBLE_DEVICES=0 PROJECT=p EXP=e bash cam/scripts/mis/mis_er700_distill.sh \
#       --method CAMPrior --prior_tau0 1 --epochs 20 --N_anneal 20 --save_every_epoch --seed 0
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cam_require_gpu

export PRECISION=no
export PROJECT="${PROJECT:-mis_er700}"
export EXP="${EXP:-rlsa_distill}"
DATA="${CAM_ER700_DIR:-/data/shengyuf/MIS/data_er}"
SAVE_DIR="$CAM_RUNDIR/$PROJECT/$EXP"
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
PORT=$((29500 + RANDOM % 1000))
MODE_ARGS=(--do_train)

cd "$CAM_ROOT"
nice -n 10 "$CAM_ACCELERATE" launch --mixed_precision=$PRECISION \
    --num_processes=$NPROC --num_machines=1 --dynamo_backend=no --main_process_port $PORT distill_rlsa.py \
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
