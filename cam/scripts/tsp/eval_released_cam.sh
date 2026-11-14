#!/bin/bash
# Test-set evaluation of the released CAM TSP checkpoints (local addition, see cam/UPSTREAM.md):
# huggingface.co/shengyuf/CAM_models, <size> = 500 or 1000, downloaded to $CAM_MODELS
# (default /usr0/home/shengyuf/yunyi/data/CAM_models). Inference as CAM paper App. B: 10 steps,
# 16 samples, greedy merge, then TWO_OPT iterations of 2-opt (paper: 1000; the public scripts leave
# the code default 100). One process, so the per-instance time is comparable. Log and result under
# runs/tsp_cam_eval/tsp<size>_2opt<TWO_OPT>/.
#
#   SIZE=500 TWO_OPT=1000 GPU=3 bash cam/scripts/tsp/eval_released_cam.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source cam/scripts/env.sh
SIZE="${SIZE:?set SIZE=500 or 1000}"; TWO_OPT="${TWO_OPT:-1000}"; GPU="${GPU:?set GPU}"
MODELS="${CAM_MODELS:-/usr0/home/shengyuf/yunyi/data/CAM_models}"
DATA="${CAM_TSP_DIR:-/usr0/home/shengyuf/yunyi/data/CAM_data/TSP}"
CK="$MODELS/tsp_${SIZE}_cam/cam/best.pt"; TEST="$DATA/tsp${SIZE}_test_concorde.txt"
[ -f "$CK" ] && [ -f "$TEST" ] || { echo "error: $CK or $TEST missing" >&2; exit 1; }
OUT="$PWD/runs/tsp_cam_eval/tsp${SIZE}_2opt${TWO_OPT}"; mkdir -p "$OUT"
(CUDA_VISIBLE_DEVICES="$GPU" cam_require_gpu)
cd "$CAM_ROOT"
CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled "$CAM_ACCELERATE" launch --mixed_precision=no \
    --num_processes=1 --num_machines=1 --dynamo_backend=no --main_process_port $((29500 + RANDOM % 1000)) main.py \
    --train_path "$TEST" --valid_path "$TEST" --test_path "$TEST" --problem tsp \
    --sparse_factor 50 --num_t 10 --num_k 16 --num_tp 10 --num_kp 8 --num_h 256 --num_l 12 \
    --method CAM --model AnGNN --two_opt_iterations "$TWO_OPT" --checkpoint "$CK" \
    --save_dir "$OUT" --log_file "$OUT/test_log.txt" --wandb_name tsp_cam_eval --wandb_run_name "tsp${SIZE}_2opt${TWO_OPT}" \
    > "$OUT/run.log" 2>&1
tail -n 3 "$OUT/test_log.txt"
