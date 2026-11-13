#!/bin/bash
# Test of docs/label_correction.md 7.5 on ER-[700-800] (local addition, see cam/UPSTREAM.md):
# continue a trained CAMPrior run (prior tau0 = 1) from epoch FROM (checkpoint epoch_<FROM-1>.pt of
# seed FROM_SEED) to epoch TO with one of three training labels. Schedules (temperature anneal,
# cosine learning rate) still run over 100 epochs, so the continuation sits where the original run
# was; optimizer moments start fresh in every arm.
#   ARM=cam     CAM's label on all nodes (continuation baseline)
#   ARM=camsub  CAM's label on the same subsample as simpol (control for label density)
#   ARM=simpol  on-policy simulated label (rollouts of prior + network, soft value with tau * KL;
#               KL=off drops the tau * KL cost and keeps the end-energy difference)
# Subsample: round(FRAC * N) nodes (FRAC = 0.00267: 2 nodes on 700-800-node graphs) in a fraction
# STATES of the stored states per graph. The training log goes to
# runs/queue_cam_er700/mis_er700__<exp>.log; an existing run directory is never reused.
#
#   ARM=simpol GPU=2 bash cam/scripts/mis/simpol_er700.sh
#   ... FROM_SEED=1 FROM=50 TO=60 FRAC=0.00267 STATES=0.1 KL=on DRY=1
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source cam/scripts/env.sh
ARM="${ARM:?set ARM=cam, camsub or simpol}"; GPU="${GPU:?set GPU}"
FROM_SEED="${FROM_SEED:-1}"; FROM="${FROM:-50}"; TO="${TO:-60}"
FRAC="${FRAC:-0.00267}"; STATES="${STATES:-0.1}"; KL="${KL:-on}"; DRY="${DRY:-0}"
SRC="runs/cam/mis_er700/CAMPrior_p1_e100_s${FROM_SEED}"
CK="$PWD/$SRC/$(printf 'epoch_%04d.pt' $((FROM - 1)))"
[ -f "$CK" ] || { echo "error: $CK not found" >&2; exit 1; }
case "$ARM" in
    cam)    EXTRA=(); tag=camfull ;;
    camsub) EXTRA=(--label_mode cam --label_frac "$FRAC" --label_states "$STATES"); tag="camsub_f${FRAC}_s${STATES}" ;;
    simpol) EXTRA=(--label_mode simpol --label_frac "$FRAC" --label_states "$STATES" --simpol_chunk 32 --simpol_kl "$KL")
            tag="simpol_f${FRAC}_s${STATES}"; [ "$KL" = on ] || tag="${tag}_kl${KL}" ;;
    *) echo "error: ARM must be cam, camsub or simpol" >&2; exit 1 ;;
esac
exp="CAMPrior_p1_s${FROM_SEED}_from${FROM}_${tag}"
(CUDA_VISIBLE_DEVICES="$GPU" cam_require_gpu)
if [ -e "runs/cam/mis_er700/$exp" ]; then echo "error: runs/cam/mis_er700/$exp exists" >&2; exit 1; fi
LOGDIR=runs/queue_cam_er700; mkdir -p "$LOGDIR"
log="$LOGDIR/mis_er700__$exp.log"
echo "GPU $GPU  $exp  (from $CK, epochs $FROM..$((TO - 1)))"
if [ "$DRY" != 1 ]; then
    CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PROJECT=mis_er700 EXP=$exp \
        nohup bash cam/scripts/mis/mis_er700_cam.sh --epochs 100 --N_anneal 100 --save_every_epoch \
        --seed "$FROM_SEED" --method CAMPrior --prior_tau0 1 --checkpoint "$CK" --start_epoch "$FROM" \
        --stop_epoch "$TO" "${EXTRA[@]}" > "$log" 2>&1 < /dev/null &
    echo "    pid $!  log $log"
fi
