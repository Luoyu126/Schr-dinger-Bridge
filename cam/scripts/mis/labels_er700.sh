#!/bin/bash
# Training-label ablation on ER-[700-800] (local addition, see cam/UPSTREAM.md; docs/status.md).
# CAMPrior with prior tau0 = 1 and the seed settings of seeds_er700.sh (100 epochs, annealing and
# cosine schedule over 100 epochs, a checkpoint every epoch); only the training label changes:
#   SET=sim     simulated labels (RLSA rollouts, solvers/sim_labels.py) on 10% of the nodes per state;
#               trainer on GPU, label workers on LABEL_GPUS (restarted / dropped on failure)
#   SET=camsub  control: CAM's own labels on the same 10% (same subsampling and loss rescaling)
#   SET=lin     first-order labels (solvers/lin_labels.py, own perturbation only) on all nodes,
#               computed on the trainer's GPU (FRAC is ignored)
# The training log goes to runs/queue_cam_er700/mis_er700__<exp>.log; an existing run directory is
# never reused.
#
#   SET=sim GPU=0 LABEL_GPUS=1,2,3 SEED=0 bash cam/scripts/mis/labels_er700.sh
#   SET=camsub GPU=5 SEED=0 bash cam/scripts/mis/labels_er700.sh
#   SET=lin GPU=6 SEED=0 bash cam/scripts/mis/labels_er700.sh
#   ... DRY=1 ...   (print only)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source cam/scripts/env.sh
SET="${SET:?set SET=sim, camsub or lin}"; GPU="${GPU:?set GPU}"; SEED="${SEED:-0}"
FRAC="${FRAC:-0.1}"; EPOCHS="${EPOCHS:-100}"; DRY="${DRY:-0}"
case "$SET" in
    sim)
        LABEL_GPUS="${LABEL_GPUS:?set LABEL_GPUS, e.g. 1,2,3}"
        (CUDA_VISIBLE_DEVICES="$LABEL_GPUS" cam_require_gpu)
        EXTRA="--label_mode sim --label_frac $FRAC --label_gpus $LABEL_GPUS"
        exp="CAMPrior_p1_simlab${FRAC}_e${EPOCHS}_s${SEED}" ;;
    camsub)
        EXTRA="--label_mode cam --label_frac $FRAC"
        exp="CAMPrior_p1_camlab${FRAC}_e${EPOCHS}_s${SEED}" ;;
    lin)
        EXTRA="--label_mode lin"
        exp="CAMPrior_p1_linlab_e${EPOCHS}_s${SEED}" ;;
    *) echo "error: SET must be sim, camsub or lin" >&2; exit 1 ;;
esac
(CUDA_VISIBLE_DEVICES="$GPU" cam_require_gpu)
if [ -e "runs/cam/mis_er700/$exp" ]; then echo "error: runs/cam/mis_er700/$exp exists" >&2; exit 1; fi
LOGDIR=runs/queue_cam_er700; mkdir -p "$LOGDIR"
log="$LOGDIR/mis_er700__$exp.log"
cmd="CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PROJECT=mis_er700 EXP=$exp bash cam/scripts/mis/mis_er700_cam.sh --epochs $EPOCHS --N_anneal $EPOCHS --save_every_epoch --seed $SEED --method CAMPrior --prior_tau0 1 $EXTRA"
echo "GPU $GPU  $exp"
echo "    $cmd"
if [ "$DRY" != 1 ]; then
    nohup bash -c "$cmd" > "$log" 2>&1 < /dev/null &
    echo "    pid $!  log $log"
fi
