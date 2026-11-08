#!/bin/bash
# More seeds of the two chosen ER-[700-800] groups (local addition, see cam/UPSTREAM.md):
# CAM tau0=0.1 and CAMPrior with prior tau0=1.0, each exactly as its seed-0 run
# (runs/cam/mis_er700/jobs_e100_s0.tsv: one GPU per run, 100 epochs with the annealing and the
# cosine schedule over 100 epochs, validation and a checkpoint every epoch). One run per GPU,
# detached; the training log goes to runs/queue_cam_er700/mis_er700__<exp>.log.
#
#   GPUS="0 1 2 3" SEEDS="1 2" bash cam/scripts/mis/seeds_er700.sh            # start the 4 runs
#   GPUS="0 1 2 3" SEEDS="1 2" DRY=1 bash cam/scripts/mis/seeds_er700.sh      # print only
#
# Runs are assigned to GPUS in the order (seed, group). An existing run directory is never
# reused: the script stops instead.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
SEEDS=(${SEEDS:-1 2}); GPUS=(${GPUS:?set GPUS, e.g. GPUS=\"0 1 2 3\"})
EPOCHS="${EPOCHS:-100}"; DRY="${DRY:-0}"
GROUPS_=("CAM_tau0.1|--tau0 0.1" "CAMPrior_p1|--method CAMPrior --prior_tau0 1")
LOGDIR=runs/queue_cam_er700; mkdir -p "$LOGDIR"

jobs=()
for s in "${SEEDS[@]}"; do for gx in "${GROUPS_[@]}"; do jobs+=("$s|$gx"); done; done
if [ ${#jobs[@]} -gt ${#GPUS[@]} ]; then
    echo "error: ${#jobs[@]} runs but only ${#GPUS[@]} GPUs in GPUS" >&2; exit 1
fi
for j in "${jobs[@]}"; do
    IFS="|" read -r s g extra <<< "$j"
    exp="${g}_e${EPOCHS}_s${s}"
    if [ -e "runs/cam/mis_er700/$exp" ]; then echo "error: runs/cam/mis_er700/$exp exists" >&2; exit 1; fi
done

i=0
for j in "${jobs[@]}"; do
    IFS="|" read -r s g extra <<< "$j"
    exp="${g}_e${EPOCHS}_s${s}"; gpu="${GPUS[$i]}"; i=$((i + 1))
    log="$LOGDIR/mis_er700__$exp.log"
    cmd="CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PROJECT=mis_er700 EXP=$exp bash cam/scripts/mis/mis_er700_cam.sh --epochs $EPOCHS --N_anneal $EPOCHS --save_every_epoch --seed $s $extra"
    echo "GPU $gpu  $exp"
    echo "    $cmd"
    if [ "$DRY" != 1 ]; then
        nohup bash -c "$cmd" > "$log" 2>&1 < /dev/null &
        echo "    pid $!  log $log"
    fi
done
