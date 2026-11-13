#!/bin/bash
# CAM vs CAMPrior on RB-[200-300] and RB-[800-1200] (local addition, see cam/UPSTREAM.md), with the
# protocol of the ER-[700-800] runs: one GPU per run, 1/10 of the official epochs (RB-small 100,
# RB-large 150) with the temperature annealing and the cosine schedule over the shortened run,
# validation and a checkpoint every epoch, official hyperparameters otherwise (mis_rb_cam.sh).
# One run per GPU, detached; the training log goes to runs/queue_cam_rb/<problem>_rb_<scale>__<exp>.log,
# checkpoints to runs/cam/<problem>_rb_<scale>/<exp>/. PROBLEM=mis (default) or mcl (maximum clique
# on the same graphs, envs/mcl.py).
#
# Groups (the run name carries every flag that differs, so the evaluation scripts can rebuild them):
#   CAM_tau0.1                 --tau0 0.1
#   CAMPrior_p<T>_d<D>         --method CAMPrior --prior_tau0 <T> --prior_d <D>
#   <group>_reg<L>_D<N>        the same plus CAM App. C.1's update-magnitude penalty
#                              --lambd <L> --num_d <N> (L * (expected flips per step - N)^2; training only)
#
#   GPUS="4 5 6 7" SCALES="small large" RUN_GROUPS="CAM_tau0.1 CAMPrior_p1_d5" DRY=1 bash cam/scripts/mis/rb_runs.sh
#
# Runs are assigned to GPUS in the order (scale, group, seed). An existing run directory is never
# reused: the script stops instead.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
SCALES=(${SCALES:-small large}); GROUPS_=(${RUN_GROUPS:?set RUN_GROUPS, e.g. RUN_GROUPS=\"CAM_tau0.1 CAMPrior_p1_d5\"})
SEEDS=(${SEEDS:-0}); GPUS=(${GPUS:?set GPUS, e.g. GPUS=\"4 5 6 7\"}); DRY="${DRY:-0}"
PROBLEM="${PROBLEM:-mis}"
LOGDIR=runs/queue_cam_rb; mkdir -p "$LOGDIR"

epochs_of() { case "$1" in small) echo "${EPOCHS_SMALL:-100}" ;; large) echo "${EPOCHS_LARGE:-150}" ;; esac; }
flags_of() {   # group name -> main.py flags
    local g=$1 reg=""
    if [[ "$g" =~ ^(.+)_reg([0-9.]+)_D([0-9]+)$ ]]; then
        g=${BASH_REMATCH[1]}; reg=" --lambd ${BASH_REMATCH[2]} --num_d ${BASH_REMATCH[3]}"
    fi
    local base
    base=$(base_flags_of "$g") || return 1
    echo "$base$reg"
}
base_flags_of() {
    local g=$1
    if [ "$g" = CAM_tau0.1 ]; then echo "--tau0 0.1"
    elif [[ "$g" =~ ^CAMPrior_p([0-9.]+)_d([0-9]+)$ ]]; then
        echo "--method CAMPrior --prior_tau0 ${BASH_REMATCH[1]} --prior_d ${BASH_REMATCH[2]}"
    else echo "error: unknown group $g" >&2; return 1; fi
}

jobs=()
for sc in "${SCALES[@]}"; do
    case "$sc" in small|large) ;; *) echo "error: scale must be small or large" >&2; exit 1 ;; esac
    for g in "${GROUPS_[@]}"; do flags_of "$g" > /dev/null; for s in "${SEEDS[@]}"; do jobs+=("$sc|$g|$s"); done; done
done
if [ ${#jobs[@]} -gt ${#GPUS[@]} ]; then
    echo "error: ${#jobs[@]} runs but only ${#GPUS[@]} GPUs in GPUS" >&2; exit 1
fi
for j in "${jobs[@]}"; do
    IFS="|" read -r sc g s <<< "$j"
    exp="${g}_e$(epochs_of "$sc")_s${s}"
    if [ -e "runs/cam/${PROBLEM}_rb_$sc/$exp" ]; then echo "error: runs/cam/${PROBLEM}_rb_$sc/$exp exists" >&2; exit 1; fi
done

i=0
for j in "${jobs[@]}"; do
    IFS="|" read -r sc g s <<< "$j"
    e=$(epochs_of "$sc"); exp="${g}_e${e}_s${s}"; gpu="${GPUS[$i]}"; i=$((i + 1))
    log="$LOGDIR/${PROBLEM}_rb_${sc}__$exp.log"
    cmd="CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PROBLEM=$PROBLEM SCALE=$sc PROJECT=${PROBLEM}_rb_$sc EXP=$exp bash cam/scripts/mis/mis_rb_cam.sh --epochs $e --N_anneal $e --save_every_epoch --seed $s $(flags_of "$g")"
    echo "GPU $gpu  RB-$sc  $exp"
    echo "    $cmd"
    if [ "$DRY" != 1 ]; then
        nohup bash -c "$cmd" > "$log" 2>&1 < /dev/null &
        echo "    pid $!  log $log"
    fi
done
