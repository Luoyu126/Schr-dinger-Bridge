#!/bin/bash
# Distillation warm start on ER-[700-800] (local addition, see cam/UPSTREAM.md): the conventional
# warm start of CAM from RLSA, as a baseline for CAMPrior (docs/status.md). One GPU per seed, the
# two stages run back to back on it:
#   1. RLSA_distill_p1_s<seed>: CAM's network fitted to the per-step flip probabilities of RLSA
#      with prior tau0 = 1 (the sampler the CAMPrior p1 runs start from), DISTILL_EPOCHS epochs
#      (default 20) of 1000 training graphs; best.pt = best validation score of the distilled
#      CAM sampler (mis_er700_distill.sh, solvers/rlsa_distill.py).
#   2. CAM_tau0.1_distill_e100_s<seed>: plain CAM with exactly the CAM tau0=0.1 command of
#      seeds_er700.sh (100 epochs, annealing and cosine schedule over 100 epochs, a checkpoint
#      every epoch), plus --checkpoint <stage-1 best.pt> (weights only; RLSA is in neither the
#      policy nor the objective). It starts only if stage 1 finished without error.
# Logs: runs/queue_cam_er700/mis_er700__<exp>.log for each stage.
#
#   GPUS="0 1 2" SEEDS="0 1 2" bash cam/scripts/mis/distill_warmstart_er700.sh
#   ... DRY=1 ...            print the commands only
#   ... STAGE=1 ...          stage 1 only (e.g. a pilot); STAGE=2 runs stage 2 from an existing stage 1
#
# Runs are assigned to GPUS in the order of SEEDS. An existing run directory is never reused.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
SEEDS=(${SEEDS:-0}); GPUS=(${GPUS:?set GPUS, e.g. GPUS=\"0 1 2\"})
EPOCHS="${EPOCHS:-100}"; DISTILL_EPOCHS="${DISTILL_EPOCHS:-20}"; DRY="${DRY:-0}"; STAGE="${STAGE:-both}"
case "$STAGE" in both|1|2) ;; *) echo "error: STAGE must be both, 1 or 2" >&2; exit 1 ;; esac
LOGDIR=runs/queue_cam_er700; mkdir -p "$LOGDIR"
RUNS=runs/cam/mis_er700

if [ ${#SEEDS[@]} -gt ${#GPUS[@]} ]; then
    echo "error: ${#SEEDS[@]} seeds but only ${#GPUS[@]} GPUs in GPUS" >&2; exit 1
fi
for s in "${SEEDS[@]}"; do
    d1="RLSA_distill_p1_s$s"; d2="CAM_tau0.1_distill_e${EPOCHS}_s$s"
    if [ "$STAGE" != 2 ] && [ -e "$RUNS/$d1" ]; then echo "error: $RUNS/$d1 exists" >&2; exit 1; fi
    if [ "$STAGE" != 1 ] && [ -e "$RUNS/$d2" ]; then echo "error: $RUNS/$d2 exists" >&2; exit 1; fi
    if [ "$STAGE" = 2 ] && [ ! -f "$RUNS/$d1/best.pt" ]; then echo "error: $RUNS/$d1/best.pt missing" >&2; exit 1; fi
done

i=0
for s in "${SEEDS[@]}"; do
    gpu="${GPUS[$i]}"; i=$((i + 1))
    d1="RLSA_distill_p1_s$s"; d2="CAM_tau0.1_distill_e${EPOCHS}_s$s"
    env_="CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PROJECT=mis_er700"
    cmd1="$env_ EXP=$d1 bash cam/scripts/mis/mis_er700_distill.sh --method CAMPrior --prior_tau0 1 --epochs $DISTILL_EPOCHS --N_anneal $DISTILL_EPOCHS --save_every_epoch --seed $s"
    cmd2="$env_ EXP=$d2 bash cam/scripts/mis/mis_er700_cam.sh --epochs $EPOCHS --N_anneal $EPOCHS --save_every_epoch --seed $s --tau0 0.1 --checkpoint '$PWD/$RUNS/$d1/best.pt'"
    log1="$LOGDIR/mis_er700__$d1.log"; log2="$LOGDIR/mis_er700__$d2.log"
    case "$STAGE" in
        both) cmd="$cmd1 > '$log1' 2>&1 && $cmd2 > '$log2' 2>&1" ;;
        1) cmd="$cmd1 > '$log1' 2>&1" ;;
        2) cmd="$cmd2 > '$log2' 2>&1" ;;
    esac
    echo "GPU $gpu  seed $s"
    if [ "$STAGE" != 2 ]; then echo "    stage 1: $cmd1"; fi
    if [ "$STAGE" != 1 ]; then echo "    stage 2: $cmd2"; fi
    if [ "$DRY" != 1 ]; then
        nohup bash -c "$cmd" > /dev/null 2>&1 < /dev/null &
        echo "    pid $!"
        if [ "$STAGE" != 2 ]; then echo "    log $log1"; fi
        if [ "$STAGE" != 1 ]; then echo "    log $log2"; fi
    fi
done
