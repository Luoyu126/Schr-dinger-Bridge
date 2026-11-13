#!/bin/bash
# Test curve of one MDS run (local addition, see cam/UPSTREAM.md): evaluate epoch k (= epoch_{k-1}.pt,
# k = 0 is epoch_init.pt) and best.pt on the BA test set, 50 steps x 20 chains (the training
# protocol). Points that already have a result are skipped, so the script can be rerun while the
# run is still training; a missing checkpoint is waited for (up to WAIT_H hours).
#
#   GPU=3 SCALE=large EXP=CAMPrior_p1_d20_e100_s0 bash cam/scripts/mds/mds_curve.sh --method CAMPrior --prior_tau0 1 --prior_d 20
#   ... EPOCHS="0 1 2 5 10"  (default 0 1 2 5 10 20 ... 100, then best)  NTEST=8  (smoke test)
#   EPOCHS_TOTAL=100 (best.pt is evaluated once epoch_<EPOCHS_TOTAL-1>.pt exists)
# Pass the run's training flags (method, tau0 / prior_tau0, prior_d); CAMPrior refuses to evaluate
# with settings that differ from its .prior.json. Results: runs/cam/mds_ba_<scale>_test/<EXP>/{epKKK,best}/test_log.txt
# (first line: Avg. obj = mean dominating-set size, lower is better).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
GPU="${GPU:?set GPU}"; SCALE="${SCALE:?set SCALE=small or large}"; EXP="${EXP:?set EXP}"
EPOCHS="${EPOCHS:-0 1 2 5 10 20 30 40 50 60 70 80 90 100 best}"; WAIT_H="${WAIT_H:-96}"; NTEST="${NTEST:--1}"
CK_DIR="$PWD/runs/cam/mds_ba_$SCALE/$EXP"; OUT_DIR="runs/cam/mds_ba_${SCALE}_test/$EXP"
for k in $EPOCHS; do
    if [ "$k" = best ]; then ck=best.pt; out=$OUT_DIR/best
    else ck=epoch_init.pt; [ "$k" -gt 0 ] && ck=$(printf 'epoch_%04d.pt' $((k - 1))); out=$OUT_DIR/ep$(printf '%03d' "$k"); fi
    [ -f "$out/test_log.txt" ] && continue
    waited=0
    # best.pt changes during training: evaluate it only once the last epoch exists
    need=$CK_DIR/$ck; [ "$k" = best ] && need=$CK_DIR/$(printf 'epoch_%04d.pt' $(( ${EPOCHS_TOTAL:-100} - 1 )))
    until [ -f "$need" ]; do
        [ $waited -ge $((WAIT_H*60)) ] && { echo "$(date '+%F %T') $EXP $k: no $need after ${WAIT_H}h, stop"; exit 1; }
        sleep 60; waited=$((waited + 1))
    done
    sleep 30                                             # let a checkpoint being written finish
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SCALE=$SCALE \
        PROJECT=mds_ba_$SCALE EXP=$EXP MODE=test bash cam/scripts/mds/mds_ba_local.sh "$@" --seed 0 \
        --test_samples "$NTEST" --checkpoint "$CK_DIR/$ck" --log_file "$PWD/$out/test_log.txt" > "$out/run.log" 2>&1
    rc=$?
    echo "$(date '+%F %T') $EXP $k exit $rc $(head -1 "$out/test_log.txt" 2>/dev/null)"
    [ $rc -ne 0 ] && rm -f "$out/test_log.txt"
done
echo "$(date '+%F %T') $EXP done"
