#!/bin/bash
# Generalisation curve (local addition, see cam/UPSTREAM.md): evaluates the ER-[700-800] runs on the
# 16 ER-[9000-11000] graphs with CAM's protocol for that benchmark (200 steps x 20 samples), for the
# untrained weights and after every 10th epoch, on one GPU, several evaluations at a time. Loops,
# picking up checkpoints as training writes them, until every target is done.
#   GPU=1 PAR=4 nohup bash cam/scripts/mis/large_curve_er700.sh > runs/cam/mis_er700_large/queue.log 2>&1 &
# The same queue evaluates the 128 ER-[700-800] test graphs (CAM's 50 steps x 20 samples) with
#   SPLIT=test GPU=1 PAR=2 nohup bash cam/scripts/mis/large_curve_er700.sh > runs/cam/mis_er700_test10/queue.log 2>&1 &
# Other runs (e.g. more seeds): EXPS="CAM_tau0.1_e100_s1 CAMPrior_p1_e100_s1" ...
# Epoch k (after k epochs of training) is the checkpoint epoch_{k-1}.pt; k = 0 is epoch_init.pt,
# taken from runs/cam/mis_er700_init2/<exp>/ (rebuilt for the seed-0 runs, which predate it) or
# else from the run directory itself (runs started with --save_every_epoch write it there).
# Data: $CAM_ER700_DIR (default /data/shengyuf/MIS/data_er), as in mis_er700_cam.sh.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
GPU="${GPU:-1}"; PAR="${PAR:-4}"; EVERY="${EVERY:-10}"; LAST="${LAST:-100}"
EXPS=(${EXPS:-CAM_tau0.1_e100_s0 CAM_tau0.05_e100_s0 CAMPrior_p0.01_e100_s0 CAMPrior_p0.1_e100_s0 CAMPrior_p1_e100_s0})
SPLIT="${SPLIT:-large}"
export CAM_ER700_DIR="${CAM_ER700_DIR:-/data/shengyuf/MIS/data_er}"
case "$SPLIT" in
    large) TESTDIR=er_large_test; NUM_T=200; OUT=runs/cam/mis_er700_large ;;
    test)  TESTDIR=er_test;       NUM_T=50;  OUT=runs/cam/mis_er700_test10 ;;
    *) echo "SPLIT must be large or test" >&2; exit 1 ;;
esac
mkdir -p "$OUT"
total=$(( ${#EXPS[@]} * (LAST / EVERY + 1) ))
while :; do
    todo=(); done_n=0
    for k in $(seq 0 "$EVERY" "$LAST"); do for x in "${EXPS[@]}"; do
        log="$OUT/$x/ep$(printf '%03d' "$k")/test_log.txt"
        if [ -f "$log.per_graph.csv" ]; then done_n=$((done_n + 1)); continue; fi
        if [ "$k" = 0 ]; then
            ck="runs/cam/mis_er700_init2/$x/epoch_init.pt"
            [ -f "$ck" ] || ck="runs/cam/mis_er700/$x/epoch_init.pt"
        else ck="runs/cam/mis_er700/$x/$(printf 'epoch_%04d.pt' $((k - 1)))"; fi
        [ -f "$ck" ] && todo+=("$x|$k|$ck|$log")
    done; done
    echo "$(date '+%F %T') done $done_n/$total, ready ${#todo[@]}"
    [ "$done_n" -ge "$total" ] && break
    if [ ${#todo[@]} -gt 0 ]; then
        printf '%s\n' "${todo[@]}" | GPU=$GPU TESTDIR=$TESTDIR NUM_T=$NUM_T xargs -P "$PAR" -I{} bash -c '
            IFS="|" read -r x k ck log <<< "{}"
            g=${x%_e*_s*}
            declare -A EXTRA=( [CAM_tau0.1]="--tau0 0.1" [CAM_tau0.05]="--tau0 0.05"
              [CAMPrior_p0.01]="--method CAMPrior --prior_tau0 0.01" [CAMPrior_p0.1]="--method CAMPrior --prior_tau0 0.1"
              [CAMPrior_p1]="--method CAMPrior --prior_tau0 1"
              [CAMPrior_p1_warmstart]="--method CAMPrior --prior_tau0 1 --prior_kl_ref uniform"
              [CAMPrior_p1_refonly]="--method CAMPrior --prior_tau0 1 --prior_policy off"
              [CAMPrior_p1_fixed]="--method CAMPrior --prior_tau0 1 --prior_fixed"
              [CAMPrior_p10]="--method CAMPrior --prior_tau0 10" )
            [ -n "${EXTRA[$g]+x}" ] || { echo "FAILED $x ep$k: unknown group $g"; exit 0; }
            mkdir -p "$(dirname "$log")"
            CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MODE=test \
              PROJECT=mis_er700_eval EXP=$x bash cam/scripts/mis/mis_er700_cam.sh ${EXTRA[$g]} \
              --checkpoint "$PWD/$ck" --test_path "$CAM_ER700_DIR/$TESTDIR" \
              --num_t $NUM_T --num_k 20 --log_file "$PWD/$log" > "$log.out" 2>&1 \
              && echo "done $x ep$k $(head -1 "$log")" || echo "FAILED $x ep$k"'
    else
        sleep 600
    fi
done
echo "all done"
