#!/bin/bash
# Test-set curve of the RB runs (local addition, see cam/UPSTREAM.md): evaluates checkpoints of
# runs/cam/mis_rb_<scale>/<exp>/ on all 1000 RB_<scale>_test graphs with CAM's protocol (50 steps x
# 20 samples, best of 20), on one GPU or on CPU, several at a time. Loops, picking up checkpoints as
# training writes them, until every point is done; "best" (best.pt) is evaluated once training ended.
# Epoch k = after k epochs = epoch_{k-1}.pt; k = 0 is the untrained epoch_init.pt.
# A failed point is retried at most twice (then reported and skipped); the queue never retries
# immediately: it waits WAIT seconds between passes that made no progress.
#
#   SCALE=small DEVICE=gpu GPU=4 PAR=2 EXPS="CAM_tau0.1_e100_s0 CAMPrior_p1_d5_e100_s0" \
#     nohup bash cam/scripts/mis/rb_curve.sh > runs/cam/mis_rb_small_test/queue.log 2>&1 &
#   SCALE=large DEVICE=cpu THR=4 PAR=2 KS="0 1 2 5 10" EXPS=... nohup bash cam/scripts/mis/rb_curve.sh ...
# Env: KS (epochs; default 0 1 2 5 10 20 ... LAST), BEST (1: also best.pt), LAST (default 100 small,
# 150 large), TEST_SAMPLES (default all; >0 only for dry runs), OUT (default runs/cam/mis_rb_<scale>_test).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
SCALE="${SCALE:?SCALE=small|large}"; DEVICE="${DEVICE:-gpu}"; PAR="${PAR:-2}"; THR="${THR:-4}"
case "$SCALE" in small) LAST="${LAST:-100}" ;; large) LAST="${LAST:-150}" ;; *) echo "bad SCALE" >&2; exit 1 ;; esac
case "$DEVICE" in gpu) GPU="${GPU:?set GPU}" ;; cpu) GPU="" ;; *) echo "DEVICE must be gpu or cpu" >&2; exit 1 ;; esac
EXPS=(${EXPS:?set EXPS}); KS=(${KS:-0 1 2 5 $(seq -s ' ' 10 10 "$LAST")}); BEST="${BEST:-1}"
TEST_SAMPLES="${TEST_SAMPLES:--1}"; WAIT="${WAIT:-600}"
RUNS="${RUNS:-runs/cam/mis_rb_$SCALE}"; OUT="${OUT:-runs/cam/mis_rb_${SCALE}_test}"
mkdir -p "$OUT"
declare -A FAILS=()
points=("${KS[@]}"); [ "$BEST" = 1 ] && points+=(best)
total=$(( ${#points[@]} * ${#EXPS[@]} ))
while :; do
    todo=(); done_n=0; gave_up=0
    for k in "${points[@]}"; do for x in "${EXPS[@]}"; do
        if [ "$k" = best ]; then
            log="$OUT/$x/best/test_log.txt"; ck="$RUNS/$x/best.pt"
            [ -f "$RUNS/$x/$(printf 'epoch_%04d.pt' $((LAST - 1)))" ] || ck=""   # training not over yet
        else
            log="$OUT/$x/ep$(printf '%03d' "$k")/test_log.txt"
            if [ "$k" = 0 ]; then ck="$RUNS/$x/epoch_init.pt"; else ck="$RUNS/$x/$(printf 'epoch_%04d.pt' $((k - 1)))"; fi
        fi
        if [ -f "$log.per_graph.csv" ]; then done_n=$((done_n + 1)); continue; fi
        if [ "${FAILS[$x|$k]:-0}" -ge 3 ]; then gave_up=$((gave_up + 1)); continue; fi
        [ -n "$ck" ] && [ -f "$ck" ] && todo+=("$x|$k|$ck|$log")
    done; done
    echo "$(date '+%F %T') done $done_n/$total, gave up $gave_up, ready ${#todo[@]}"
    [ $((done_n + gave_up)) -ge "$total" ] && break
    if [ ${#todo[@]} -eq 0 ]; then sleep "$WAIT"; continue; fi
    results=$(printf '%s\n' "${todo[@]}" | SCALE=$SCALE DEVICE=$DEVICE GPU=$GPU THR=$THR TEST_SAMPLES=$TEST_SAMPLES \
      xargs -P "$PAR" -I{} bash -c '
        IFS="|" read -r x k ck log <<< "{}"
        g=${x%_e*_s*}
        if [ "$g" = CAM_tau0.1 ]; then extra="--tau0 0.1"
        elif [[ "$g" =~ ^CAMPrior_p([0-9.]+)_d([0-9]+)$ ]]; then
            extra="--method CAMPrior --prior_tau0 ${BASH_REMATCH[1]} --prior_d ${BASH_REMATCH[2]}"
        else echo "FAIL|$x|$k|unknown group $g"; exit 0; fi
        mkdir -p "$(dirname "$log")"
        if CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=$THR MKL_NUM_THREADS=$THR \
             PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MODE=test PROJECT=mis_rb_${SCALE}_eval EXP=$x \
             bash cam/scripts/mis/mis_rb_cam.sh $extra --checkpoint "$PWD/$ck" --test_path "${CAM_RB_DIR:-/data/shengyuf/MIS}/RB_${SCALE}_test" \
             --test_samples $TEST_SAMPLES --log_file "$PWD/$log" > "$log.out" 2>&1 && [ -f "$log.per_graph.csv" ]; then
            echo "OK|$x|$k|$(head -1 "$log")"
        else
            echo "FAIL|$x|$k|$(grep -m1 -E "error|Error" "$log.out" | cut -c1-200)"
        fi')
    progress=0
    while IFS="|" read -r st x k msg; do
        [ -z "$st" ] && continue
        if [ "$st" = OK ]; then progress=1; echo "done $x ep$k $msg"
        else FAILS[$x|$k]=$(( ${FAILS[$x|$k]:-0} + 1 )); echo "FAILED (${FAILS[$x|$k]}/3) $x ep$k: $msg"; fi
    done <<< "$results"
    [ "$progress" = 1 ] || sleep "$WAIT"
done
echo "all done ($gave_up points given up)"
