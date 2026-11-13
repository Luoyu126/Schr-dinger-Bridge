#!/bin/bash
# Validation sweep of RLSA on MDS BA-[200-300] / BA-[800-1200] (local addition, see cam/UPSTREAM.md), as
# done for MIS on 11-08 and MCl to choose the prior step size d: RLSA (solvers/rlsa.py, = the untrained
# CAMPrior prior) on the first NVAL validation graphs with CAM's sampling budget (50 steps x 20
# chains), over a grid of initial temperature tau0 and step size d. Never looks at the test set.
# The same grid gives the same-budget RLSA baseline (best (tau0, d) on validation).
#
#   GPUS="2 3" PAR=2 SCALES="small large" TAUS="0.1 1 4" DS="2 5 20" \
#     nohup bash cam/scripts/mds/mds_rlsa_sweep.sh > runs/queue_cam_mds/rlsa_sweep.log 2>&1 &
#   ... DRY=1  print the jobs only
# Output: runs/cam/<problem>_rb_<scale>_rlsa_sweep/t<tau0>_d<d>/test_log.txt (first line: Avg. obj,
# = the mean dominating set size, lower is better), then a table sorted by validation score at the end of the log.
# Jobs are dealt round-robin to GPUS, PAR at a time per GPU; finished points are skipped on rerun.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
PROBLEM=mds; SCALES=(${SCALES:-small large}); TAUS=(${TAUS:-0.01 0.1 1 4})
DS=(${DS:-2 5 10 20 50}); GPUS=(${GPUS:?set GPUS, e.g. GPUS=\"2 3 4 5 6 7\"}); PAR="${PAR:-2}"
NVAL="${NVAL:-64}"; NUM_T="${NUM_T:-50}"; NUM_K="${NUM_K:-20}"; DRY="${DRY:-0}"
DATA="${CAM_MDS_DIR:-/home/shengyuf/yunyi/data/CAM_data/MDS}"

jobs=()
for sc in "${SCALES[@]}"; do for t in "${TAUS[@]}"; do for d in "${DS[@]}"; do
    out="runs/cam/mds_ba_${sc}_rlsa_sweep/t${t}_d${d}"
    [ -f "$out/test_log.txt.per_graph.csv" ] || jobs+=("$sc|$t|$d|$out")
done; done; done
echo "$(date '+%F %T') ${#jobs[@]} points to run on GPUs ${GPUS[*]} ($PAR per GPU)"

run_gpu() {   # gpu, then its jobs
    local gpu=$1; shift
    printf '%s\n' "$@" | GPU=$gpu PROBLEM=$PROBLEM NVAL=$NVAL NUM_T=$NUM_T NUM_K=$NUM_K DATA=$DATA \
      xargs -r -P "$PAR" -I{} bash -c '
        IFS="|" read -r sc t d out <<< "{}"
        mkdir -p "$out"
        if CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PROBLEM=$PROBLEM \
             SCALE=$sc MODE=test PROJECT=mds_ba_${sc}_rlsa_sweep EXP=t${t}_d${d} \
             bash cam/scripts/mds/mds_ba_local.sh --method RLSA --tau0 $t --num_d $d \
             --test_path "$DATA/BA_${sc}_valid" --test_samples $NVAL --num_t $NUM_T --num_k $NUM_K \
             --log_file "$PWD/$out/test_log.txt" > "$out/test_log.txt.out" 2>&1 \
             && [ -f "$out/test_log.txt.per_graph.csv" ]; then
            echo "$(date "+%F %T") done BA-$sc tau0=$t d=$d $(head -1 "$out/test_log.txt")"
        else
            echo "$(date "+%F %T") FAILED BA-$sc tau0=$t d=$d, see $out/test_log.txt.out"
        fi'
}

if [ "$DRY" = 1 ]; then printf '  %s\n' "${jobs[@]}"; exit 0; fi
pids=()
for i in "${!GPUS[@]}"; do
    mine=()
    for j in "${!jobs[@]}"; do [ $((j % ${#GPUS[@]})) -eq "$i" ] && mine+=("${jobs[$j]}"); done
    [ ${#mine[@]} -gt 0 ] && { run_gpu "${GPUS[$i]}" "${mine[@]}" & pids+=($!); }
done
for p in "${pids[@]}"; do wait "$p"; done

echo "$(date '+%F %T') all done. Validation ($NVAL graphs, $NUM_T steps x $NUM_K chains), best first:"
for sc in "${SCALES[@]}"; do
    for f in runs/cam/mds_ba_${sc}_rlsa_sweep/t*_d*/test_log.txt; do
        [ -f "$f.per_graph.csv" ] || continue
        v=$(head -1 "$f" | sed 's/.*: *//'); n=$(basename "$(dirname "$f")")
        awk -v sc="$sc" -v n="$n" -v v="$v" 'BEGIN { printf "  BA-%s  %-12s %.3f\n", sc, n, v }'
    done | sort -k3 -n
done
