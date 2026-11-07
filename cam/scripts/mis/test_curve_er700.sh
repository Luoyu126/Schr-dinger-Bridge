#!/bin/bash
# Test-set curve for the ER-[700-800] CAM-arm runs (local addition, see cam/UPSTREAM.md):
# evaluates saved epoch checkpoints on the 128 ER-[700-800] test graphs on CPU (the GPUs are
# busy training), CAM's protocol: 50 steps x 20 samples, best of 20, CAM's greedy decoder.
#   bash cam/scripts/mis/test_curve_er700.sh "<exp names>" "<epochs>" [parallel=4] [threads=8]
# e.g. bash cam/scripts/mis/test_curve_er700.sh "CAM_tau0.1_e100_s0 CAMPrior_p1_e100_s0" "0 5 10"
# Writes runs/cam/mis_er700_test/<exp>/epoch_XXXX/test_log.txt(.per_graph.csv); finished
# checkpoints are skipped.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EXPS=($1); EPOCHS=($2); PAR="${3:-4}"; THR="${4:-8}"
DATA="${CAM_ER700_DIR:-/data/shengyuf/MIS/data_er}"
OUT="$CAM_RUNDIR/mis_er700_test"
jobs=()
for e in "${EPOCHS[@]}"; do for x in "${EXPS[@]}"; do
    ck="$CAM_RUNDIR/mis_er700/$x/$(printf 'epoch_%04d.pt' "$e")"
    log="$OUT/$x/$(printf 'epoch_%04d' "$e")/test_log.txt"
    [ -f "$ck" ] || continue
    [ -f "$log.per_graph.csv" ] && continue
    jobs+=("$x|$ck|$log")
done; done
echo "${#jobs[@]} checkpoints to evaluate"
export DATA THR CAM_PYTHON
printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -I{} bash -c '
    IFS="|" read -r x ck log <<< "{}"
    method=CAM; [[ "$x" == CAMPrior* ]] && method=CAMPrior
    mkdir -p "$(dirname "$log")"
    cd "$CAM_ROOT" && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=$THR MKL_NUM_THREADS=$THR nice -n 15 \
      "$CAM_PYTHON" main.py --train_path "$DATA/er_train" --valid_path "$DATA/er_valid" \
      --test_path "$DATA/er_test" --problem mis --batch_size 1 --mini_batch_size 512 \
      --num_t 50 --num_k 20 --num_tp 50 --num_kp 20 --num_h 64 --num_l 8 --method $method \
      --model EncodeProcessDecode --checkpoint "$ck" --wandb_name test --wandb_run_name test \
      --log_file "$log" > "$log.out" 2>&1 && echo "done $x $(basename $ck)" || echo "FAILED $x $(basename $ck)"'
