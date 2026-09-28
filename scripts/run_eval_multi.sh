#!/bin/bash
# Multi-agent baseline evaluation (CoT, optional ICL).
# Runs evaluate_mas.py without prior optimization.

set -e
. "$(dirname "$0")/_common.sh"

TASK_TYPE="${1:?Usage: $0 <task_type> [use_icl] [output_file_prefix] [model]}"
USE_ICL="${2:-0}"
OUTPUT_FILE_PREFIX="${3:-mas_baseline_$(date +%Y%m%d)}"
MODEL="${4:-$(default_model_for "${TASK_TYPE}")}"

if [[ "${USE_ICL}" != "0" && "${USE_ICL}" != "1" ]]; then
    echo "Error: use_icl must be 0 or 1; got '${USE_ICL}'"
    exit 1
fi

icl_flag=""
icl_label="cot"
if [[ "${USE_ICL}" == "1" ]]; then
    icl_flag="--use_icl"
    icl_label="cot_icl"
fi

echo "=========================================="
echo "ANT-OR Multi-Agent Evaluation"
echo "  task:    ${TASK_TYPE}"
echo "  method:  ${icl_label}"
echo "  prefix:  ${OUTPUT_FILE_PREFIX}"
echo "  model:   ${MODEL}"
echo "=========================================="

mkdir -p logs/${TASK_TYPE} output/${TASK_TYPE}

log_file="logs/${TASK_TYPE}/eval_multi_${icl_label}_${OUTPUT_FILE_PREFIX}.log"
start=$(date +%s)

python -u src/evaluate_mas.py \
    --task_type "${TASK_TYPE}" \
    --output_file_prefix "${OUTPUT_FILE_PREFIX}" \
    --model "${MODEL}" \
    ${icl_flag} \
    > "${log_file}" 2>&1

elapsed=$(($(date +%s) - start))
echo "[Done] Evaluation completed in ${elapsed}s ($((elapsed / 60))m $((elapsed % 60))s)"
echo "Log: ${log_file}"
