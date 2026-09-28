#!/bin/bash
# Single-agent baseline evaluation: --mode {direct, cot, icl}
# Runs evaluate_single_agent.py with the chosen prompting strategy.

set -e
. "$(dirname "$0")/_common.sh"

TASK_TYPE="${1:?Usage: $0 <task_type> [mode] [output_file_prefix] [model]}"
MODE="${2:-cot}"
OUTPUT_FILE_PREFIX="${3:-${MODE}_baseline_$(date +%Y%m%d)}"
MODEL="${4:-$(default_model_for "${TASK_TYPE}")}"

if [[ "${MODE}" != "direct" && "${MODE}" != "cot" && "${MODE}" != "icl" ]]; then
    echo "Error: mode must be one of {direct, cot, icl}; got '${MODE}'"
    exit 1
fi

echo "=========================================="
echo "ANT-OR Single-Agent Evaluation"
echo "  task:    ${TASK_TYPE}"
echo "  mode:    ${MODE}"
echo "  prefix:  ${OUTPUT_FILE_PREFIX}"
echo "  model:   ${MODEL}"
echo "=========================================="

mkdir -p logs/${TASK_TYPE} output/${TASK_TYPE}

log_file="logs/${TASK_TYPE}/eval_single_${MODE}_${OUTPUT_FILE_PREFIX}.log"
start=$(date +%s)

python -u src/evaluate_single_agent.py \
    --task_type "${TASK_TYPE}" \
    --mode "${MODE}" \
    --output_file_prefix "${OUTPUT_FILE_PREFIX}" \
    --model "${MODEL}" \
    > "${log_file}" 2>&1

elapsed=$(($(date +%s) - start))
echo "[Done] Evaluation completed in ${elapsed}s ($((elapsed / 60))m $((elapsed % 60))s)"
echo "Log: ${log_file}"
