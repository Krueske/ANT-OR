#!/bin/bash
# Multi-agent SPO pipeline: optimize MAS prompts, then evaluate with best epoch.

set -e
. "$(dirname "$0")/_common.sh"

TASK_TYPE="${1:?Usage: $0 <task_type> [output_file_prefix] [model]}"
OUTPUT_FILE_PREFIX="${2:-spo_multi_$(date +%Y%m%d)}"
MODEL="${3:-$(default_model_for "${TASK_TYPE}")}"

echo "=========================================="
echo "ANT-OR Multi-Agent SPO Pipeline"
echo "  task:    ${TASK_TYPE}"
echo "  prefix:  ${OUTPUT_FILE_PREFIX}"
echo "  model:   ${MODEL}"
echo "=========================================="

mkdir -p logs/${TASK_TYPE} output/${TASK_TYPE} cases/${TASK_TYPE}

# Step 1: Multi-agent SPO optimization
echo ""
echo "[Step 1/2] Multi-agent SPO optimization..."
opt_log="logs/${TASK_TYPE}/spo_multi_optimize_${OUTPUT_FILE_PREFIX}.log"
opt_start=$(date +%s)

python -u src/optimize_mas_spo.py \
    --task_type "${TASK_TYPE}" \
    --output_file_prefix "${OUTPUT_FILE_PREFIX}" \
    --model "${MODEL}" \
    > "${opt_log}" 2>&1

opt_elapsed=$(($(date +%s) - opt_start))
echo "[Done] Optimization completed in ${opt_elapsed}s ($((opt_elapsed / 60))m $((opt_elapsed % 60))s)"

# Step 2: Evaluate with best epoch prompts
echo ""
echo "[Step 2/2] Multi-agent evaluation with best-epoch prompts..."
eval_log="logs/${TASK_TYPE}/spo_multi_evaluate_${OUTPUT_FILE_PREFIX}.log"
eval_start=$(date +%s)

python -u src/evaluate_mas.py \
    --task_type "${TASK_TYPE}" \
    --output_file_prefix "${OUTPUT_FILE_PREFIX}" \
    --model "${MODEL}" \
    --use_best_epoch_prompt \
    > "${eval_log}" 2>&1

eval_elapsed=$(($(date +%s) - eval_start))
echo "[Done] Evaluation completed in ${eval_elapsed}s ($((eval_elapsed / 60))m $((eval_elapsed % 60))s)"

total=$((opt_elapsed + eval_elapsed))
echo ""
echo "=========================================="
echo "Pipeline finished in ${total}s ($((total / 60))m $((total % 60))s)"
echo "Logs: ${opt_log}  ${eval_log}"
echo "Output dir: output/${TASK_TYPE}/"
echo "=========================================="
