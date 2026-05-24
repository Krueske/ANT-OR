#!/bin/bash
# Text Task Execution Script
# Runs optimization and evaluation for text tasks

set -e

# Configuration
OUTPUT_FILE_PREFIX="${1:-ant_or_text_optimization}"
TASK_TYPE="${2:-disease_privacy_assessment}"

echo "=========================================="
echo "ANT-OR Text Task Execution Script"
echo "=========================================="
echo "Task Type: ${TASK_TYPE}"
echo "Output Prefix: ${OUTPUT_FILE_PREFIX}"
echo "=========================================="

# Create log directories
mkdir -p logs/${TASK_TYPE}
mkdir -p output/${TASK_TYPE}
mkdir -p cases/${TASK_TYPE}

# Step 1: Run optimization
echo ""
echo "[Step 1/2] Starting prompt optimization..."
cmd_start_time=$(date +%s)

python -u src/optimize_text.py ${TASK_TYPE} ${OUTPUT_FILE_PREFIX} > logs/${TASK_TYPE}/optimize_${OUTPUT_FILE_PREFIX}.log 2>&1

cmd_end_time=$(date +%s)
optimize_cost=$((cmd_end_time - cmd_start_time))
echo "[Done] Optimization completed in ${optimize_cost} seconds ($((optimize_cost/60))m $((optimize_cost%60))s)"

# Step 2: Run evaluation
echo ""
echo "[Step 2/2] Starting evaluation..."
cmd_start_time=$(date +%s)

python -u src/evaluate_text.py \
    --task_type ${TASK_TYPE} \
    --output_file_prefix ${OUTPUT_FILE_PREFIX} \
    --use_best_epoch_prompt > logs/${TASK_TYPE}/evaluate_${OUTPUT_FILE_PREFIX}.log 2>&1

cmd_end_time=$(date +%s)
evaluate_cost=$((cmd_end_time - cmd_start_time))
echo "[Done] Evaluation completed in ${evaluate_cost} seconds ($((evaluate_cost/60))m $((evaluate_cost%60))s)"

# Total time
total_cost=$((optimize_cost + evaluate_cost))
echo ""
echo "=========================================="
echo "All tasks completed!"
echo "Total time: ${total_cost} seconds ($((total_cost/60))m $((total_cost%60))s)"
echo "Log directory: logs/${TASK_TYPE}/"
echo "Output directory: output/${TASK_TYPE}/"
echo "=========================================="
