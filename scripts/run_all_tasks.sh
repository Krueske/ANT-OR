#!/bin/bash
# Batch Execution Script for Multiple Tasks

set -e

OUTPUT_FILE_PREFIX="${1:-ant_or_batch_$(date +%Y%m%d)}"

echo "=========================================="
echo "ANT-OR Batch Task Execution Script"
echo "=========================================="
echo "Output Prefix: ${OUTPUT_FILE_PREFIX}"
echo "=========================================="

# Define text tasks
TEXT_TASKS=(
    "disease_privacy_assessment"
    "billing_scenario_classification"
    "social_text_classification"
    "entity_extraction"
)

# Define multimodal tasks
MULTIMODAL_TASKS=(
    "image_classification"
    "image_text_relevance"
    "image_attribution_recognition"
)

echo ""
echo "[Text Tasks]"
for task in "${TEXT_TASKS[@]}"; do
    echo "  - ${task}"
done

echo ""
echo "[Multimodal Tasks]"
for task in "${MULTIMODAL_TASKS[@]}"; do
    echo "  - ${task}"
done

# Run text tasks
echo ""
echo "=========================================="
echo "Starting Text Tasks"
echo "=========================================="
for TASK_TYPE in "${TEXT_TASKS[@]}"; do
    echo ""
    echo ">>> Processing: ${TASK_TYPE}"
    bash scripts/run_text_task.sh ${OUTPUT_FILE_PREFIX}_${TASK_TYPE} ${TASK_TYPE}
    echo "<<< Completed: ${TASK_TYPE}"
done

# Run multimodal tasks
echo ""
echo "=========================================="
echo "Starting Multimodal Tasks"
echo "=========================================="
for TASK_TYPE in "${MULTIMODAL_TASKS[@]}"; do
    echo ""
    echo ">>> Processing: ${TASK_TYPE}"
    bash scripts/run_multimodal_task.sh ${OUTPUT_FILE_PREFIX}_${TASK_TYPE} ${TASK_TYPE}
    echo "<<< Completed: ${TASK_TYPE}"
done

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "Output Prefix: ${OUTPUT_FILE_PREFIX}"
echo "=========================================="
