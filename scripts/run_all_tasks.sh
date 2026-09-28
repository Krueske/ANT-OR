#!/bin/bash
# Orchestrator: run one method across all 7 tasks (text + multimodal mixed).

set -e
. "$(dirname "$0")/_common.sh"

METHOD="${1:?Usage: $0 <method> [output_file_prefix] [model]
Valid methods:
  eval_single_direct  - single-agent direct evaluation
  eval_single_cot     - single-agent CoT evaluation
  eval_single_icl     - single-agent ICL evaluation
  eval_multi_cot      - multi-agent CoT evaluation
  eval_multi_icl      - multi-agent CoT+ICL evaluation
  spo_single          - single-agent SPO pipeline
  spo_multi           - multi-agent SPO pipeline
  antor               - ANTOR pipeline

When [model] is omitted, defaults to DeepSeek-V3.1-Terminus for text-only
tasks and Qwen3-VL-235B-A22B-Instruct for multimodal tasks.
}"
OUTPUT_FILE_PREFIX="${2:-${METHOD}_$(date +%Y%m%d)}"
MODEL_OVERRIDE="${3:-}"

ALL_TASKS=(
    "disease_privacy_assessment"
    "billing_scenario_classification"
    "social_text_classification"
    "entity_extraction"
    "image_classification"
    "image_text_relevance"
    "image_attribution_recognition"
)

echo "=========================================="
echo "ANT-OR Batch Orchestrator"
echo "  method:  ${METHOD}"
echo "  prefix:  ${OUTPUT_FILE_PREFIX}"
echo "  model:   ${MODEL_OVERRIDE:-<task-aware default>}"
echo "  tasks:   ${#ALL_TASKS[@]}"
echo "=========================================="
for task in "${ALL_TASKS[@]}"; do echo "    - ${task}"; done
echo ""

# Dispatch one task to the appropriate sub-script.
dispatch() {
    local task="$1"
    local prefix="${OUTPUT_FILE_PREFIX}_${task}"
    local model="${MODEL_OVERRIDE:-$(default_model_for "${task}")}"
    case "${METHOD}" in
        eval_single_direct)
            bash scripts/run_eval_single.sh "${task}" direct "${prefix}" "${model}"
            ;;
        eval_single_cot)
            bash scripts/run_eval_single.sh "${task}" cot "${prefix}" "${model}"
            ;;
        eval_single_icl)
            bash scripts/run_eval_single.sh "${task}" icl "${prefix}" "${model}"
            ;;
        eval_multi_cot)
            bash scripts/run_eval_multi.sh "${task}" 0 "${prefix}" "${model}"
            ;;
        eval_multi_icl)
            bash scripts/run_eval_multi.sh "${task}" 1 "${prefix}" "${model}"
            ;;
        spo_single)
            bash scripts/run_spo_single.sh "${task}" "${prefix}" "${model}"
            ;;
        spo_multi)
            bash scripts/run_spo_multi.sh "${task}" "${prefix}" "${model}"
            ;;
        antor)
            bash scripts/run_antor.sh "${task}" "${prefix}" "${model}"
            ;;
        *)
            echo "Unknown method: ${METHOD}"
            exit 1
            ;;
    esac
}

batch_start=$(date +%s)
for task in "${ALL_TASKS[@]}"; do
    echo ""
    echo "=========================================="
    echo ">>> [${METHOD}] ${task}"
    echo "=========================================="
    dispatch "${task}"
    echo "<<< [${METHOD}] ${task} done"
done

batch_elapsed=$(($(date +%s) - batch_start))
echo ""
echo "=========================================="
echo "All ${#ALL_TASKS[@]} tasks completed in ${batch_elapsed}s ($((batch_elapsed / 60))m $((batch_elapsed % 60))s)"
echo "Output dirs: output/<task>/"
echo "=========================================="
