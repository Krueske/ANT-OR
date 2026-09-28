#!/bin/bash
# Shared helpers for ANT-OR launcher scripts.
# Source via: . scripts/_common.sh

# Pick the default model for a given task. Text-only tasks default to
# DeepSeek-V3.1-Terminus; multimodal tasks default to Qwen3-VL-235B-A22B-Instruct.
default_model_for() {
    local task="$1"
    case "${task}" in
        disease_privacy_assessment | \
        billing_scenario_classification | \
        social_text_classification | \
        entity_extraction)
            echo "DeepSeek-V3.1-Terminus"
            ;;
        image_classification | \
        image_text_relevance | \
        image_attribution_recognition)
            echo "Qwen3-VL-235B-A22B-Instruct"
            ;;
        *)
            # Unknown task — fall back to the VL model (handles both modalities).
            echo "Qwen3-VL-235B-A22B-Instruct"
            ;;
    esac
}
