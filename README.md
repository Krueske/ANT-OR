# ANT-OR

ANT-OR is an automated annotation framework through multi-agent optimization with agent-level contrastive knowledge to improve accuracy and reliability in complex annotation tasks.

## Supported Tasks

### Text Tasks
- `disease_privacy_assessment`: Disease privacy compliance assessment
- `billing_scenario_classification`: Billing scenario classification
- `social_text_classification`: Social media text classification
- `entity_extraction`: medical entity extraction

### Multimodal Tasks
- `image_classification`: Image classification (category, subcategory, guidance analysis)
- `image_text_relevance`: Image-text relevance assessment
- `image_attribution_recognition`: Image attribute recognition

## Installation

```bash
# Install Python dependencies
pip install openai numpy pandas pillow requests
pip install tiktoken
pip install opencv-python imagehash
```

## Usage

### 1. Configure API

Edit `utils/call_llm_api.py` to configure your LLM API:

```python
# Configure in call_llm_general function
API_BASE_URL = "https://your-api-endpoint.com/v1"
API_KEY = "your-api-key"
```

### 2. Data Preparation

> **Note on Language:** The original annotation data is in Chinese, and the experiments in our paper were primarily conducted on the Chinese version. We also provide an English-translated version of the benchmark for broader accessibility.

> **Note on Privacy:** The benchmark data involving personal privacy has been anonymized.

This project includes parsed annotation data for 7 tasks in the `data/` directory:

- `billing_scenario_classification/`: Billing scenario classification data
- `entity_extraction/`: Medical entity extraction data
- `image_attribution_recognition/`: Image attribution recognition data
- `image_classification/`: Image classification data
- `image_text_relevance/`: Image-text relevance data
- `disease_privacy_assessment/`: Disease privacy assessment data
- `social_text_classification/`: Social media text classification data

Each task directory contains:
- `*_parsed.json`: Parsed annotation data
- `*_rules.md`: Annotation rule documents

**Note on Multimodal Data:** The `image_url` fields in multimodal tasks (image_classification, image_text_relevance, image_attribution_recognition) are currently empty due to anonymization. These URLs will be provided after the anonymization period.

### 3. Run Single Task

**Text Task:**
```bash
bash scripts/run_text_task.sh <output_prefix> <task_type>
```

Example:
```bash
bash scripts/run_text_task.sh my_experiment disease_privacy_assessment
```

**Multimodal Task:**
```bash
bash scripts/run_multimodal_task.sh <output_prefix> <task_type>
```

Example:
```bash
bash scripts/run_multimodal_task.sh my_experiment image_classification
```

### 4. Batch Run

```bash
bash scripts/run_all_tasks.sh <output_prefix>
```

Example:
```bash
bash scripts/run_all_tasks.sh batch_20250523
```

### 5. Run Python Scripts Directly

**Optimization Phase:**
```bash
# Text task
python src/optimize_text.py <task_type> <output_prefix>

# Multimodal task
python src/optimize_multimodal.py <task_type> <output_prefix>
```

**Evaluation Phase:**
```bash
# Text task
python src/evaluate_text.py \
    --task_type <task_type> \
    --output_file_prefix <prefix> \
    --use_best_epoch_prompt

# Multimodal task
python src/evaluate_multimodal.py \
    --task_type <task_type> \
    --output_file_prefix <prefix> \
    --use_best_epoch_prompt
```

## Core Parameters

### Optimization Script Parameters
- `task_type`: Task type (e.g., disease_privacy_assessment)
- `output_prefix`: Output file prefix for experiment identification

### Evaluation Script Parameters
- `--task_type`: Task type
- `--output_file_prefix`: Output file prefix
- `--use_best_epoch_prompt`: Use prompts from the best performing epoch
- `--use_optimized_prompt`: Use optimized prompts (the final epoch)
- `--use_contrastive_context`: Enable contrastive example context injection (default: on)

