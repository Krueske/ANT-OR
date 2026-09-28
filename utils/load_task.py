import json
import os
from collections import Counter
from typing import List, Dict, Any, Optional

# ============================================================================
# ============================================================================
BASEPATH = "benchmark"
NEW_DATASET_BASEPATH = "benchmark"


# ============================================================================
# ============================================================================

def _normalize_annotation(annotation: Any) -> str:
    """Normalize annotation result to string for voting.
    For list/dict annotations (e.g. NER), serialize to sorted JSON string for consistent comparison.
    """
    if isinstance(annotation, str):
        return annotation.strip()
    elif isinstance(annotation, list):
        try:
            sorted_list = sorted(annotation, key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))
            return json.dumps(sorted_list, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return json.dumps(annotation, ensure_ascii=False)
    elif isinstance(annotation, dict):
        return json.dumps(annotation, ensure_ascii=False, sort_keys=True)
    else:
        return str(annotation).strip()


def _compute_vote_info(annotations: List[Any], min_votes: int = 2) -> Dict[str, Any]:
    """Compute vote info from multiple annotator results."""
    n = len(annotations)
    normalized = [_normalize_annotation(a) for a in annotations]
    vote_counter = Counter(normalized)

    norm_to_original = {}
    for ann, norm in zip(annotations, normalized):
        if norm not in norm_to_original:
            norm_to_original[norm] = ann

    sorted_votes = vote_counter.most_common()
    max_votes = sorted_votes[0][1] if sorted_votes else 0

    if max_votes >= 3:
        if len(sorted_votes) <= 2 and max_votes >= 3:
            if (len(sorted_votes) == 2 and sorted_votes[1][1] >= 2 and
                    sorted_votes[0][1] + sorted_votes[1][1] == n):
                agreement_level = "split"  # 3:2
            else:
                agreement_level = "consensus"  # 5:0, 4:1, 3:1:1
        else:
            agreement_level = "consensus"
    else:
        agreement_level = "ambiguous"

    acceptable_answers = []
    for norm_ans, count in sorted_votes:
        if count >= min_votes:
            acceptable_answers.append(norm_to_original[norm_ans])

    majority_answer = norm_to_original[sorted_votes[0][0]] if sorted_votes else None
    vote_distribution = dict(vote_counter)

    vote_distribution_detail = []
    for norm_ans, count in sorted_votes:
        vote_distribution_detail.append({
            "answer": norm_to_original[norm_ans],
            "normalized": norm_ans,
            "count": count
        })

    return {
        "all_annotations": annotations,
        "acceptable_answers": acceptable_answers,
        "majority_answer": majority_answer,
        "agreement_level": agreement_level,
        "vote_distribution": vote_distribution,
        "vote_distribution_detail": vote_distribution_detail,
    }


def _find_requirement_doc(dataset_dir: str) -> str:
    """Auto-find rule document (.md or .txt) under dataset directory."""
    requirement_doc_path = None
    if os.path.exists(dataset_dir):
        for fname in os.listdir(dataset_dir):
            if fname.endswith("zh.md") or fname.endswith(".txt"):
                if "rules" in fname.lower():
                    requirement_doc_path = os.path.join(dataset_dir, fname)
                    break
        if requirement_doc_path is None:
            for fname in os.listdir(dataset_dir):
                if fname.endswith(".md") or fname.endswith(".txt"):
                    requirement_doc_path = os.path.join(dataset_dir, fname)
                    break
    if requirement_doc_path is None:
        raise FileNotFoundError(f"未找到规则文档: {dataset_dir}")
    return requirement_doc_path


def _extract_annotations(item: Dict[str, Any], num_annotators: int = 5):
    """Extract annotations from data entry and count empty annotations."""
    annotations = []
    empty_count = 0
    for i in range(1, num_annotators + 1):
        ann_key = f"annotation_{i}"
        ann_value = item.get(ann_key)
        annotations.append(ann_value)
        if ann_value is None or ann_value == "" or ann_value == [] or ann_value == {}:
            empty_count += 1
    return annotations, empty_count


# ============================================================================
# ============================================================================

def load_billing_scenario_classification_data(test_num: Optional[int] = None, model_name: str = "qwen3-235b-thinking"):
    """Load billing scenario classification multi-annotation data."""
    dataset_name = "billing_scenario_classification"
    dataset_dir = os.path.join(NEW_DATASET_BASEPATH, dataset_name)
    data_file = os.path.join(dataset_dir, f"{dataset_name}_parsed_zh.json")
    requirement_doc_path = _find_requirement_doc(dataset_dir)
    output_path = f"output/{model_name}_{dataset_name}_multi_annotation_results.json"

    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    queries = []
    answers = []
    filtered_count = 0

    for item in raw_data:
        query = item.get("query", "")
        annotations, empty_count = _extract_annotations(item)
        if empty_count >= 3:
            filtered_count += 1
            continue

        queries.append(query)
        answers.append(_compute_vote_info(annotations))

        if test_num is not None and len(queries) >= test_num:
            break

    return {
        "requirement_doc_path": requirement_doc_path,
        "output_path": output_path,
        "task_type": dataset_name,
        "queries": queries,
        "answers": answers,
        "is_multi_annotation": True,
    }


def load_entity_extraction_data(test_num: Optional[int] = None, model_name: str = "qwen3-235b-thinking"):
    """Load entity extraction multi-annotation data."""
    dataset_name = "entity_extraction"
    dataset_dir = os.path.join(NEW_DATASET_BASEPATH, dataset_name)
    data_file = os.path.join(dataset_dir, f"{dataset_name}_parsed_zh.json")
    requirement_doc_path = _find_requirement_doc(dataset_dir)
    output_path = f"output/{model_name}_{dataset_name}_multi_annotation_results.json"

    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    queries = []
    answers = []
    filtered_count = 0

    for item in raw_data:
        query = item.get("query", "")
        annotations, empty_count = _extract_annotations(item)
        if empty_count >= 3:
            filtered_count += 1
            continue

        queries.append(query)
        answers.append(_compute_vote_info(annotations))

        if test_num is not None and len(queries) >= test_num:
            break

    return {
        "requirement_doc_path": requirement_doc_path,
        "output_path": output_path,
        "task_type": dataset_name,
        "queries": queries,
        "answers": answers,
        "is_multi_annotation": True,
    }


def load_disease_privacy_assessment_data(test_num: Optional[int] = None, model_name: str = "qwen3-235b-thinking"):
    """Load disease privacy assessment multi-annotation data."""
    dataset_name = "disease_privacy_assessment"
    dataset_dir = os.path.join(NEW_DATASET_BASEPATH, dataset_name)
    data_file = os.path.join(dataset_dir, f"{dataset_name}_parsed_zh.json")
    requirement_doc_path = _find_requirement_doc(dataset_dir)
    output_path = f"output/{model_name}_{dataset_name}_multi_annotation_results.json"

    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    queries = []
    answers = []
    filtered_count = 0

    for item in raw_data:
        query = item.get("query", "")
        annotations, empty_count = _extract_annotations(item)
        if empty_count >= 3:
            filtered_count += 1
            continue

        queries.append(query)
        answers.append(_compute_vote_info(annotations))

        if test_num is not None and len(queries) >= test_num:
            break

    return {
        "requirement_doc_path": requirement_doc_path,
        "output_path": output_path,
        "task_type": dataset_name,
        "queries": queries,
        "answers": answers,
        "is_multi_annotation": True,
    }


def load_social_text_classification_data(test_num: Optional[int] = None, model_name: str = "qwen3-235b-thinking"):
    """Load social text classification multi-annotation data."""
    dataset_name = "social_text_classification"
    dataset_dir = os.path.join(NEW_DATASET_BASEPATH, dataset_name)
    data_file = os.path.join(dataset_dir, f"{dataset_name}_parsed_zh.json")
    requirement_doc_path = _find_requirement_doc(dataset_dir)
    output_path = f"output/{model_name}_{dataset_name}_multi_annotation_results.json"

    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    queries = []
    answers = []
    filtered_count = 0

    for item in raw_data:
        query = item.get("query", "")
        annotations, empty_count = _extract_annotations(item)
        if empty_count >= 3:
            filtered_count += 1
            continue

        queries.append(query)
        answers.append(_compute_vote_info(annotations))

        if test_num is not None and len(queries) >= test_num:
            break

    return {
        "requirement_doc_path": requirement_doc_path,
        "output_path": output_path,
        "task_type": dataset_name,
        "queries": queries,
        "answers": answers,
        "is_multi_annotation": True,
    }


def load_image_text_relevance_data(test_num: Optional[int] = None, model_name: str = "qwen3-235b-thinking"):
    """Load image-text relevance multi-annotation data."""
    dataset_name = "image_text_relevance"
    dataset_dir = os.path.join(NEW_DATASET_BASEPATH, dataset_name)
    data_file = os.path.join(dataset_dir, f"{dataset_name}_parsed_zh.json")
    requirement_doc_path = _find_requirement_doc(dataset_dir)
    output_path = f"output/{model_name}_{dataset_name}_multi_annotation_results.json"

    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    queries = []
    answers = []
    filtered_count = 0

    for item in raw_data:
        search_term = item.get("query", "")
        image_url = item.get("image_url", "")
        query = {
            "text": search_term,
            "image_urls": [image_url] if image_url else []
        }
        annotations, empty_count = _extract_annotations(item)
        if empty_count >= 3:
            filtered_count += 1
            continue

        queries.append(query)
        answers.append(_compute_vote_info(annotations))

        if test_num is not None and len(queries) >= test_num:
            break

    return {
        "requirement_doc_path": requirement_doc_path,
        "output_path": output_path,
        "task_type": dataset_name,
        "queries": queries,
        "answers": answers,
        "is_multi_annotation": True,
    }


def load_image_classification_data(test_num: Optional[int] = None, model_name: str = "qwen3-235b-thinking"):
    """Load image classification (product attribute classification) multi-annotation data."""
    dataset_name = "image_classification"
    dataset_dir = os.path.join(NEW_DATASET_BASEPATH, dataset_name)
    data_file = os.path.join(dataset_dir, f"{dataset_name}_parsed_zh.json")
    requirement_doc_path = _find_requirement_doc(dataset_dir)
    output_path = f"output/{model_name}_{dataset_name}_multi_annotation_results.json"

    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    queries = []
    answers = []
    filtered_count = 0

    for item in raw_data:
        image_url = item.get("query", "")
        query = {
            "text": "",
            "image_urls": [image_url] if image_url else []
        }
        annotations, empty_count = _extract_annotations(item)
        if empty_count >= 3:
            filtered_count += 1
            continue

        queries.append(query)
        answers.append(_compute_vote_info(annotations))

        if test_num is not None and len(queries) >= test_num:
            break

    return {
        "requirement_doc_path": requirement_doc_path,
        "output_path": output_path,
        "task_type": dataset_name,
        "queries": queries,
        "answers": answers,
        "is_multi_annotation": True,
    }


def load_image_attribution_recognition_data(test_num: Optional[int] = None, model_name: str = "qwen3-235b-thinking"):
    """Load image attribution recognition multi-annotation data."""
    dataset_name = "image_attribution_recognition"
    dataset_dir = os.path.join(NEW_DATASET_BASEPATH, dataset_name)
    data_file = os.path.join(dataset_dir, f"{dataset_name}_parsed_zh.json")
    requirement_doc_path = _find_requirement_doc(dataset_dir)
    output_path = f"output/{model_name}_{dataset_name}_multi_annotation_results.json"

    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    queries = []
    answers = []
    filtered_count = 0

    for item in raw_data:
        image_url = item.get("query", "")
        query = {
            "text": "",
            "image_urls": [image_url] if image_url else []
        }
        annotations, empty_count = _extract_annotations(item)
        if empty_count >= 3:
            filtered_count += 1
            continue

        queries.append(query)
        answers.append(_compute_vote_info(annotations))

        if test_num is not None and len(queries) >= test_num:
            break

    return {
        "requirement_doc_path": requirement_doc_path,
        "output_path": output_path,
        "task_type": dataset_name,
        "queries": queries,
        "answers": answers,
        "is_multi_annotation": True,
    }

# ============================================================================
# ============================================================================

def get_task_config(task_type, test_num=None, model_name="qwen3-235b-thinking"):
    """Get configuration and data by task type.

    Supports two formats:
    1. Task types with multi_ prefix (e.g. "multi_billing_scenario_classification")
    2. Task types without multi_ prefix (e.g. "billing_scenario_classification")

    For tasks already migrated to NewDataset, uniformly load multi-annotation data;
    for unmigrated legacy tasks, keep original single-annotation loading as fallback.
    """
    if task_type.startswith("multi_"):
        task_type = task_type[len("multi_"):]

    if task_type == "billing_scenario_classification":
        return load_billing_scenario_classification_data(test_num, model_name)
    elif task_type == "entity_extraction":
        return load_entity_extraction_data(test_num, model_name)
    elif task_type == "disease_privacy_assessment":
        return load_disease_privacy_assessment_data(test_num, model_name)
    elif task_type == "social_text_classification":
        return load_social_text_classification_data(test_num, model_name)
    elif task_type == "image_text_relevance":
        return load_image_text_relevance_data(test_num, model_name)
    elif task_type == "image_classification":
        return load_image_classification_data(test_num, model_name)
    elif task_type == "image_attribution_recognition":
        return load_image_attribution_recognition_data(test_num, model_name)
    else:
        raise ValueError(f"不支持的任务类型: {task_type}")
