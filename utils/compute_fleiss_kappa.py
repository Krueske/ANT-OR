import json
import re
import argparse
import os
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

TASK_JSON_KEYS = {
    "medical_query_evaluation": "隐私合规评估",
    "billing_scenario_classification": "账单类目",
    "classification": "label",
    "social_text_classification": "label",
    "image_classification": None, 
    "image_text_relevance": "label",
    "image_attribution_recognition": "label",
}

TASK_LABEL_MAPS = {
    "medical_query_evaluation": {"无问题": "无", "有问题": "有"},
    "billing_scenario_classification": None,
    "classification": None,
    "social_text_classification": None,
    "image_classification": None,
    "image_text_relevance": None,
    "image_attribution_recognition": None,
}

TASK_NAMES = {
    "medical_query_evaluation": "Classification Task (Medical Query Evaluation - Privacy Compliance)",
    "billing_scenario_classification": "Classification Task (Billing Scenario Classification)",
    "classification": "Classification Task (Generic Classification)",
    "social_text_classification": "Classification Task (Corpus Classification)",
    "entity_extraction": "Entity Extraction Task",
    "image_classification": "Image Classification Task",
    "image_text_relevance": "Image-Text Relevance Task",
    "image_attribution_recognition": "Image Attribution Recognition Task",
}

ENTITY_EXTRACTION_KEY = "entities"

def compute_gwet_ac1_pairwise(rater1: List, rater2: List) -> Dict[str, Any]:
    assert len(rater1) == len(rater2), f"different length: {len(rater1)} vs {len(rater2)}"
    n = len(rater1)
    if n == 0:
        return {"ac1": float("nan"), "kappa": float("nan"),
                "pa": float("nan"), "pe_ac1": float("nan"), "pe_kappa": float("nan"),
                "n": 0, "categories": [], "agreements": 0}

    all_categories = sorted(set(rater1) | set(rater2))

    agreements = sum(1 for a, b in zip(rater1, rater2) if a == b)
    pa = agreements / n

    rater1_counts = Counter(rater1)
    rater2_counts = Counter(rater2)

    # Gwet's AC1 chance agreement
    pe_ac1 = 0.0
    for cat in all_categories:
        p1_k = rater1_counts.get(cat, 0) / n
        p2_k = rater2_counts.get(cat, 0) / n
        q_k = (p1_k + p2_k) / 2.0
        pe_ac1 += q_k * (1 - q_k)

    if pe_ac1 >= 1.0:
        ac1 = 1.0 if pa >= 1.0 else 0.0
    else:
        ac1 = (pa - pe_ac1) / (1 - pe_ac1)

    # Cohen's Kappa chance agreement
    # pe_kappa = Σ p1ₖ * p2ₖ
    pe_kappa = 0.0
    for cat in all_categories:
        p1_k = rater1_counts.get(cat, 0) / n
        p2_k = rater2_counts.get(cat, 0) / n
        pe_kappa += p1_k * p2_k

    if pe_kappa >= 1.0:
        kappa = 1.0 if pa >= 1.0 else 0.0
    else:
        kappa = (pa - pe_kappa) / (1 - pe_kappa)

    return {
        "ac1": ac1, "kappa": kappa,
        "pa": pa, "pe_ac1": pe_ac1, "pe_kappa": pe_kappa,
        "n": n, "categories": all_categories, "agreements": agreements,
    }

def compute_binary_ac1_kappa(rater1_binary: List[int], rater2_binary: List[int]) -> Dict[str, Any]:
    """Binary AC1 + Kappa (for entity-level matching: 1=detected, 0=missed)."""
    return compute_gwet_ac1_pairwise(rater1_binary, rater2_binary)

def compute_fleiss_kappa(ratings_matrix: List[List]) -> Dict[str, Any]:
    """
    Compute Fleiss' Kappa (multi-rater agreement coefficient).

    Args:
      ratings_matrix: N×n matrix, ratings_matrix[i][j] = label assigned by rater j to item i.
                      None indicates missing annotation.

    Formula:
      For each item i and category j, n_ij = number of raters who assigned category j
      P_i = (1 / (n_i * (n_i - 1))) * (Σ_j n_ij² - n_i)   where n_i = valid raters for item i
      P̄  = (1 / N) * Σ_i P_i
      p_j = (1 / Σ n_i) * Σ_i n_ij
      P̄_e = Σ_j p_j²
      κ   = (P̄ - P̄_e) / (1 - P̄_e)
    """
    if not ratings_matrix:
        return {"fleiss_kappa": float("nan"), "P_bar": float("nan"),
                "P_e": float("nan"), "n_items": 0, "n_raters": 0, "n_categories": 0}

    all_categories = set()
    for row in ratings_matrix:
        for label in row:
            if label is not None:
                all_categories.add(label)
    all_categories = sorted(all_categories)
    cat_index = {c: idx for idx, c in enumerate(all_categories)}
    n_cats = len(all_categories)

    if n_cats == 0:
        return {"fleiss_kappa": float("nan"), "P_bar": float("nan"),
                "P_e": float("nan"), "n_items": 0, "n_raters": 0, "n_categories": 0}

    N = len(ratings_matrix)
    n_raters = len(ratings_matrix[0]) if N > 0 else 0

    P_i_list = []
    total_assignments = 0
    cat_totals = [0] * n_cats

    valid_items = 0
    for i in range(N):
        row = ratings_matrix[i]
        n_i = sum(1 for label in row if label is not None)
        if n_i < 2:
            continue
        valid_items += 1
        total_assignments += n_i

        counts = [0] * n_cats
        for label in row:
            if label is not None:
                counts[cat_index[label]] += 1

        for j in range(n_cats):
            cat_totals[j] += counts[j]

        sum_sq = sum(c * c for c in counts)
        P_i = (sum_sq - n_i) / (n_i * (n_i - 1))
        P_i_list.append(P_i)

    if valid_items == 0 or total_assignments == 0:
        return {"fleiss_kappa": float("nan"), "P_bar": float("nan"),
                "P_e": float("nan"), "n_items": N, "n_raters": n_raters,
                "n_categories": n_cats, "valid_items": 0}

    P_bar = sum(P_i_list) / valid_items

    P_e = sum((ct / total_assignments) ** 2 for ct in cat_totals)

    if P_e >= 1.0:
        fleiss_kappa = 1.0 if P_bar >= 1.0 else 0.0
    else:
        fleiss_kappa = (P_bar - P_e) / (1 - P_e)

    return {
        "fleiss_kappa": fleiss_kappa,
        "P_bar": P_bar,
        "P_e": P_e,
        "n_items": N,
        "n_raters": n_raters,
        "n_categories": n_cats,
        "valid_items": valid_items,
    }

def compute_entity_fleiss_kappa(
    raters_entities_per_sample: List[List[List[Dict]]],
    n_items: int,
) -> Dict[str, Any]:
    """
    Standard Fleiss' Kappa (global aggregation, for entity extraction tasks).

    Flattens candidate entities from all samples into a binary decision matrix
    (one row per entity, each rater votes 0/1), then computes P_bar, P_e, and kappa
    globally using the standard Fleiss formula.

    Candidate set = union of entities from all raters.
    """
    n_raters = len(raters_entities_per_sample)
    if n_raters < 2:
        return {"fleiss_kappa": float("nan"), "P_bar": float("nan"),
                "P_e": float("nan"), "n_items": n_items, "n_raters": n_raters,
                "n_categories": 2, "valid_items": 0, "total_entities": 0}

    global_sum_sq_minus_n = 0
    global_total_1 = 0
    global_total_entities = 0
    valid_items = 0

    for idx in range(n_items):
        keys_set = set()
        for r in range(n_raters):
            for e in raters_entities_per_sample[r][idx]:
                keys_set.add(entity_to_key(e))
        if not keys_set:
            continue

        valid_items += 1
        rater_sets = [set(entity_to_key(e) for e in raters_entities_per_sample[r][idx])
                      for r in range(n_raters)]

        for ek in keys_set:
            n_1 = sum(1 for r in range(n_raters) if ek in rater_sets[r])
            n_0 = n_raters - n_1
            global_sum_sq_minus_n += (n_1 * n_1 + n_0 * n_0 - n_raters)
            global_total_1 += n_1
            global_total_entities += 1

    if global_total_entities == 0:
        return {"fleiss_kappa": float("nan"), "P_bar": float("nan"),
                "P_e": float("nan"), "n_items": n_items, "n_raters": n_raters,
                "n_categories": 2, "valid_items": 0, "total_entities": 0}

    N = global_total_entities
    P_bar = global_sum_sq_minus_n / (N * n_raters * (n_raters - 1))

    p1 = global_total_1 / (N * n_raters)
    p0 = 1.0 - p1
    P_e = p0 ** 2 + p1 ** 2

    if P_e >= 1.0:
        fleiss_kappa = 1.0 if P_bar >= 1.0 else 0.0
    else:
        fleiss_kappa = (P_bar - P_e) / (1 - P_e)

    return {
        "fleiss_kappa": fleiss_kappa,
        "P_bar": P_bar,
        "P_e": P_e,
        "n_items": n_items,
        "n_raters": n_raters,
        "n_categories": 2,
        "valid_items": valid_items,
        "total_entities": global_total_entities,
    }

# ============================================================
# ============================================================

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from text (utility), extracting the last JSON block by default."""
    if not text:
        return None

    try:
        return json.loads(text)
    except:
        pass

    for pattern in [r'```json\s*(\{.*?\})\s*```', r'```\s*(\{.*?\})\s*```']:
        json_matches = re.findall(pattern, text, re.DOTALL)
        if json_matches:
            try:
                return json.loads(json_matches[-1])
            except:
                pass

    json_matches = re.findall(r'\{[^{}]*\}', text)
    if json_matches:
        for match in reversed(json_matches):
            try:
                return json.loads(match)
            except:
                pass

    return None

def extract_simple_kv(text: str, key: str) -> str:
    """Simple key-value extraction (supports multiple formats), returning the last match by default."""
    pattern1 = rf'"{key}"\s*:\s*"([^"]+)"'
    matches1 = re.findall(pattern1, text)
    if matches1:
        return matches1[-1]

    pattern2 = rf'"{key}"\s*:\s*([^\s,}}]+)'
    matches2 = re.findall(pattern2, text)
    if matches2:
        return matches2[-1].strip('"')

    pattern3 = rf'{key}\s*[:：]\s*["\']?([^"\',\s]+)["\']?'
    matches3 = re.findall(pattern3, text, re.IGNORECASE)
    if matches3:
        return matches3[-1]

    return ""

def parse_label_value_for_classification(val: Any) -> Optional[float]:
    """Convert a single value to a standard label."""
    if isinstance(val, (int, float)):
        return float(val)

    if isinstance(val, str):
        val = val.strip()
        try:
            return float(val)
        except ValueError:
            pass

        m = re.search(r'(\d+(?:\.\d+)?)\s*分', val)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

        m = re.search(r'\((\d+(?:\.\d+)?)\)', val)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

        if '强相关' in val:
            return 2.0
        if '较相关' in val:
            return 1.5
        if '弱相关' in val:
            return 1.0
        if '不相关' in val:
            return 0.0

        if val in ('2', '1.5', '1', '0'):
            return float(val)

        m = re.search(r'(\d+(?:\.\d+)?)', val)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    return None

def extract_social_text_label(annotation_text: str) -> str:
    """
    Social text classification label extraction.
    Format: {"label": "number-text"} or plain text.
    Extracts the text part, stripping the numeric prefix.
    Example: "6-news" -> "news", "guide" -> "guide"
    """
    if not annotation_text:
        return "__PARSE_ERROR__"

    label = None

    data = extract_json_from_text(annotation_text)
    if data and isinstance(data, dict):
        label = data.get("label", "")
    else:
        label = extract_simple_kv(annotation_text, "label")

    if not label:
        label = annotation_text

    return extract_label_text(str(label))

def extract_label_text(val: str) -> str:
    """
    Extract text part from "number-text" format.
    Example: "6-news" -> "news", "guide" -> "guide"
    """
    if not val:
        return ""

    val = str(val).strip()

    m = re.search(r'^\d+\s*[-–—:]\s*(.+)$', val)
    if m:
        return m.group(1).strip()

    return val

def extract_llm_label_generic(annotation_text: str, json_key: str = None,
                               label_map: Dict[str, str] = None,
                               task_type: str = None) -> str:
    """Extract classification label from LLM annotation text."""
    if task_type == "image_text_relevance":
        return extract_image_text_relevance_label(annotation_text)

    if task_type == "image_attribution_recognition":
        return extract_image_attribution_recognition_label(annotation_text)

    if task_type == "social_text_classification":
        return extract_social_text_label(annotation_text)

    json_match = re.search(r"```json\s*(.*?)\s*```", annotation_text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            if isinstance(result, dict):
                if task_type == "image_classification":
                    keys = ["一级品类", "二级品类", "话术分析"]
                    values = []
                    for k in keys:
                        if k in result:
                            values.append(f"{k}:{result[k]}")
                    if values:
                        return "|".join(values)
                    return "|".join([f"{k}:{v}" for k, v in result.items()])

                if json_key and json_key in result:
                    label = str(result[json_key])
                    if label_map and label in label_map:
                        return label_map[label]
                    return label
                first_val = next(iter(result.values()))
                label = str(first_val)
                if label_map and label in label_map:
                    return label_map[label]
                return label
            elif isinstance(result, str):
                if label_map and result in label_map:
                    return label_map[result]
                return result
        except (json.JSONDecodeError, StopIteration):
            pass
    if label_map:
        for from_label, to_label in label_map.items():
            if from_label in annotation_text:
                return to_label
    return "__PARSE_ERROR__"

def extract_image_text_relevance_label(annotation_text: str) -> str:
    """
    Image-text relevance label extraction.
    Format: {"label": "relevance_score"}
    Supports: strong(2), moderate(1.5), weak(1), irrelevant(0)
    """
    if not annotation_text:
        return "__PARSE_ERROR__"

    data = extract_json_from_text(annotation_text)
    if data and isinstance(data, dict):
        pred = data.get("label", "")
        if pred is not None:
            parsed = parse_label_value_for_classification(pred)
            if parsed is not None:
                return str(parsed)
            return str(pred).strip()

    pred = extract_simple_kv(annotation_text, "label")
    if pred:
        parsed = parse_label_value_for_classification(pred)
        if parsed is not None:
            return str(parsed)
        return pred

    if "label" in annotation_text.lower():
        match = re.search(r'[:：]\s*"?([^"\',\n]+)"?', annotation_text, re.IGNORECASE)
        if match:
            pred = match.group(1).strip()
            parsed = parse_label_value_for_classification(pred)
            if parsed is not None:
                return str(parsed)
            return pred

    parsed = parse_label_value_for_classification(annotation_text)
    if parsed is not None:
        return str(parsed)

    return "__PARSE_ERROR__"

def extract_image_attribution_recognition_label(annotation_text: str) -> str:
    if not annotation_text:
        return "0"

    data = extract_json_from_text(annotation_text)
    if data and isinstance(data, dict):
        pred = data.get("label", "")
        if pred is not None:
            return parse_attribution_label(str(pred))

    pred = extract_simple_kv(annotation_text, "label")
    if pred:
        return parse_attribution_label(pred)

    return parse_attribution_label(annotation_text)

def parse_attribution_label(val: str) -> str:
    if val is None:
        return "0"

    val = str(val).strip()

    if val in ("0", "1", "2"):
        return val

    for digit in ["0", "1", "2"]:
        pattern = r"(?:^|[^\\d])" + digit + r"(?:[^\\d]|$)"
        if re.search(pattern, val):
            return digit

    m = re.search(r"[012]", val)
    if m:
        return m.group(0)

    return "0"

def extract_text_part(val: str) -> str:
    if not val:
        return ""
    val = str(val).strip()
    m = re.search(r'^\d+\s*[-–—:]\s*(.+)$', val)
    if m:
        return m.group(1).strip()
    return val

def extract_guidance_part(val: str) -> str:
    if not val:
        return ""
    val = str(val).strip()

    if val.startswith('[') and val.endswith(']'):
        try:
            parsed = json.loads(val.replace("'", '"'))
            if isinstance(parsed, list):
                matches = []
                for item in parsed:
                    item_str = str(item).strip()
                    m = re.search(r'([^，,、\s\[\]\'"]*导向)', item_str)
                    if m:
                        matches.append(m.group(1))
                if matches:
                    return "、".join(matches)
        except:
            pass

    matches = re.findall(r'([^，,、\s\[\]\'"]*导向)', val)
    if matches:
        return "、".join(matches)
    return val

def extract_image_classification_labels(annotation_text: str) -> Dict[str, str]:
    keys = ["一级品类", "二级品类", "话术分析"]
    result_dict = {k: None for k in keys}

    if not annotation_text:
        return result_dict

    json_match = re.search(r"```json\s*(.*?)\s*```", annotation_text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            if isinstance(result, dict):
                if "一级品类" in result:
                    val = result["一级品类"]
                    if isinstance(val, list) and len(val) > 0:
                        result_dict["一级品类"] = extract_text_part(str(val[0]))
                    elif val is not None:
                        result_dict["一级品类"] = extract_text_part(str(val))
                if "二级品类" in result:
                    val = result["二级品类"]
                    if isinstance(val, list) and len(val) > 0:
                        result_dict["二级品类"] = extract_text_part(str(val[0]))
                    elif val is not None:
                        result_dict["二级品类"] = extract_text_part(str(val))
                if "话术分析" in result:
                    val = result["话术分析"]
                    if isinstance(val, list):
                        guidance_items = []
                        for item in val:
                            item_str = str(item).strip()
                            if "导向" in item_str:
                                m = re.search(r'([^，,、\s\[\]\'"]*导向)', item_str)
                                if m:
                                    guidance_items.append(m.group(1))
                        if guidance_items:
                            result_dict["话术分析"] = "、".join(guidance_items)
                        else:
                            result_dict["话术分析"] = "其他"
                    elif val is not None:
                        result_dict["话术分析"] = extract_guidance_part(str(val))
        except json.JSONDecodeError:
            pass

    return result_dict

def normalize_human_label(label: Any, task_type: str = None) -> str:
    """Normalize human annotation labels to match LLM parsing results."""
    if label is None:
        return None

    if task_type == "image_text_relevance":
        parsed = parse_label_value_for_classification(str(label))
        if parsed is not None:
            return str(parsed)
        return str(label).strip()

    if task_type == "image_attribution_recognition":
        return parse_attribution_label(str(label))

    if task_type == "social_text_classification":
        return extract_label_text(str(label))

    return str(label).strip()

def process_classification(file_path: str, task_name: str,
                            json_key: str = None,
                            label_map: Dict[str, str] = None,
                            task_type: str = None) -> Dict[str, Any]:
    """Generic classification processing: compute AC1 & Kappa between LLM and each annotator, then average."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "runs" in data:
        test_results = data["runs"][0]["test_results"]
    elif "test_results" in data:
        test_results = data["test_results"]
    else:
        test_results = data["results"]
    n_items = len(test_results)
    n_annotators = 5

    llm_labels = []
    annotator_labels = [[] for _ in range(n_annotators)]
    skipped = 0
    category_stats = Counter()

    for r in test_results:
        gt = r["ground_truth"]
        all_anns = gt["all_annotations"]
        if "annotation" not in r:
            r["annotation"] = "其他"
        llm_label = extract_llm_label_generic(r["annotation"], json_key, label_map, task_type)
        category_stats[llm_label] += 1

        valid = all(ann is not None for ann in all_anns)
        if not valid:
            skipped += 1

        llm_labels.append(llm_label)
        for j in range(n_annotators):
            raw_label = all_anns[j] if j < len(all_anns) and all_anns[j] is not None else None
            normalized_label = normalize_human_label(raw_label, task_type)
            annotator_labels[j].append(normalized_label)

    pairwise_results = []
    for j in range(n_annotators):
        pair_llm, pair_human = [], []
        for k in range(n_items):
            if annotator_labels[j][k] is not None and llm_labels[k] != "__PARSE_ERROR__":
                pair_llm.append(llm_labels[k])
                pair_human.append(annotator_labels[j][k])

        result = compute_gwet_ac1_pairwise(pair_llm, pair_human)
        result["annotator"] = f"Annotator_{j+1}"
        result["n_valid"] = len(pair_llm)
        pairwise_results.append(result)

    ac1_values = [r["ac1"] for r in pairwise_results]
    kappa_values = [r["kappa"] for r in pairwise_results]
    avg_ac1 = sum(ac1_values) / len(ac1_values)
    avg_kappa = sum(kappa_values) / len(kappa_values)

    human_pairwise = []
    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            pair_i, pair_j = [], []
            for k in range(n_items):
                if annotator_labels[i][k] is not None and annotator_labels[j][k] is not None:
                    pair_i.append(annotator_labels[i][k])
                    pair_j.append(annotator_labels[j][k])
            result = compute_gwet_ac1_pairwise(pair_i, pair_j)
            result["pair"] = f"Annotator_{i+1} vs Annotator_{j+1}"
            human_pairwise.append(result)

    human_ac1_values = [r["ac1"] for r in human_pairwise]
    human_kappa_values = [r["kappa"] for r in human_pairwise]
    avg_human_ac1 = sum(human_ac1_values) / len(human_ac1_values) if human_ac1_values else float("nan")
    avg_human_kappa = sum(human_kappa_values) / len(human_kappa_values) if human_kappa_values else float("nan")

    fleiss_all_matrix = []
    for k in range(n_items):
        row = [llm_labels[k] if llm_labels[k] != "__PARSE_ERROR__" else None]
        for j in range(n_annotators):
            row.append(annotator_labels[j][k])
        fleiss_all_matrix.append(row)
    fleiss_all = compute_fleiss_kappa(fleiss_all_matrix)

    fleiss_human_matrix = []
    for k in range(n_items):
        row = [annotator_labels[j][k] for j in range(n_annotators)]
        fleiss_human_matrix.append(row)
    fleiss_human = compute_fleiss_kappa(fleiss_human_matrix)

    return {
        "task": task_name,
        "total_samples": n_items,
        "skipped_samples_with_none": skipped,
        "n_annotators": n_annotators,
        "n_categories": len(category_stats),
        "category_distribution": dict(category_stats.most_common()),
        "llm_vs_annotator_pairwise": pairwise_results,
        "llm_avg_ac1": avg_ac1,
        "llm_avg_kappa": avg_kappa,
        "human_pairwise": human_pairwise,
        "human_avg_ac1": avg_human_ac1,
        "human_avg_kappa": avg_human_kappa,
        "fleiss_kappa_all": fleiss_all,
        "fleiss_kappa_human": fleiss_human,
    }

def process_image_classification(file_path: str) -> Dict[str, Any]:
    """
    Image classification task processing: compute AC1 & Kappa for three fields separately, then average.
    Fields: primary_category, secondary_category, speech_analysis
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "runs" in data:
        test_results = data["runs"][0]["test_results"]
    elif "test_results" in data:
        test_results = data["test_results"]
    else:
        test_results = data["results"]
    n_items = len(test_results)
    n_annotators = 5

    keys = ["一级品类", "二级品类", "话术分析"]

    llm_labels_per_field = {k: [] for k in keys}
    annotator_labels_per_field = {k: [[] for _ in range(n_annotators)] for k in keys}
    skipped = 0

    llm_category_stats_per_field = {k: Counter() for k in keys}
    human_category_stats_per_field = {k: Counter() for k in keys}
    all_categories_per_field = {k: set() for k in keys}

    for r in test_results:
        gt = r["ground_truth"]
        all_anns = gt["all_annotations"]

        if "annotation" not in r:
            llm_field_labels = {k: "其他" for k in keys}
        else:
            llm_field_labels = extract_image_classification_labels(r["annotation"])
            for k in keys:
                if llm_field_labels[k] is None:
                    llm_field_labels[k] = "其他"

        annotator_field_labels = []
        for j in range(n_annotators):
            if j < len(all_anns) and all_anns[j] is not None:
                try:
                    if isinstance(all_anns[j], str):
                        ann_data = json.loads(all_anns[j])
                    else:
                        ann_data = all_anns[j]
                    field_labels = {}
                    for k in keys:
                        val = ann_data.get(k)
                        if val is None:
                            field_labels[k] = "其他"
                        elif k == "话术分析" and isinstance(val, list):
                            guidance_items = []
                            for item in val:
                                item_str = str(item).strip()
                                if "导向" in item_str:
                                    m = re.search(r'([^，,、\s\[\]\'"]*导向)', item_str)
                                    if m:
                                        guidance_items.append(m.group(1))
                            if guidance_items:
                                field_labels[k] = "、".join(guidance_items)
                            else:
                                field_labels[k] = "其他"
                        elif isinstance(val, list):
                            if len(val) > 0:
                                field_labels[k] = extract_text_part(str(val[0]))
                            else:
                                field_labels[k] = "其他"
                        else:
                            field_labels[k] = extract_text_part(str(val))
                except (json.JSONDecodeError, TypeError):
                    field_labels = {k: "其他" for k in keys}
                annotator_field_labels.append(field_labels)
            else:
                annotator_field_labels.append({k: None for k in keys})

        valid = all(
            annotator_field_labels[j][k] is not None
            for j in range(n_annotators)
            for k in keys
            if j < len(annotator_field_labels)
        )
        if not valid:
            skipped += 1

        for k in keys:
            llm_labels_per_field[k].append(llm_field_labels[k])
            llm_category_stats_per_field[k][llm_field_labels[k]] += 1
            all_categories_per_field[k].add(llm_field_labels[k])
            for j in range(n_annotators):
                label_val = annotator_field_labels[j][k] if j < len(annotator_field_labels) else None
                annotator_labels_per_field[k][j].append(label_val)
                if label_val is not None:
                    human_category_stats_per_field[k][label_val] += 1
                    all_categories_per_field[k].add(label_val)

    field_results = {}
    for k in keys:
        pairwise_results = []
        for j in range(n_annotators):
            pair_llm, pair_human = [], []
            for idx in range(n_items):
                if (annotator_labels_per_field[k][j][idx] is not None and
                    llm_labels_per_field[k][idx] != "__PARSE_ERROR__"):
                    pair_llm.append(llm_labels_per_field[k][idx])
                    pair_human.append(annotator_labels_per_field[k][j][idx])

            result = compute_gwet_ac1_pairwise(pair_llm, pair_human)
            result["annotator"] = f"Annotator_{j+1}"
            result["n_valid"] = len(pair_llm)
            pairwise_results.append(result)

        ac1_values = [r["ac1"] for r in pairwise_results]
        kappa_values = [r["kappa"] for r in pairwise_results]
        avg_ac1 = sum(ac1_values) / len(ac1_values) if ac1_values else float("nan")
        avg_kappa = sum(kappa_values) / len(kappa_values) if kappa_values else float("nan")

        human_pairwise = []
        for i in range(n_annotators):
            for j in range(i + 1, n_annotators):
                pair_i, pair_j = [], []
                for idx in range(n_items):
                    if (annotator_labels_per_field[k][i][idx] is not None and
                        annotator_labels_per_field[k][j][idx] is not None):
                        pair_i.append(annotator_labels_per_field[k][i][idx])
                        pair_j.append(annotator_labels_per_field[k][j][idx])
                result = compute_gwet_ac1_pairwise(pair_i, pair_j)
                result["pair"] = f"Annotator_{i+1} vs Annotator_{j+1}"
                human_pairwise.append(result)

        human_ac1_values = [r["ac1"] for r in human_pairwise]
        human_kappa_values = [r["kappa"] for r in human_pairwise]
        avg_human_ac1 = sum(human_ac1_values) / len(human_ac1_values) if human_ac1_values else float("nan")
        avg_human_kappa = sum(human_kappa_values) / len(human_kappa_values) if human_kappa_values else float("nan")

        fleiss_all_matrix = []
        for idx in range(n_items):
            row = [llm_labels_per_field[k][idx] if llm_labels_per_field[k][idx] != "__PARSE_ERROR__" else None]
            for j in range(n_annotators):
                row.append(annotator_labels_per_field[k][j][idx])
            fleiss_all_matrix.append(row)
        fleiss_all = compute_fleiss_kappa(fleiss_all_matrix)

        fleiss_human_matrix = []
        for idx in range(n_items):
            row = [annotator_labels_per_field[k][j][idx] for j in range(n_annotators)]
            fleiss_human_matrix.append(row)
        fleiss_human = compute_fleiss_kappa(fleiss_human_matrix)

        field_results[k] = {
            "pairwise": pairwise_results,
            "avg_ac1": avg_ac1,
            "avg_kappa": avg_kappa,
            "human_pairwise": human_pairwise,
            "avg_human_ac1": avg_human_ac1,
            "avg_human_kappa": avg_human_kappa,
            "fleiss_kappa_all": fleiss_all,
            "fleiss_kappa_human": fleiss_human,
            "llm_human_category_count": len(all_categories_per_field[k]),
            "human_only_category_count": len(human_category_stats_per_field[k]),
            "llm_category_distribution": dict(llm_category_stats_per_field[k]),
            "human_category_distribution": dict(human_category_stats_per_field[k]),
        }

    all_avg_ac1 = [field_results[k]["avg_ac1"] for k in keys]
    all_avg_kappa = [field_results[k]["avg_kappa"] for k in keys]
    all_avg_human_ac1 = [field_results[k]["avg_human_ac1"] for k in keys]
    all_avg_human_kappa = [field_results[k]["avg_human_kappa"] for k in keys]

    all_fleiss_all = [field_results[k]["fleiss_kappa_all"]["fleiss_kappa"] for k in keys]
    all_fleiss_human = [field_results[k]["fleiss_kappa_human"]["fleiss_kappa"] for k in keys]

    overall_avg_ac1 = sum(all_avg_ac1) / len(all_avg_ac1)
    overall_avg_kappa = sum(all_avg_kappa) / len(all_avg_kappa)
    overall_avg_human_ac1 = sum(all_avg_human_ac1) / len(all_avg_human_ac1)
    overall_avg_human_kappa = sum(all_avg_human_kappa) / len(all_avg_human_kappa)
    overall_fleiss_all = sum(all_fleiss_all) / len(all_fleiss_all)
    overall_fleiss_human = sum(all_fleiss_human) / len(all_fleiss_human)

    return {
        "task": "Image Classification Task (Image Classification - 3-field average)",
        "fields": keys,
        "field_results": field_results,
        "total_samples": n_items,
        "skipped_samples_with_none": skipped,
        "n_annotators": n_annotators,
        "llm_avg_ac1": overall_avg_ac1,
        "llm_avg_kappa": overall_avg_kappa,
        "human_avg_ac1": overall_avg_human_ac1,
        "human_avg_kappa": overall_avg_human_kappa,
        "fleiss_kappa_all": {"fleiss_kappa": overall_fleiss_all},
        "fleiss_kappa_human": {"fleiss_kappa": overall_fleiss_human},
    }

# ============================================================
# ============================================================

def extract_entity_llm_entities(annotation_text: str) -> List[Dict[str, str]]:
    """Extract entity list from LLM annotation text."""
    json_match = re.search(r"```json\s*(.*?)\s*```", annotation_text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            if isinstance(result, dict) and "entities" in result:
                return result["entities"]
            elif isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return []

def entity_to_key(entity: Dict[str, str]) -> Tuple[str, str]:
    return (entity.get("text", ""), entity.get("category", ""))

def compute_entity_level_ac1_kappa(
    llm_entities_list: List[List[Dict]],
    human_entities_list: List[List[Dict]]
) -> Dict[str, Any]:
    """Entity-Level AC1 & Kappa: each candidate entity serves as a binary decision unit."""
    llm_binary, human_binary = [], []

    for llm_ents, human_ents in zip(llm_entities_list, human_entities_list):
        llm_set = set(entity_to_key(e) for e in llm_ents)
        human_set = set(entity_to_key(e) for e in human_ents)
        union_set = llm_set | human_set

        for ek in sorted(union_set):
            llm_binary.append(1 if ek in llm_set else 0)
            human_binary.append(1 if ek in human_set else 0)

    if len(llm_binary) == 0:
        return {"ac1": float("nan"), "kappa": float("nan"),
                "pa": float("nan"), "pe_ac1": float("nan"), "pe_kappa": float("nan"),
                "n_entity_decisions": 0, "agreements": 0}

    result = compute_binary_ac1_kappa(llm_binary, human_binary)
    result["n_entity_decisions"] = len(llm_binary)
    return result

def process_entity_extraction(file_path: str) -> Dict[str, Any]:
    """Process entity extraction file, compute Entity-Level AC1 & Kappa between LLM and each annotator."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "runs" in data:
        test_results = data["runs"][0]["test_results"]
    elif "test_results" in data:
        test_results = data["test_results"]
    else:
        test_results = data["results"]

    n_items = len(test_results)
    n_annotators = 5

    llm_entities_per_sample = []
    annotator_entities_per_sample = [[] for _ in range(n_annotators)]

    for r in test_results:
        gt = r["ground_truth"]
        all_anns = gt["all_annotations"]
        if "annotation" not in r:
            llm_ents = []
        else:
            llm_ents = extract_entity_llm_entities(r["annotation"])
        llm_entities_per_sample.append(llm_ents)
        for j in range(n_annotators):
            if j < len(all_anns) and all_anns[j] is not None:
                annotator_entities_per_sample[j].append(all_anns[j])
            else:
                annotator_entities_per_sample[j].append([])

    # LLM vs each human annotator
    pairwise_results = []
    for j in range(n_annotators):
        result = compute_entity_level_ac1_kappa(
            llm_entities_per_sample, annotator_entities_per_sample[j]
        )
        result["annotator"] = f"Annotator_{j+1}"
        pairwise_results.append(result)

    ac1_values = [r["ac1"] for r in pairwise_results]
    kappa_values = [r["kappa"] for r in pairwise_results]
    avg_ac1 = sum(ac1_values) / len(ac1_values)
    avg_kappa = sum(kappa_values) / len(kappa_values)

    human_pairwise = []
    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            result = compute_entity_level_ac1_kappa(
                annotator_entities_per_sample[i], annotator_entities_per_sample[j]
            )
            result["pair"] = f"Annotator_{i+1} vs Annotator_{j+1}"
            human_pairwise.append(result)

    human_ac1_values = [r["ac1"] for r in human_pairwise]
    human_kappa_values = [r["kappa"] for r in human_pairwise]
    avg_human_ac1 = sum(human_ac1_values) / len(human_ac1_values) if human_ac1_values else float("nan")
    avg_human_kappa = sum(human_kappa_values) / len(human_kappa_values) if human_kappa_values else float("nan")

    micro_stats = compute_entity_micro_prf(
        llm_entities_per_sample, annotator_entities_per_sample, n_annotators
    )

    raters_all = [llm_entities_per_sample] + annotator_entities_per_sample

    exact_all = compute_sample_exact_match_agreement(raters_all, n_items)
    exact_human = compute_sample_exact_match_agreement(annotator_entities_per_sample, n_items)

    f1_all = compute_sample_pairwise_f1(raters_all, n_items)
    f1_human = compute_sample_pairwise_f1(annotator_entities_per_sample, n_items)

    return {
        "task": "Entity Extraction",
        "total_samples": n_items,
        "n_annotators": n_annotators,
        "llm_vs_annotator_pairwise": pairwise_results,
        "llm_avg_ac1": avg_ac1,
        "llm_avg_kappa": avg_kappa,
        "human_pairwise": human_pairwise,
        "human_avg_ac1": avg_human_ac1,
        "human_avg_kappa": avg_human_kappa,
        "micro_precision_recall_f1": micro_stats,
        "exact_match_all": exact_all,
        "exact_match_human": exact_human,
        "pairwise_f1_all": f1_all,
        "pairwise_f1_human": f1_human,
    }

def compute_entity_micro_prf(
    llm_entities_list, annotator_entities_list, n_annotators
):
    """Micro-averaged P/R/F1 (gold: >=3/5)."""
    tp = fp = fn = 0
    for idx in range(len(llm_entities_list)):
        llm_set = set(entity_to_key(e) for e in llm_entities_list[idx])
        entity_counts = Counter()
        for j in range(n_annotators):
            for e in annotator_entities_list[j][idx]:
                entity_counts[entity_to_key(e)] += 1
        majority_set = set(k for k, v in entity_counts.items() if v >= 3)
        tp += len(llm_set & majority_set)
        fp += len(llm_set - majority_set)
        fn += len(majority_set - llm_set)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}

def compute_sample_exact_match_agreement(
    raters_entities_per_sample: List[List[List[Dict]]],
    n_items: int,
) -> Dict[str, Any]:
    n_raters = len(raters_entities_per_sample)
    if n_raters < 2:
        return {"exact_match_agreement": float("nan"), "n_items": n_items,
                "n_raters": n_raters, "valid_items": 0}

    sample_agree_list = []
    n_pairs = n_raters * (n_raters - 1) // 2

    for idx in range(n_items):
        rater_sets = [frozenset(entity_to_key(e) for e in raters_entities_per_sample[r][idx])
                      for r in range(n_raters)]
        matches = 0
        for i in range(n_raters):
            for j in range(i + 1, n_raters):
                if rater_sets[i] == rater_sets[j]:
                    matches += 1
        sample_agree_list.append(matches / n_pairs)

    return {
        "exact_match_agreement": sum(sample_agree_list) / len(sample_agree_list),
        "n_items": n_items,
        "n_raters": n_raters,
        "valid_items": len(sample_agree_list),
    }

def compute_sample_pairwise_f1(
    raters_entities_per_sample: List[List[List[Dict]]],
    n_items: int,
) -> Dict[str, Any]:
    n_raters = len(raters_entities_per_sample)
    if n_raters < 2:
        return {"pairwise_f1": float("nan"), "n_items": n_items,
                "n_raters": n_raters, "valid_items": 0}

    sample_f1_list = []
    n_pairs = n_raters * (n_raters - 1) // 2

    for idx in range(n_items):
        rater_sets = [set(entity_to_key(e) for e in raters_entities_per_sample[r][idx])
                      for r in range(n_raters)]
        pair_f1_sum = 0.0
        for i in range(n_raters):
            for j in range(i + 1, n_raters):
                si, sj = rater_sets[i], rater_sets[j]
                if not si and not sj:
                    pair_f1_sum += 1.0
                    continue
                if not si or not sj:
                    continue
                overlap = len(si & sj)
                p = overlap / len(si)
                r = overlap / len(sj)
                if p + r > 0:
                    pair_f1_sum += 2 * p * r / (p + r)
        sample_f1_list.append(pair_f1_sum / n_pairs)

    return {
        "pairwise_f1": sum(sample_f1_list) / len(sample_f1_list),
        "n_items": n_items,
        "n_raters": n_raters,
        "valid_items": len(sample_f1_list),
    }

def print_image_classification_results(result: Dict[str, Any]):
    print("=" * 78)
    print(f"  {result['task']}")
    print("=" * 78)
    print(f"  Total Samples: {result['total_samples']}    Annotators: {result['n_annotators']}")
    if result.get("skipped_samples_with_none", 0) > 0:
        print(f"  Samples with None annotation: {result['skipped_samples_with_none']}")
    print()

    field_results = result["field_results"]
    keys = result["fields"]

    for k in keys:
        print(f"  ── Field: {k} ──")
        print(f"    LLM+Human Categories: {field_results[k]['llm_human_category_count']}    Human-only Categories: {field_results[k]['human_only_category_count']}")

        print(f"    LLM Category Distribution:")
        for cat, cnt in sorted(field_results[k]["llm_category_distribution"].items(), key=lambda x: -x[1])[:10]:
            print(f"      {cat}: {cnt}")

        print(f"    Human Category Distribution:")
        for cat, cnt in sorted(field_results[k]["human_category_distribution"].items(), key=lambda x: -x[1])[:10]:
            print(f"      {cat}: {cnt}")

        print(f"  {'Annotators':<16} {'AC1':>8} {'Kappa':>8} {'Pa':>8} {'Pe(AC1)':>8} {'Pe(Kappa)':>9} {'Agreements':>6} {'Valid N':>6}")
        print("  " + "-" * 76)
        for r in field_results[k]["pairwise"]:
            print(f"  {r['annotator']:<16} {r['ac1']:>8.4f} {r['kappa']:>8.4f} {r['pa']:>8.4f} "
                  f"{r['pe_ac1']:>8.4f} {r['pe_kappa']:>9.4f} {r['agreements']:>6} {r['n_valid']:>6}")
        print("  " + "-" * 76)
        print(f"  {'Average':<16} {field_results[k]['avg_ac1']:>8.4f} {field_results[k]['avg_kappa']:>8.4f}")
        print()

    print("  ── Among Human Annotators (Ref - Primary Category) ──")
    print(f"  {'Annotator Pair':<33} {'AC1':>8} {'Kappa':>8} {'Pa':>8}")
    print("  " + "-" * 63)
    for r in field_results[keys[0]]["human_pairwise"]:
        print(f"  {r['pair']:<33} {r['ac1']:>8.4f} {r['kappa']:>8.4f} {r['pa']:>8.4f}")
    print("  " + "-" * 63)
    print(f"  {'Human Average':<33} {field_results[keys[0]]['avg_human_ac1']:>8.4f} {field_results[keys[0]]['avg_human_kappa']:>8.4f}")
    print()

    print("  ── Fleiss' Kappa (Multi-rater Agreement) ──")
    print(f"  {'Field':<20} {'Fleiss(w/ LLM)':>14} {'Fleiss(Human)':>14}")
    print("  " + "-" * 50)
    for k in keys:
        fk_all = field_results[k]["fleiss_kappa_all"]["fleiss_kappa"]
        fk_human = field_results[k]["fleiss_kappa_human"]["fleiss_kappa"]
        print(f"  {k:<20} {fk_all:>14.4f} {fk_human:>14.4f}")
    print("  " + "-" * 50)
    print()

    print("  " + "=" * 60)
    print("  [Three-Field Average Results]")
    print(f"  {'Metric':<20} {'LLM vs Human':>12} {'Among Humans':>12}")
    print("  " + "-" * 46)
    print(f"  {'AC1':<20} {result['llm_avg_ac1']:>12.4f} {result['human_avg_ac1']:>12.4f}")
    print(f"  {'Kappa':<20} {result['llm_avg_kappa']:>12.4f} {result['human_avg_kappa']:>12.4f}")
    print("  " + "-" * 46)
    print(f"  {'Fleiss Kappa (w/ LLM)':<20} {result['fleiss_kappa_all']['fleiss_kappa']:>12.4f}")
    print(f"  {'Fleiss Kappa (Human-only)':<20} {result['fleiss_kappa_human']['fleiss_kappa']:>12.4f}")
    print("  " + "=" * 60)
    print()

def print_classification_results(result: Dict[str, Any]):
    print("=" * 78)
    print(f"  {result['task']}")
    print("=" * 78)
    print(f"  Total Samples: {result['total_samples']}    Categories: {result['n_categories']}    Annotators: {result['n_annotators']}")
    if result.get("skipped_samples_with_none", 0) > 0:
        print(f"  Samples with None annotation: {result['skipped_samples_with_none']}")
    if result.get("category_distribution"):
        print(f"  LLM Category Distribution (top 10):")
        for cat, cnt in list(result["category_distribution"].items())[:10]:
            print(f"    {cat}: {cnt}")
    print()
    print("  ── LLM vs Each Annotator ──")
    print(f"  {'Annotators':<16} {'AC1':>8} {'Kappa':>8} {'Pa':>8} {'Pe(AC1)':>8} {'Pe(Kappa)':>9} {'Agreements':>6} {'Valid N':>6}")
    print("  " + "-" * 76)
    for r in result["llm_vs_annotator_pairwise"]:
        print(f"  {r['annotator']:<16} {r['ac1']:>8.4f} {r['kappa']:>8.4f} {r['pa']:>8.4f} "
              f"{r['pe_ac1']:>8.4f} {r['pe_kappa']:>9.4f} {r['agreements']:>6} {r['n_valid']:>6}")
    print("  " + "-" * 76)
    print(f"  {'Average':<16} {result['llm_avg_ac1']:>8.4f} {result['llm_avg_kappa']:>8.4f}")
    print()

    print("  ── Among Human Annotators (Reference) ──")
    print(f"  {'Annotator Pair':<33} {'AC1':>8} {'Kappa':>8} {'Pa':>8}")
    print("  " + "-" * 63)
    for r in result["human_pairwise"]:
        print(f"  {r['pair']:<33} {r['ac1']:>8.4f} {r['kappa']:>8.4f} {r['pa']:>8.4f}")
    print("  " + "-" * 63)
    print(f"  {'Human Average':<33} {result['human_avg_ac1']:>8.4f} {result['human_avg_kappa']:>8.4f}")
    print()

    fleiss_all = result.get("fleiss_kappa_all", {})
    fleiss_human = result.get("fleiss_kappa_human", {})
    print("  ── Fleiss' Kappa (Multi-rater Agreement) ──")
    print(f"  {'Scope':<28} {'Fleiss κ':>10} {'P̄':>8} {'P̄_e':>8} {'Valid Items':>8} {'Categories':>6}")
    print("  " + "-" * 72)
    print(f"  {'LLM + All Annotators':<28} {fleiss_all.get('fleiss_kappa', float('nan')):>10.4f} "
          f"{fleiss_all.get('P_bar', float('nan')):>8.4f} {fleiss_all.get('P_e', float('nan')):>8.4f} "
          f"{fleiss_all.get('valid_items', 0):>8} {fleiss_all.get('n_categories', 0):>6}")
    print(f"  {'Human-onlyAnnotators':<28} {fleiss_human.get('fleiss_kappa', float('nan')):>10.4f} "
          f"{fleiss_human.get('P_bar', float('nan')):>8.4f} {fleiss_human.get('P_e', float('nan')):>8.4f} "
          f"{fleiss_human.get('valid_items', 0):>8} {fleiss_human.get('n_categories', 0):>6}")
    print("  " + "-" * 72)
    print()

def print_entity_results(result: Dict[str, Any]):
    print("=" * 78)
    print(f"  {result['task']}")
    print("=" * 78)
    print(f"  Total Samples: {result['total_samples']}    Annotators: {result['n_annotators']}")
    print(f"  Evaluation: Entity-Level (Exact boundary + type match)")
    print()
    print("  ── LLM vs Each Annotator ──")
    print(f"  {'Annotators':<16} {'AC1':>8} {'Kappa':>8} {'Pa':>8} {'Pe(AC1)':>8} {'Pe(Kappa)':>9} {'Agreements':>6} {'Decisions':>8}")
    print("  " + "-" * 78)
    for r in result["llm_vs_annotator_pairwise"]:
        print(f"  {r['annotator']:<16} {r['ac1']:>8.4f} {r['kappa']:>8.4f} {r['pa']:>8.4f} "
              f"{r['pe_ac1']:>8.4f} {r['pe_kappa']:>9.4f} {r['agreements']:>6} {r['n_entity_decisions']:>8}")
    print("  " + "-" * 78)
    print(f"  {'Average':<16} {result['llm_avg_ac1']:>8.4f} {result['llm_avg_kappa']:>8.4f}")
    print()

    print("  ── Among Human Annotators (Reference) ──")
    print(f"  {'Annotator Pair':<33} {'AC1':>8} {'Kappa':>8} {'Pa':>8}")
    print("  " + "-" * 63)
    for r in result["human_pairwise"]:
        print(f"  {r['pair']:<33} {r['ac1']:>8.4f} {r['kappa']:>8.4f} {r['pa']:>8.4f}")
    print("  " + "-" * 63)
    print(f"  {'Human Average':<33} {result['human_avg_ac1']:>8.4f} {result['human_avg_kappa']:>8.4f}")
    print()

    micro = result["micro_precision_recall_f1"]
    print("  ── Micro Precision / Recall / F1 (vs majority vote gold) ──")
    print(f"  TP={micro['tp']}  FP={micro['fp']}  FN={micro['fn']}")
    print(f"  Precision: {micro['precision']:.4f}   Recall: {micro['recall']:.4f}   F1: {micro['f1']:.4f}")
    print()

    fleiss_all = result.get("fleiss_all", {})
    fleiss_human = result.get("fleiss_human", {})
    if fleiss_all:
        print("  ── Fleiss' Kappa (Candidate set = all raters incl. LLM) ──")
        print(f"  {'Scope':<20} {'Fleiss κ':>10} {'P̄':>8} {'Pe':>10} {'Valid Items':>8}")
        print("  " + "-" * 58)
        for tag, data in [("LLM+Annotators", fleiss_all), ("Human-only", fleiss_human)]:
            print(f"  {tag:<20} {data.get('fleiss_kappa', float('nan')):>10.4f} "
                  f"{data.get('P_bar', float('nan')):>8.4f} "
                  f"{data.get('P_e', float('nan')):>10.4f} "
                  f"{data.get('valid_items', 0):>8}")
        print("  " + "-" * 58)
        print()

    exact_all = result.get("exact_match_all", {})
    exact_human = result.get("exact_match_human", {})
    if exact_all or exact_human:
        print("  ── Sample-level Exact Match Agreement ──")
        print(f"  {'Scope':<20} {'Exact Match':>12} {'Valid Items':>8}")
        print("  " + "-" * 44)
        for tag, data in [("LLM+Annotators", exact_all), ("Human-only", exact_human)]:
            print(f"  {tag:<20} {data.get('exact_match_agreement', float('nan')):>12.4f} "
                  f"{data.get('valid_items', 0):>8}")
        print("  " + "-" * 44)
        print()

    f1_all = result.get("pairwise_f1_all", {})
    f1_human = result.get("pairwise_f1_human", {})
    if f1_all or f1_human:
        print("  ── Sample-level Pairwise F1 Agreement ──")
        print(f"  {'Scope':<20} {'Pairwise F1':>12} {'Valid Items':>8}")
        print("  " + "-" * 44)
        for tag, data in [("LLM+Annotators", f1_all), ("Human-only", f1_human)]:
            print(f"  {tag:<20} {data.get('pairwise_f1', float('nan')):>12.4f} "
                  f"{data.get('valid_items', 0):>8}")
        print("  " + "-" * 44)
        print()

# ============================================================

def detect_task_type(file_path: str) -> str:
    """Auto-detect task type from file path."""
    file_name = os.path.basename(file_path).lower()
    path_lower = file_path.lower()

    if "medical_query" in path_lower or "medicalquery" in file_name:
        return "medical_query_evaluation"
    elif "billing" in path_lower or "billing_scenario" in path_lower:
        return "billing_scenario_classification"
    elif "social" in path_lower:
        return "social_text_classification"
    elif "entity" in path_lower or "ner" in file_name:
        return "entity_extraction"
    elif "image_classification" in path_lower or ("image" in path_lower and "classification" in path_lower):
        return "image_classification"
    elif "image_text" in path_lower or "imagetext" in file_name or "text_relevance" in path_lower:
        return "image_text_relevance"
    elif "attribution" in path_lower or "image_attribution" in path_lower:
        return "image_attribution_recognition"
    elif "classification" in path_lower:
        return "classification"
    else:
        return "classification"  # default classification task

def process_task(file_path: str, task_type: str = None) -> Dict[str, Any]:
    """
    Generic task processing entry point, automatically selects handler by task type.

    Args:
        file_path: Path to the result file.
        task_type: Task type (optional, auto-detected if not provided).
    """
    if not os.path.exists(file_path):
        print(f"  Warning: File not found - {file_path}")
        return None

    if task_type is None:
        task_type = detect_task_type(file_path)

    task_name = TASK_NAMES.get(task_type, f"Classification Task ({task_type})")
    json_key = TASK_JSON_KEYS.get(task_type)
    label_map = TASK_LABEL_MAPS.get(task_type)

    if task_type == "entity_extraction":
        return process_entity_extraction(file_path)
    elif task_type == "image_classification":
        return process_image_classification(file_path)
    else:
        return process_classification(
            file_path,
            task_name=task_name,
            json_key=json_key,
            label_map=label_map,
            task_type=task_type
        )

def main():
    parser = argparse.ArgumentParser(
        description="Compute Gwet's AC1 & Cohen's Kappa & Fleiss' Kappa consistency metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Note:
  When the number of categories is large (e.g., 29), Gwet's AC1 pe = Σ qₖ(1-qₖ) may approach 1,
  causing AC1 to be low or even extremely negative. In such cases, Cohen's Kappa or Fleiss' Kappa are more reliable complementary metrics.

Supported task types:
  - medical_query_evaluation: Medical query privacy compliance evaluation
  - billing_scenario_classification: Billing scenario classification
  - classification: Generic classification
  - social_text_classification: Social text classification
  - entity_extraction: Entity extraction
  - image_classification: Image classification
  - image_text_relevance: Image-text relevance
  - image_attribution_recognition: Image attribution recognition
        """,
    )
    parser.add_argument(
        "--cls", type=str,
        default=None,
        help="Path to medical query classification result file"
    )
    parser.add_argument(
        "--billing", type=str,
        default=None,
        help="Path to billing scenario classification result file"
    )
    parser.add_argument(
        "--ner", type=str,
        default=None,
        help="Path to entity extraction result file"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Single task result file path (generic, auto-detects task type)"
    )
    parser.add_argument(
        "--task-type", type=str, default=None,
        help="Specify task type (auto-detected if not provided)"
    )
    parser.add_argument(
        "--files", type=str, nargs="+", default=None,
        help="Multiple task result file paths (batch processing)"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save results to JSON file (optional)"
    )
    args = parser.parse_args()

    all_results = {}
    results_list = []  # For summary table

    print("\n" + "█" * 78)
    print("█  Gwet's AC1 & Cohen's Kappa & Fleiss' Kappa Consistency Analysis Report")
    print("█" * 78 + "\n")

    if args.file:
        print(f"Processing file: {args.file}")
        result = process_task(args.file, args.task_type)
        if result:
            task_key = args.task_type or detect_task_type(args.file)
            if result.get("task", "").startswith("Entity Extraction"):
                print_entity_results(result)
            elif result.get("task", "").startswith("Image Classification"):
                print_image_classification_results(result)
            else:
                print_classification_results(result)
            all_results[task_key] = result
            results_list.append((task_key, result))

    elif args.files:
        for file_path in args.files:
            print(f"Processing file: {file_path}")
            result = process_task(file_path)
            if result:
                task_key = detect_task_type(file_path)
                if result.get("task", "").startswith("Entity Extraction"):
                    print_entity_results(result)
                elif result.get("task", "").startswith("Image Classification"):
                    print_image_classification_results(result)
                else:
                    print_classification_results(result)
                all_results[task_key] = result
                results_list.append((task_key, result))

    else:
        if os.path.exists(args.cls):
            cls_result = process_classification(
                args.cls,
                task_name="Classification Task (Medical Query Evaluation - Privacy Compliance)",
                json_key="隐私合规评估",
                label_map={"无问题": "无", "有问题": "有"},
                task_type="medical_query_evaluation"
            )
            print_classification_results(cls_result)
            all_results["classification_medical_query"] = cls_result
            results_list.append(("Medical Query Classification", cls_result))
        else:
            print(f"  Skipped: Medical query classification file not found - {args.cls}")

        if os.path.exists(args.billing):
            billing_result = process_classification(
                args.billing,
                task_name="Classification Task (Billing Scenario Classification)",
                json_key="账单类目",
                label_map=None,
                task_type="billing_scenario_classification"
            )
            print_classification_results(billing_result)
            all_results["classification_billing_scenario"] = billing_result
            results_list.append(("Billing Scenario Classification", billing_result))
        else:
            print(f"  Skipped: Billing scenario classification file not found - {args.billing}")

        if os.path.exists(args.ner):
            ner_result = process_entity_extraction(args.ner)
            print_entity_results(ner_result)
            all_results["entity_extraction"] = ner_result
            results_list.append(("Entity Extraction Task", ner_result))
        else:
            print(f"  Skipped: Entity extraction file not found - {args.ner}")

    if results_list:
        print("=" * 100)
        print("  Summary Comparison")
        print("=" * 100)
        print(f"  {'Task':<22} {'LLM AC1':>10} {'Human AC1':>10} {'LLM Kappa':>10} {'Human Kappa':>10} "
              f"{'Fleiss(w/ LLM)':>14} {'Fleiss(Human)':>13}")
        print("  " + "-" * 97)
        for label, res in results_list:
            fleiss_all_data = res.get("fleiss_all")
            if fleiss_all_data:
                fk_all = fleiss_all_data.get("fleiss_kappa", float("nan"))
                fk_human = res.get("fleiss_human", {}).get("fleiss_kappa", float("nan"))
            else:
                fk_all = res.get("fleiss_kappa_all", {}).get("fleiss_kappa", float("nan"))
                fk_human = res.get("fleiss_kappa_human", {}).get("fleiss_kappa", float("nan"))
            print(f"  {label:<22} {res['llm_avg_ac1']:>10.4f} {res['human_avg_ac1']:>10.4f} "
                  f"{res['llm_avg_kappa']:>10.4f} {res['human_avg_kappa']:>10.4f} "
                  f"{fk_all:>14.4f} {fk_human:>13.4f}")
        print("  " + "-" * 97)
        print()

    if args.save and all_results:
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, float) and (obj != obj):
                return None
            return obj

        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(make_serializable(all_results), f, ensure_ascii=False, indent=2)
        print(f"  Saved: {args.save}")

if __name__ == "__main__":
    main()
