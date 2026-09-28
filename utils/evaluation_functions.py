# evaluation_functions.py
import json
import re
from typing import Any, Dict, Optional, List


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from text (generic utility), default extracts the last JSON block"""
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


# ============================================================================
# ============================================================================

def extract_value_by_regex(text: str, key: str, pattern: str = r'"' + 'key' + r'"\s*:\s*"([^"]+)"') -> str:
    """Extract value for a given key using regex"""
    try:
        pattern = pattern.replace('key', key)
        match = re.search(pattern, text)
        return match.group(1) if match else ""
    except:
        return ""

def extract_simple_kv(text: str, key: str) -> str:
    """Simple key-value extraction (supports multiple formats), defaults to the last match"""
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


# ============================================================================
# ============================================================================

def evaluate_billing_scenario(annotation: str, answer: Any) -> bool:
    """Billing scenario classification evaluation
    Fallback: extract value after key
    """
    try:
        data = extract_json_from_text(annotation)
        if data:
            pred = data.get("账单类目", "")
            if pred:
                return _compare_answer(pred, answer)
        
        pred = extract_simple_kv(annotation, "账单类目")
        if pred:
            return _compare_answer(pred, answer)
        
        if "账单类目" in annotation:
            parts = annotation.split("账单类目")
            if len(parts) > 1:
                after_key = parts[1]
                match = re.search(r'[:：]\s*["\']?([^"\',\n]+)', after_key)
                if match:
                    pred = match.group(1).strip()
                    return _compare_answer(pred, answer)
        if "```json" in annotation:
            annotation = annotation.split("```json")[-1]
        return _compare_answer(annotation.strip(), answer)
        
    except Exception as e:
        print(f"⚠️ 账单场景评估失败: {e}")
        return False


def evaluate_disease_privacy(annotation: str, answer: Any) -> bool:
    """Disease privacy assessment evaluation
    Fallback: match field by field
    Supports multi-annotation format: answer can be string or dict
    """
    if answer in ["有", "无"]:
        return evaluate_disease_privacy_simple(annotation=annotation, answer=answer)
    try:
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except:
                data = extract_json_from_text(annotation)
                if data:
                    pred_val = data.get("隐私合规评估", "").strip().strip('"\'').strip()
                    return pred_val == answer.strip().strip('"\'').strip()
                return False

        data = extract_json_from_text(annotation)
        if data:
            if data["隐私合规评估"].strip().strip('"\'').strip() == answer["隐私合规评估"].strip().strip('"\'').strip():
                return True

        fields = ["隐私合规评估"]
        for field in fields:
            pred_val = extract_simple_kv(annotation, field)

            ans_val = answer.get(field, "") if isinstance(answer, dict) else ""

            if pred_val and ans_val:
                if pred_val == ans_val and field == "隐私合规评估":
                    return True

        return False

    except Exception as e:
        print(f"⚠️ 医疗Query评估失败: {e}")
        return False


def evaluate_disease_privacy_simple(annotation: str, answer: str) -> bool:
    """Disease privacy assessment (simplified)
    Handles cases where answer is boolean-like Chinese strings.
    """
    try:
        expected_value = str(answer).strip().strip('"\'').strip()
        if expected_value not in ["有", "无"]:
            return False
        
        def normalize_pred(val: str) -> str:
            val_clean = val.strip().strip('"\'').strip()
            if val_clean.startswith("有"):
                return "有"
            elif val_clean.startswith("无"):
                return "无"
            return ""
        
        data = extract_json_from_text(annotation)
        if data:
            if "隐私合规评估" in data:
                pred_val = str(data["隐私合规评估"]).strip().strip('"\'').strip()
                return normalize_pred(pred_val) == expected_value
            return False
        
        pred_val = extract_simple_kv(annotation, "隐私合规评估")
        if pred_val:
            return normalize_pred(pred_val) == expected_value
        
        annotation_clean = annotation.strip().strip('"\'').strip()
        if annotation_clean in ["有问题", "无问题", "有", "无"]:
            return normalize_pred(annotation_clean) == expected_value
        
        if "隐私合规评估" in annotation:
            match = re.search(r'隐私合规评估\s*[:：]\s*["\']?(有|无)[问题]?["\']?', annotation)
            if match:
                return match.group(1) == expected_value
        
        has_you = "有" in annotation
        has_wu = "无" in annotation
        if has_you and not has_wu:
            return expected_value == "有"
        if has_wu and not has_you:
            return expected_value == "无"
        
        return False
        
    except Exception as e:
        print(f"⚠️ 医疗Query评估(简化版)失败: {e}")
        return False


def normalize_category_for_corpus(category: str) -> str:
    """
    Normalize classification labels, handle different representations.
    """
    if not category:
        return category
    
    category = category.strip()
    
    return category

def evaluate_corpus_classification(annotation: str, answer: Any) -> bool:
    """Corpus classification evaluation
    Format: {"label": "result"}
    Fallback: extract value after "label"
    Supports multi-annotation format: answer can be string or dict
    """
    if isinstance(answer, dict):
        answer = answer.get("label", "")
    answer = normalize_category_for_corpus(str(answer))

    try:
        data = extract_json_from_text(annotation)
        if data:
            pred = data.get("label", "")
            pred = normalize_category_for_corpus(str(pred))
            if pred:
                return _compare_answer(pred, answer)

        pred = extract_simple_kv(annotation, "label")
        if pred:
            pred = normalize_category_for_corpus(str(pred))
            return _compare_answer(pred, answer)

        pred = normalize_category_for_corpus(annotation.split("```json")[-1])
        return _compare_answer(pred.strip(), answer)

    except Exception as e:
        print(f"social text classification evaluate failed: {e}")
        return False



def extract_ground_truth_score_for_classification(ground_truth: str) -> Optional[float]:
    """Extract score from ground_truth string"""
    match = re.search(r'(\d+\.?\d*)分', ground_truth)
    if match:
        return float(match.group(1))
    return None


def parse_label_value_for_classification(val: Any) -> Optional[float]:
    """Convert a single value to standard label"""
    if isinstance(val, (int, float)):
        return float(val)
    
    if isinstance(val, str):
        val = val.strip()
        try:
            return float(val)
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
    
    return None


def parse_label_value_from_text(text: str) -> Optional[float]:
    """Extract numeric value from text
    
    Rules:
    - If text contains exactly one number (int or float), return it
    - If multiple numbers exist, return None
    - If no number found, return None
    
    Args:
        text: input text
    
    Returns:
        Optional[float]: extracted number, or None
    """
    if not text or not isinstance(text, str):
        return None
    
    pattern = r'[+-]?(?:\d+\.?\d*|\.\d+)'
    matches = re.findall(pattern, text)
    
    if len(matches) == 1:
        try:
            return float(matches[0])
        except ValueError:
            return None
    
    return None

def parse_string_list(val: Any) -> List[str]:
    """Parse string list, supports JSON list, string representation of list, or single string"""
    if isinstance(val, list):
        return [str(v).strip().strip('"\'').strip() for v in val if v is not None]
    if isinstance(val, str):
        val = val.strip()
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(v).strip().strip('"\'').strip() for v in parsed if v is not None]
        except (json.JSONDecodeError, ValueError):
            pass
        matches = re.findall(r'"([^"]+)"', val)
        if matches:
            return [m.strip() for m in matches]
        cleaned = val.strip('"\'[] ')
        if cleaned:
            return [cleaned]
    return []



def parse_ground_truth_for_entity(gt_input: Any) -> List[Dict[str, str]]:
    """Parse ground_truth, supports string or list format, returns empty list on failure"""
    entities = []
    try:
        if isinstance(gt_input, list):
            for ent in gt_input:
                if isinstance(ent, dict) and "text" in ent and "category" in ent:
                    entities.append({
                        "text": str(ent["text"]).strip(),
                        "category": str(ent["category"]).strip()
                    })
            return entities

        gt_str = str(gt_input).strip()
        if gt_str.startswith('"') and gt_str.endswith('"'):
            gt_str = gt_str[1:-1].replace('\\"', '"')

        for ent in json.loads(gt_str):
            if isinstance(ent, dict) and "text" in ent and "category" in ent:
                entities.append({
                    "text": str(ent["text"]).strip(),
                    "category": str(ent["category"]).strip()
                })
    except:
        pass

    return entities


def evaluate_entity_extraction(annotation: str, answer: Any) -> bool:
    """Medical entity extraction evaluation
    Format: {"entities": [{"text": "entity text","category": "category name"}]}
    Fallback: extract entity list
    Supports multi-annotation format: answer can be list [{"text": "...", "category": "..."}, ...]
    """
    answer = parse_ground_truth_for_entity(answer)
    try:
        data = extract_json_from_text(annotation)
        if data:
            pred_entities = data.get("entities", [])
            if pred_entities:
                return _compare_entities(pred_entities, answer)
        
        entity_pattern = r'"text"\s*:\s*"([^"]+)"\s*,\s*"category"\s*:\s*"([^"]+)"'
        matches = re.findall(entity_pattern, annotation, re.IGNORECASE)
        
        if matches:
            pred_entities = [{"text": text, "category": cat} for text, cat in matches]
            if pred_entities:
                return _compare_entities(pred_entities, answer)
        
        simple_pattern = r'([^\s:,;]+)\s*[:：]\s*([^\s,;]+)'
        simple_matches = re.findall(simple_pattern, annotation)
        
        if simple_matches:
            pred_entities = [{"text": text, "category": cat} for text, cat in simple_matches]
            return _compare_entities(pred_entities, answer)
        
        return False
        
    except Exception as e:
        print(f"⚠️ 实体识别评估失败: {e}")
        return False


def evaluate_image_text_relevance(annotation: str, answer: Any) -> bool:
    """Image-text relevance evaluation
    Format: {"label": "relevance score"}
    Fallback: extract value after "label"
    """
    gt_score = float(re.search(r'(\d+\.?\d*)', answer).group(1))
    try:
        data = extract_json_from_text(annotation)
        if data:
            pred = data.get("label", "")
            pred_score = parse_label_value_for_classification(pred)
            is_valid = pred_score is not None and gt_score is not None
            is_correct = is_valid and pred_score == gt_score
            return is_correct
        
        pred = extract_simple_kv(annotation, "label")
        if pred:
            pred_score = parse_label_value_for_classification(pred)
            is_valid = pred_score is not None and gt_score is not None
            is_correct = is_valid and pred_score == gt_score
            return is_correct
        
        if "label" in annotation:
            patterns = [
                r'[:：]\s*"?([^"\',\n]+)"?'
            ]
            for pattern in patterns:
                match = re.search(pattern, annotation)
                if match:
                    pred = match.group(1).strip()
                    pred_score = parse_label_value_for_classification(pred)
                    is_valid = pred_score is not None and gt_score is not None
                    is_correct = is_valid and pred_score == gt_score
                    return is_correct
        
        pred_score = parse_label_value_for_classification(annotation.split("```json")[-1])
        is_valid = pred_score is not None and gt_score is not None
        is_correct = is_valid and pred_score == gt_score
        return is_correct
        
    except Exception as e:
        print(f"⚠️ 相关性评估失败: {e}")
        return False


def evaluate_image_attribution_recognition(annotation: str, answer: Any) -> bool:
    """Image recognizable attribute count
    Format: {"label": " recognizable attribute count (at most 2)"}
    Fallback: extract value after "label"
    """
    gt_score = extract_ground_truth_score_for_classification(answer)
    try:
        data = extract_json_from_text(annotation)
        if data:
            pred = data.get("label", "")
            pred_score = parse_label_value_from_text(pred)
            is_valid = pred_score is not None and gt_score is not None
            is_correct = is_valid and pred_score == gt_score
            return is_correct
        
        pred = extract_simple_kv(annotation, "label")
        if pred:
            pred_score = parse_label_value_from_text(pred)
            is_valid = pred_score is not None and gt_score is not None
            is_correct = is_valid and pred_score == gt_score
            return is_correct
        
        if "label" in annotation:
            patterns = [
                r'[:：]\s*"?([^"\',\n]+)"?'
            ]
            for pattern in patterns:
                match = re.search(pattern, annotation)
                if match:
                    pred = match.group(1).strip()
                    pred_score = parse_label_value_from_text(pred)
                    is_valid = pred_score is not None and gt_score is not None
                    is_correct = is_valid and pred_score == gt_score
                    return is_correct
        
        pred_score = parse_label_value_from_text(annotation.split("```json")[-1])
        is_valid = pred_score is not None and gt_score is not None
        is_correct = is_valid and pred_score == gt_score
        return is_correct
        
    except Exception as e:
        print(f"⚠️ 相关性评估失败: {e}")
        return False


def extract_text_part(val: str) -> str:
    """
    Extract text part, remove numeric prefix etc.
    """
    if not val:
        return ""
    val = str(val).strip()
    m = re.search(r'^\d+\s*[-–—:]\s*(.+)$', val)
    if m:
        return m.group(1).strip()
    return val


def extract_guidance_part(val: str) -> str:
    """
    Extract text parts containing guidance direction keywords.
    """
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


def parse_image_classification_field(val: Any, field: str) -> str:
    """
    Parse image classification field values, referencing compute_gwet_ac1.py logic.
    """
    if val is None:
        return "其他"

    if field in ["一级品类", "二级品类"]:
        if isinstance(val, list) and len(val) > 0:
            return extract_text_part(str(val[0]))
        else:
            return extract_text_part(str(val))
    elif field == "话术分析":
        if isinstance(val, list):
            guidance_items = []
            for item in val:
                item_str = str(item).strip()
                if "导向" in item_str:
                    m = re.search(r'([^，,、\s\[\]\'"]*导向)', item_str)
                    if m:
                        guidance_items.append(m.group(1))
            if guidance_items:
                return "、".join(guidance_items)
            return "其他"
        else:
            result = extract_guidance_part(str(val))
            return result if result else "其他"

    return str(val)


def evaluate_image_classification(annotation: str, answer: Any) -> bool:
    """Image classification evaluation
    Uses parsing logic from compute_gwet_ac1.py
    """
    required_fields = ["一级品类", "二级品类", "话术分析"]

    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except:
            return False
    if not isinstance(answer, dict):
        return False

    answer_norm = {}
    for field in required_fields:
        answer_norm[field] = parse_image_classification_field(answer.get(field), field)

    try:
        json_match = re.search(r"```json\s*(.*?)\s*```", annotation, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                if isinstance(result, dict):
                    all_match = True
                    for field in required_fields:
                        pred = parse_image_classification_field(result.get(field), field)
                        if pred != answer_norm[field]:
                            all_match = False
                            break
                    if all_match:
                        return True
            except json.JSONDecodeError:
                pass

        data = extract_json_from_text(annotation)
        if data and isinstance(data, dict):
            all_match = True
            for field in required_fields:
                pred = parse_image_classification_field(data.get(field), field)
                if pred != answer_norm[field]:
                    all_match = False
                    break
            if all_match:
                return True

        preds = {}
        for field in required_fields:
            val = extract_simple_kv(annotation, field)
            preds[field] = parse_image_classification_field(val, field)

        all_match = True
        for field in required_fields:
            if preds[field] != answer_norm[field]:
                all_match = False
                break
        if all_match:
            return True

        preds = {}
        for field in required_fields:
            pattern_list = rf'"{field}"\s*:\s*\[(.*?)\]'
            match = re.search(pattern_list, annotation, re.DOTALL)
            if match:
                list_content = match.group(1)
                try:
                    parsed = json.loads(f"[{list_content}]")
                    preds[field] = parse_image_classification_field(parsed, field)
                except:
                    preds[field] = parse_image_classification_field(list_content, field)
            else:
                pattern_str = rf'"{field}"\s*:\s*"([^"]*)"'
                match = re.search(pattern_str, annotation)
                if match:
                    preds[field] = parse_image_classification_field(match.group(1), field)
                else:
                    preds[field] = "其他"

        all_match = True
        for field in required_fields:
            if preds[field] != answer_norm[field]:
                all_match = False
                break
        if all_match:
            return True

        return False

    except Exception as e:
        print(f"⚠️ 图片分类评估失败: {e}")
        return False


# ============================================================================
# ============================================================================

def _compare_answer(pred: str, answer: Any) -> bool:
    """Flexible answer comparison (supports string, list, fuzzy match)"""
    if not pred:
        return False
    
    pred_norm = str(pred).strip().strip('"\'').strip()
    
    if isinstance(answer, list):
        answer_norm = [str(a).strip().strip('"\'').strip() for a in answer]
        return pred_norm in answer_norm
    
    answer_norm = str(answer).strip().strip('"\'').strip()
    
    return (pred_norm == answer_norm or 
            pred_norm in answer_norm or 
            answer_norm in pred_norm)


def _compare_entities(pred_entities: List[Dict], answer_entities: List[Dict]) -> bool:
    """Compare entity lists (supports partial match)"""
    if not isinstance(answer_entities, list):
        return False
    
    if len(pred_entities) != len(answer_entities):
        return False
    
    for pred_entity in pred_entities:
        pred_text = str(pred_entity.get("text", "")).strip()
        pred_cat = str(pred_entity.get("category", "")).strip()
        
        match_found = False
        for ans_entity in answer_entities:
            ans_text = str(ans_entity.get("text", "")).strip()
            ans_cat = str(ans_entity.get("category", "")).strip()
            
            if pred_text == ans_text and pred_cat == ans_cat:
                match_found = True
                break
        
        if not match_found:
            return False
    
    return True


# ============================================================================
# ============================================================================

def extract_label_from_annotation(annotation: str, task_type: str) -> str:
    """
    Extract label value from annotation result, used for disagreement detection in multi-Agent debate.
    
    Extraction logic fully references corresponding evaluate functions, using multi-layer fallback:
    1. JSON extraction (extract_json_from_text)
    2. Rule parsing (extract_simple_kv)
    3. Fuzzy match (regex extraction)
    4. Cleaned raw string
    
    Args:
        annotation: raw annotation result string
        task_type: task type
    
    Returns:
        extracted label string for consistency comparison
    """
    if not annotation:
        return ""
    
    cleaned_annotation = annotation
    if "```json" in cleaned_annotation:
        cleaned_annotation = cleaned_annotation.split("```json")[-1]
    if "```" in cleaned_annotation:
        parts = cleaned_annotation.split("```")
        for part in parts:
            if '{' in part or '"' in part:
                cleaned_annotation = part
                break
    
    # ============== billing_scenario_classification ==============
    if task_type == "billing_scenario_classification":
        data = extract_json_from_text(cleaned_annotation)
        if data:
            label = data.get("账单类目", "")
            if label:
                return str(label).strip().strip('"\'').strip()
        
        label = extract_simple_kv(cleaned_annotation, "账单类目")
        if label:
            return label
        
        if "账单类目" in cleaned_annotation:
            parts = cleaned_annotation.split("账单类目")
            if len(parts) > 1:
                after_key = parts[-1]
                match = re.search(r'[:：]\s*["\']?([^"\',\n]+)', after_key)
                if match:
                    return match.group(1).strip()
        
        return cleaned_annotation.strip()
    
    # ============== disease_privacy_assessment ==============
    elif task_type == "disease_privacy_assessment":
        fields = ["隐私合规评估"]
        results = {}
        
        data = extract_json_from_text(cleaned_annotation)
        if data:
            for field in fields:
                results[field] = str(data.get(field, "")).strip().strip('"\'').strip()
            return f"{results.get('隐私合规评估', '')}"
        
        for field in fields:
            val = extract_simple_kv(cleaned_annotation, field)
            if val:
                results[field] = val
        
        if results:
            return f"{results.get('隐私合规评估', '')}"
        
        for field in fields:
            if field in cleaned_annotation:
                parts = cleaned_annotation.split(field)
                if len(parts) > 1:
                    after_key = parts[-1]
                    match = re.search(r'[:：]\s*["\']?([^"\',\n]+)', after_key)
                    if match:
                        results[field] = match.group(1).strip()
        
        if results:
            return f"{results.get('隐私合规评估', '')}"
        
        return cleaned_annotation.strip()
    
    
    # ============== classification ==============
    elif task_type == "classification":
        data = extract_json_from_text(cleaned_annotation)
        if data:
            label = data.get("label", "")
            if label is not None:
                parsed = parse_label_value_for_classification(str(label))
                if parsed is not None:
                    return str(parsed)
                return str(label).strip()
        
        label = extract_simple_kv(cleaned_annotation, "label")
        if label:
            parsed = parse_label_value_for_classification(label)
            if parsed is not None:
                return str(parsed)
            return label
        
        if "label" in cleaned_annotation.lower():
            match = re.search(r'[:：]\s*["\']?([^"\',\n]+)', cleaned_annotation, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                parsed = parse_label_value_for_classification(val)
                if parsed is not None:
                    return str(parsed)
                return val
        
        parsed = parse_label_value_for_classification(cleaned_annotation)
        if parsed is not None:
            return str(parsed)
        return cleaned_annotation.strip()
    
    # ============== entity_extraction ==============
    elif task_type == "entity_extraction":
        data = extract_json_from_text(cleaned_annotation)
        if data:
            entities = data.get("entities", [])
            if isinstance(entities, list):
                entity_strs = []
                for e in entities:
                    if isinstance(e, dict):
                        text = str(e.get("text", "")).strip()
                        cat = str(e.get("category", "")).strip()
                        if text and cat:
                            entity_strs.append(f"{text}:{cat}")
                if entity_strs:
                    return "|".join(sorted(entity_strs))
        
        entity_pattern = r'"text"\s*:\s*"([^"]+)"\s*,\s*"category"\s*:\s*"([^"]+)"'
        matches = re.findall(entity_pattern, cleaned_annotation, re.IGNORECASE)
        if matches:
            entity_strs = [f"{text}:{cat}" for text, cat in matches]
            return "|".join(sorted(entity_strs))
        
        simple_pattern = r'([^\s:,;]+)\s*[:：]\s*([^\s,;]+)'
        simple_matches = re.findall(simple_pattern, cleaned_annotation)
        if simple_matches:
            entity_strs = [f"{text}:{cat}" for text, cat in simple_matches]
            return "|".join(sorted(entity_strs))
        
        return cleaned_annotation.strip()

    
    # ============== image_text_relevance ==============
    elif task_type == "image_text_relevance":
        data = extract_json_from_text(cleaned_annotation)
        if data:
            label = data.get("label", "")
            if label is not None:
                parsed = parse_label_value_for_classification(str(label))
                if parsed is not None:
                    return str(parsed)
                return str(label).strip()
        
        label = extract_simple_kv(cleaned_annotation, "label")
        if label:
            parsed = parse_label_value_for_classification(label)
            if parsed is not None:
                return str(parsed)
            return label
        
        if "label" in cleaned_annotation.lower():
            match = re.search(r'[:：]\s*["\']?([^"\',\n]+)', cleaned_annotation, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                parsed = parse_label_value_for_classification(val)
                if parsed is not None:
                    return str(parsed)
                return val
        
        parsed = parse_label_value_for_classification(cleaned_annotation)
        if parsed is not None:
            return str(parsed)
        return cleaned_annotation.strip()
    
    elif task_type == "social_text_classification":
        data = extract_json_from_text(cleaned_annotation)
        if data:
            label = data.get("label", "")
            if label is not None:
                return normalize_category_for_corpus(str(label))
        
        label = extract_simple_kv(cleaned_annotation, "label")
        if label:
            return normalize_category_for_corpus(str(label))
        
        if "label" in cleaned_annotation.lower():
            match = re.search(r'[:：]\s*["\']?([^"\',\n]+)', cleaned_annotation, re.IGNORECASE)
            if match:
                return normalize_category_for_corpus(match.group(1).strip())
        
        return normalize_category_for_corpus(cleaned_annotation.strip())
    
    # ============== image_attribution_recognition ==============
    elif task_type == "image_attribution_recognition":
        data = extract_json_from_text(cleaned_annotation)
        if data:
            label = data.get("label", "")
            if label is not None:
                parsed = parse_label_value_from_text(str(label))
                if parsed is not None:
                    return str(parsed)
                return str(label).strip()
        
        label = extract_simple_kv(cleaned_annotation, "label")
        if label:
            parsed = parse_label_value_from_text(label)
            if parsed is not None:
                return str(parsed)
            return label
        
        if "label" in cleaned_annotation.lower():
            match = re.search(r'[:：]\s*["\']?([^"\',\n]+)', cleaned_annotation, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                parsed = parse_label_value_from_text(val)
                if parsed is not None:
                    return str(parsed)
                return val
        
        parsed = parse_label_value_from_text(cleaned_annotation)
        if parsed is not None:
            return str(parsed)
        return cleaned_annotation.strip()
    
    # ============== image_classification ==============
    elif task_type == "image_classification":
        fields = ["一级品类", "二级品类", "话术分析"]
        results = {}

        json_match = re.search(r"```json\s*(.*?)\s*```", cleaned_annotation, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict):
                    for field in fields:
                        results[field] = parse_image_classification_field(data.get(field), field)
                    return f"一级:{results.get('一级品类','')}|二级:{results.get('二级品类','')}|话术:{results.get('话术分析','')}"
            except json.JSONDecodeError:
                pass

        data = extract_json_from_text(cleaned_annotation)
        if data and isinstance(data, dict):
            for field in fields:
                results[field] = parse_image_classification_field(data.get(field), field)
            return f"一级:{results.get('一级品类','')}|二级:{results.get('二级品类','')}|话术:{results.get('话术分析','')}"

        for field in fields:
            val = extract_simple_kv(cleaned_annotation, field)
            if val:
                results[field] = parse_image_classification_field(val, field)

        if results:
            return f"一级:{results.get('一级品类','')}|二级:{results.get('二级品类','')}|话术:{results.get('话术分析','')}"

        for field in fields:
            pattern_list = rf'"{field}"\s*:\s*\[(.*?)\]'
            match = re.search(pattern_list, cleaned_annotation, re.DOTALL)
            if match:
                list_content = match.group(1)
                try:
                    parsed = json.loads(f"[{list_content}]")
                    results[field] = parse_image_classification_field(parsed, field)
                except:
                    results[field] = parse_image_classification_field(list_content, field)
            else:
                pattern_str = rf'"{field}"\s*:\s*"([^"]*)"'
                match = re.search(pattern_str, cleaned_annotation)
                if match:
                    results[field] = parse_image_classification_field(match.group(1), field)

        if results:
            return f"一级:{results.get('一级品类','')}|二级:{results.get('二级品类','')}|话术:{results.get('话术分析','')}"

        return cleaned_annotation.strip()
    
    else:
        data = extract_json_from_text(cleaned_annotation)
        if data and isinstance(data, dict):
            for key in ["label", "账单类目", "分类", "category", "result"]:
                if key in data:
                    return str(data[key]).strip()
            for key, val in data.items():
                if val:
                    return f"{key}:{str(val).strip()}"
        
        return cleaned_annotation.strip()


# ============================================================================
# ============================================================================

EVALUATION_FUNCTIONS = {
    "billing_scenario_classification": evaluate_billing_scenario,
    "disease_privacy_assessment": evaluate_disease_privacy,
    "social_text_classification": evaluate_corpus_classification,
    "entity_extraction": evaluate_entity_extraction,
    "image_text_relevance": evaluate_image_text_relevance,
    "image_attribution_recognition": evaluate_image_attribution_recognition,
    "image_classification": evaluate_image_classification
}