import openai
import json
import re
import sys
import os
import traceback
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from typing import Dict, List, Any, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.evaluation_functions import EVALUATION_FUNCTIONS, extract_label_from_annotation
from utils.prompts import PROMPTS_EN as PROMPTS, OUTPUT_FORMATS
from utils.prompts import build_verification_prompt
from collections import Counter
from utils.load_task import get_task_config
from utils.call_llm_api import call_llm_general
from utils.multimodal_utils import (
    extract_multimodal_data,
    format_multimodal_query,
    prompt_to_messages,
)
import threading
import random
import copy


# CLI-provided globals (populated in __main__ before run_evaluation is invoked).
TASK_TYPE = ""
OUTPUT_FILE_PREFIX = ""
MODEL = "Qwen3-VL-235B-A22B-Instruct"

OPTIMIZE_REWARD = 0.1
OPTIMIZE_PUNISH = 0.1
OPTIMIZE_BETTER = 0
OPTIMIZE_WORSE = 0

DEFAULT_VERIFICATION_PROMPT_TEMPLATE = PROMPTS["verification_with_logic_scoring"]


class Annotator:
    def __init__(self, model: str, api_key: str, base_url: str, task_type: str, category_doc_path: str, eval_func: Callable):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.eval_func = eval_func
        self.task_type = task_type
        with open(category_doc_path, 'r', encoding='utf-8') as f:
            self.doc_content = f.read()
        self.generated_prompts = []
        self.verification_prompt_template = DEFAULT_VERIFICATION_PROMPT_TEMPLATE
        self._local = threading.local()

    def get_generated_prompts(self):
        return self.generated_prompts

    def get_verification_prompt_template(self):
        return self.verification_prompt_template

    def set_generated_prompts(self, generated_prompts: List[str]):
        self.generated_prompts = generated_prompts

    def set_verification_prompt_template(self, template: str):
        self.verification_prompt_template = template
    
    def _annotate_without_context(self, query_data: Any, prompt: str) -> str:
        query_text, image_urls = extract_multimodal_data(query_data)
        query_str = query_text if query_text else json.dumps(query_data, ensure_ascii=False)

        annotation_prompt = PROMPTS["annotation_based_on_generated_prompt"].format(
            generated_prompt=prompt,
            query=query_str,
            output_format=OUTPUT_FORMATS[TASK_TYPE]
        )

        # Format multimodal prompt and build messages
        formatted_prompt = format_multimodal_query(annotation_prompt, image_urls)
        annotation_messages = prompt_to_messages(formatted_prompt, image_urls)
        annotation_result = call_llm_general(annotation_messages, self.model, temperature=0.1)
        return annotation_result

    # ==================== Hard Majority Vote ====================

    def _hard_majority_vote(self, annotation_results: List[str]) -> Tuple[str, Dict]:
        if self.task_type == "entity_extraction":
            return self._vote_entity_extraction(annotation_results)

        labels = []
        label_to_raw = {}
        empty_count = 0

        for raw_result in annotation_results:
            label = extract_label_from_annotation(raw_result, self.task_type)
            labels.append(label)
            if not label or label == "":
                empty_count += 1
            if label and label not in label_to_raw:
                label_to_raw[label] = raw_result

        vote_counter = Counter(labels)
        total_votes = len(labels)
        most_common = vote_counter.most_common()

        if not most_common or (len(most_common) == 1 and most_common[0][0] == ""):
            return annotation_results[0] if annotation_results else "", {
                "vote_distribution": {}, "winning_label": "", "winning_count": 0,
                "total_votes": total_votes, "confidence": 0.0, "margin": 0,
                "is_unanimous": False, "is_three_way_tie": False, "empty_results_count": empty_count
            }

        winning_label = most_common[0][0]
        winning_count = most_common[0][1]
        if winning_label == "" and len(most_common) > 1:
            winning_label = most_common[1][0]
            winning_count = most_common[1][1]

        is_three_way_tie = len(most_common) == 3 and winning_count == 1
        runner_up_count = 0
        if len(most_common) > 1:
            for label, count in most_common[1:]:
                if label != "":
                    runner_up_count = count
                    break

        vote_info = {
            "vote_distribution": dict(vote_counter),
            "winning_label": winning_label, "winning_count": winning_count,
            "total_votes": total_votes, "confidence": winning_count / total_votes,
            "margin": winning_count - runner_up_count,
            "is_unanimous": winning_count == total_votes,
            "is_three_way_tie": is_three_way_tie, "empty_results_count": empty_count
        }
        final_result = label_to_raw.get(winning_label, annotation_results[0] if annotation_results else "")
        return final_result, vote_info

    def _vote_entity_extraction(self, annotation_results: List[str]) -> Tuple[str, Dict]:
        total_votes = len(annotation_results)
        result_sets = []
        for result in annotation_results:
            entity_set = self._extract_entity_set(result)
            result_sets.append((result, entity_set))

        clusters = {}
        for result, entity_set in result_sets:
            key = entity_set if entity_set else frozenset()
            if key in clusters:
                rep_result, count = clusters[key]
                clusters[key] = (rep_result, count + 1)
            else:
                clusters[key] = (result, 1)

        cluster_list = sorted(
            [(es, r, c) for es, (r, c) in clusters.items()],
            key=lambda x: x[2], reverse=True
        )

        if not cluster_list:
            return annotation_results[0] if annotation_results else "", {
                "vote_distribution": {}, "winning_label": "", "winning_count": 0,
                "total_votes": total_votes, "confidence": 0.0, "margin": 0,
                "is_unanimous": False, "is_three_way_tie": False,
                "aggregation_method": "entity_set_equality"
            }

        winning_entity_set, winning_result, winning_count = cluster_list[0]
        runner_up_count = cluster_list[1][2] if len(cluster_list) > 1 else 0
        vote_dist = {}
        for entity_set, _, count in cluster_list:
            label = "|".join(sorted(entity_set)) if entity_set else ""
            vote_dist[label] = count
        winning_label = "|".join(sorted(winning_entity_set)) if winning_entity_set else ""

        return winning_result, {
            "vote_distribution": vote_dist,
            "winning_label": winning_label, "winning_count": winning_count,
            "total_votes": total_votes, "confidence": winning_count / total_votes,
            "margin": winning_count - runner_up_count,
            "is_unanimous": winning_count == total_votes,
            "is_three_way_tie": len(cluster_list) == 3 and winning_count == 1,
            "aggregation_method": "entity_set_equality",
            "num_clusters": len(cluster_list)
        }

    def _extract_entity_set(self, annotation: str) -> frozenset:
        try:
            cleaned = annotation
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[-1]
            if "```" in cleaned:
                parts = cleaned.split("```")
                for part in parts:
                    if '{' in part or '"' in part:
                        cleaned = part
                        break

            data = json.loads(cleaned)
            entities = data.get("entities", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            entity_set = set()
            for e in entities:
                if isinstance(e, dict):
                    text = str(e.get("text", "")).strip()
                    cat = str(e.get("category", "")).strip()
                    if text and cat:
                        entity_set.add(f"{text}:{cat}")
            return frozenset(entity_set)
        except:
            return frozenset()

    # ==================== Verification Agent ====================

    def _verify(self, query_str: str, annotation_results: List[str],
                annotator_names: List[str], vote_info: Dict,
                image_urls: List[str] = None) -> Tuple[str, Dict]:
        annotations_text = ""
        for i, (name, result) in enumerate(zip(annotator_names, annotation_results)):
            label = extract_label_from_annotation(result, self.task_type)
            annotations_text += f"\n=== {name} (label: {label}) ===\n{result}\n"

        is_three_way_tie = vote_info.get("is_three_way_tie", False)
        if is_three_way_tie:
            vote_text = "Vote: Three-way tie (1:1:1), no majority"
        else:
            vote_text = f"Majority vote: {vote_info.get('winning_label', 'N/A')} ({vote_info.get('winning_count', 0)}/{vote_info.get('total_votes', 0)})"

        verification_prompt = build_verification_prompt(
            verification_prompt_template=self.verification_prompt_template,
            query_str=query_str,
            annotations_text=annotations_text,
            vote_text=vote_text,
        )

        verification_messages = prompt_to_messages(verification_prompt, image_urls=image_urls)
        verification_result = call_llm_general(
            verification_messages, self.model,
            temperature=0.1
        )

        return self._parse_verification_result(
            verification_result, annotation_results, annotator_names, vote_info
        )

    def _parse_verification_result(self, verification_result: str, annotation_results: List[str],
                                   annotator_names: List[str], vote_info: Dict) -> Tuple[str, Dict]:
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', verification_result, re.DOTALL)
            json_str = json_match.group(1) if json_match else verification_result
            parsed = json.loads(json_str)
            selected_agent = parsed.get("result", ["majority_vote"])
            if isinstance(selected_agent, list):
                selected_agent = selected_agent[0]
            reason = parsed.get("reason", [])

            if selected_agent.startswith("annotator"):
                agent_idx = int(selected_agent.replace("annotator", "")) - 1
                if 0 <= agent_idx < len(annotation_results):
                    final_result = annotation_results[agent_idx]
                else:
                    final_result = self._get_majority_result(annotation_results, vote_info)
                    selected_agent = "majority_vote_fallback"
            else:
                final_result = self._get_majority_result(annotation_results, vote_info)
                selected_agent = "majority_vote"

            verification_info = {
                "selected_agent": selected_agent, "reason": reason, "was_verified": True
            }
        except Exception as e:
            print(f"  [VerificationAgent] Parse failed: {e}, fallback to majority vote")
            final_result = self._get_majority_result(annotation_results, vote_info)
            verification_info = {
                "selected_agent": "majority_vote_fallback",
                "reason": [f"Parse error: {str(e)}"], "was_verified": False
            }
        return final_result, verification_info

    def _get_majority_result(self, annotation_results: List[str], vote_info: Dict) -> str:
        winning_label = vote_info.get("winning_label", "")
        for result in annotation_results:
            label = extract_label_from_annotation(result, self.task_type)
            if label == winning_label:
                return result
        return annotation_results[0] if annotation_results else ""

    # ==================== Evaluation ====================

    def _evaluate_single_annotation(self, prediction: str, ground_truth: Any) -> Tuple[bool, int]:
        if isinstance(ground_truth, dict) and "all_annotations" in ground_truth:
            all_annotations = ground_truth.get("all_annotations", [])
            if not all_annotations:
                return False, 0
            match_count = sum(1 for ann in all_annotations if self.eval_func(prediction, ann))
            return match_count >= 2, match_count
        else:
            is_correct = self.eval_func(prediction, ground_truth)
            return is_correct, 1 if is_correct else 0

    def _evaluate_multi_annotation(self, prediction: str, ground_truth: Dict) -> Tuple[bool, int]:
        all_annotations = ground_truth.get("all_annotations", [])
        if not all_annotations:
            return False, 0
        match_count = sum(1 for ann in all_annotations if self.eval_func(prediction, ann))
        return match_count >= 2, match_count

    def evaluate_single(self, query_dict: Dict, ground_truth: Any):
        query_text, image_urls = extract_multimodal_data(query_dict)
        query_str = query_text if query_text else json.dumps(query_dict, ensure_ascii=False)

        llm_interactions = []
        self._local.interactions = llm_interactions
        annotation_results = [None] * len(self.generated_prompts)
        agent_correctness = {}

        with ThreadPoolExecutor(max_workers=len(self.generated_prompts)) as executor:
            future_to_idx = {
                executor.submit(self._annotate_without_context, query_dict, prompt): idx
                for idx, prompt in enumerate(self.generated_prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                agent_name = f"annotator{idx+1}"
                try:
                    annotation_result = future.result()
                    annotation_results[idx] = annotation_result
                    self._local.interactions.append({"step": f"annotation_{agent_name}", "prompt": "", "response": annotation_result})

                    is_annotator_correct, _ = self._evaluate_single_annotation(annotation_result, ground_truth)
                    agent_correctness[agent_name] = {"is_correct": is_annotator_correct, "annotation": annotation_result}
                    print(f"  {agent_name}: annotation success, accuracy: {is_annotator_correct}")
                except Exception as e:
                    print(f"  {agent_name}: annotation failed: {e}")
                    annotation_results[idx] = ""
                    agent_correctness[agent_name] = {"is_correct": False, "annotation": ""}

        # Step 2: Hard Majority Vote
        final_result, vote_info = self._hard_majority_vote(annotation_results)
        self._local.interactions.append({
            "step": "aggregation", "prompt": "",
            "response": f"Hard majority vote result: {vote_info['winning_label']}",
            "vote_info": vote_info
        })

        # Step 3: VerificationAgent
        verification_info = None
        was_verified = False

        if not vote_info.get("is_unanimous", False):
            annotator_names = [f"annotator{i+1}" for i in range(len(self.generated_prompts))]
            final_result, verification_info = self._verify(
                query_str=query_str,
                annotation_results=annotation_results,
                annotator_names=annotator_names,
                vote_info=vote_info,
                image_urls=image_urls if image_urls else None
            )
            was_verified = verification_info.get("was_verified", False)
            selected_agent = verification_info.get("selected_agent", "unknown")
            print(f"  [Verification] selected: {selected_agent}")

        # Step 4: evaluate
        if isinstance(ground_truth, dict) and "all_annotations" in ground_truth:
            is_correct, match_count = self._evaluate_multi_annotation(final_result, ground_truth)
        else:
            is_correct = self.eval_func(final_result, ground_truth)

        llm_interactions = self._local.interactions.copy()
        if hasattr(self._local, 'interactions'):
            del self._local.interactions

        return {
            "query": query_str,
            "annotation": final_result,
            "is_correct": is_correct,
            "llm_interactions": llm_interactions,
            "annotation_results": annotation_results,
            "agent_correctness": agent_correctness,
            "vote_info": vote_info,
            "was_verified": was_verified,
            "verification_info": verification_info,
        }
    
    def collect_error_samples_per_agent(self, query_dicts: List[Dict], ground_truths: List[Any],
                                        evaluate_results: List[Dict]) -> Dict[str, List[Dict]]:
        num_annotators = len(self.generated_prompts)
        per_agent_errors = {f"annotator{i+1}": [] for i in range(num_annotators)}
        per_agent_errors["verifier"] = []

        for idx, (query_dict, ground_truth, evaluate_result) in enumerate(zip(query_dicts, ground_truths, evaluate_results)):
            if evaluate_result.get("is_correct", False):
                continue

            query_text, image_urls = extract_multimodal_data(query_dict)
            query_str = query_text if query_text else json.dumps(query_dict, ensure_ascii=False)

            if isinstance(ground_truth, dict) and "all_annotations" in ground_truth:
                acceptable_answers = ground_truth.get("acceptable_answers", [])
                majority_answer = ground_truth.get("majority_answer", "")
                display_gt = acceptable_answers if acceptable_answers else [majority_answer]
                ground_truth_str = json.dumps(display_gt, ensure_ascii=False) if isinstance(display_gt, list) else str(display_gt)
            else:
                ground_truth_str = ground_truth if isinstance(ground_truth, str) else json.dumps(ground_truth, ensure_ascii=False)

            agent_correctness = evaluate_result.get("agent_correctness", {})
            annotation_results = evaluate_result.get("annotation_results", [])

            for prompt_idx in range(num_annotators):
                agent_name = f"annotator{prompt_idx+1}"
                agent_info = agent_correctness.get(agent_name, {})
                if not agent_info.get("is_correct", False):
                    agent_output = annotation_results[prompt_idx] if prompt_idx < len(annotation_results) else ""
                    per_agent_errors[agent_name].append({
                        "query": query_str,
                        "agent_output": agent_output,
                        "ground_truth": ground_truth_str,
                        "image_urls": image_urls,
                    })

            was_verified = evaluate_result.get("was_verified", False)
            if was_verified:
                has_correct_annotator = any(
                    info.get("is_correct", False)
                    for name, info in agent_correctness.items()
                    if name.startswith("annotator")
                )
                if has_correct_annotator:
                    verification_info = evaluate_result.get("verification_info", {})
                    vote_info = evaluate_result.get("vote_info", {})
                    per_agent_errors["verifier"].append({
                        "query": query_str,
                        "annotation_results": annotation_results,
                        "annotator_names": [f"annotator{i+1}" for i in range(num_annotators)],
                        "agent_correctness": agent_correctness,
                        "vote_info": vote_info,
                        "verifier_selection": verification_info.get("selected_agent", "unknown"),
                        "ground_truth": ground_truth_str,
                        "image_urls": image_urls,
                    })

        for agent_name, samples in per_agent_errors.items():
            if samples:
                print(f"  {agent_name}: assigned {len(samples)} error samples")

        return per_agent_errors
    
    def evaluate_batch(self, query_dicts: List[Dict], ground_truths: List[Any]):
        results = [None]*len(query_dicts)
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_index = {
                executor.submit(self.evaluate_single, query_dicts[idx], ground_truths[idx]): idx 
                for idx in range(len(query_dicts))
            }
            
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results[idx] = result
                    print(f"id {idx} annotation success")
                except Exception as e:
                    print(f"\nid {idx} failed: {e}")
                    traceback.print_exc()
                    results[idx] = {
                        "error": str(e),
                        "query": json.dumps(query_dicts[idx], ensure_ascii=False),
                        "ground_truth": ground_truths[idx],
                        "is_correct": False,
                        "llm_interactions": [],
                        "annotation": "",
                        "annotation_results": [],
                        "agent_correctness": {},
                        "vote_info": {},
                        "was_verified": False,
                        "verification_info": None,
                    }

        return results

    def optimize_prompts(self, per_agent_errors: Dict[str, List[Dict]]):
        new_generated_prompts = self.generated_prompts.copy()
        new_verification_prompt_template = self.verification_prompt_template

        optimization_tasks = []
        for prompt_idx in range(len(self.generated_prompts)):
            agent_name = f"annotator{prompt_idx+1}"
            if agent_name in per_agent_errors and per_agent_errors[agent_name]:
                optimization_tasks.append({
                    "agent_name": agent_name,
                    "prompt_num": prompt_idx,
                    "error_samples": per_agent_errors[agent_name],
                })

        if "verifier" in per_agent_errors and per_agent_errors["verifier"]:
            optimization_tasks.append({
                "agent_name": "verifier",
                "prompt_num": -1,
                "error_samples": per_agent_errors["verifier"],
            })

        if not optimization_tasks:
            print("No agent needs optimization")
            return new_generated_prompts, new_verification_prompt_template
        
        print(f"{len(optimization_tasks)} agents need optimization: {[t['agent_name'] for t in optimization_tasks]}")

        OPTIMIZE_MAX_ERROR_SAMPLES = 8

        def optimize_single_agent(task: Dict) -> Tuple[str, int, str]:
            agent_name = task["agent_name"]
            prompt_num = task["prompt_num"]
            error_samples = task["error_samples"][:OPTIMIZE_MAX_ERROR_SAMPLES]

            if prompt_num != -1:
                # Optimize annotator's prompt
                current_prompt = self.generated_prompts[prompt_num]

                text_parts = []
                all_image_urls = []
                for sid, sample in enumerate(error_samples):
                    image_urls = sample.get("image_urls", []) or []
                    image_placeholder = ""
                    if image_urls:
                        image_placeholder = "\n" + "\n".join(["<image>" for _ in image_urls])
                        all_image_urls.extend(image_urls)
                    text_parts.append(PROMPTS["spo_annotator_error_sample"].format(
                        sid=sid + 1,
                        query=sample["query"],
                        image_placeholder=image_placeholder,
                        agent_output=sample["agent_output"],
                        ground_truth=sample["ground_truth"],
                    ))
                error_text = "\n\n".join(text_parts)

                golden_text = "\n".join([
                    PROMPTS["spo_golden_text_item"].format(sid=sid + 1, ground_truth=sample["ground_truth"])
                    for sid, sample in enumerate(error_samples)
                ])

                spo_optimize_prompt = PROMPTS["spo_optimize_annotator_prompt"].format(
                    current_prompt=current_prompt,
                    error_text=error_text,
                    golden_text=golden_text,
                )
                optimization_messages = prompt_to_messages(
                    spo_optimize_prompt,
                    image_urls=all_image_urls if all_image_urls else None,
                )
                optimization_result = call_llm_general(
                    optimization_messages,
                    self.model,
                    temperature=0.1,
                    stream=True,
                )
                
                optimized_prompt = None
                prompt_match = re.search(r'<prompt>(.*?)</prompt>', optimization_result, re.DOTALL)
                if prompt_match:
                    optimized_prompt = prompt_match.group(1).strip()
                else:
                    for marker in ["Optimized prompt:", "optimized prompt:"]:
                        if marker in optimization_result:
                            optimized_prompt = optimization_result.split(marker)[-1].strip()
                            break
                    if not optimized_prompt:
                        print(f"{agent_name}: Failed to parse the optimized prompt correctly.")

                return agent_name, prompt_num, optimized_prompt
            else:
                text_parts = []
                all_image_urls = []
                for sid, sample in enumerate(error_samples):
                    ann_results = sample.get("annotation_results", [])
                    ann_names = sample.get("annotator_names", [])
                    agent_corr = sample.get("agent_correctness", {})
                    vote_info = sample.get("vote_info", {})

                    annotations_text = ""
                    for i, (name, result) in enumerate(zip(ann_names, ann_results)):
                        label = extract_label_from_annotation(result, self.task_type)
                        corr = agent_corr.get(name, {}).get("is_correct", False)
                        annotations_text += f"  {name} (label: {label}, correct: {corr}): {result}\n"

                    image_urls = sample.get("image_urls", []) or []
                    image_placeholder = ""
                    if image_urls:
                        image_placeholder = "\n" + "\n".join(["<image>" for _ in image_urls])
                        all_image_urls.extend(image_urls)
                    text_parts.append(PROMPTS["spo_verifier_error_sample"].format(
                        sid=sid + 1,
                        query=sample["query"],
                        image_placeholder=image_placeholder,
                        annotations_text=annotations_text,
                        vote_distribution=json.dumps(vote_info.get("vote_distribution", {}), ensure_ascii=False),
                        verifier_selection=sample.get("verifier_selection", "unknown"),
                        ground_truth=sample["ground_truth"],
                    ))
                error_text = "\n\n".join(text_parts)

                golden_text = "\n".join([
                    PROMPTS["spo_golden_text_item"].format(sid=sid + 1, ground_truth=sample["ground_truth"])
                    for sid, sample in enumerate(error_samples)
                ])

                spo_optimize_prompt = PROMPTS["spo_optimize_verifier_prompt"].format(
                    verification_prompt_template=self.verification_prompt_template,
                    error_text=error_text,
                    golden_text=golden_text,
                )
                optimization_messages = prompt_to_messages(
                    spo_optimize_prompt,
                    image_urls=all_image_urls if all_image_urls else None,
                )
                optimization_result = call_llm_general(
                    optimization_messages, self.model,
                    temperature=0.1, stream=True,
                )

                optimized_template = None
                prompt_match = re.search(r'<prompt>(.*?)</prompt>', optimization_result, re.DOTALL)
                if prompt_match:
                    optimized_template = prompt_match.group(1).strip()
                else:
                    for marker in ["Optimized verification prompt:", "optimized verification prompt:"]:
                        if marker in optimization_result:
                            optimized_template = optimization_result.split(marker)[-1].strip()
                            break
                    if not optimized_template:
                        print("verifier: Failed to parse the optimized verification_prompt_template")

                return agent_name, prompt_num, optimized_template
        

        with ThreadPoolExecutor(max_workers=len(optimization_tasks)) as executor:
            future_to_task = {
                executor.submit(optimize_single_agent, task): task
                for task in optimization_tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    agent_name, prompt_num, optimized_result = future.result()

                    if optimized_result:
                        if prompt_num != -1:
                            new_generated_prompts[prompt_num] = optimized_result
                            print(f"{agent_name}: prompt optimization complete (based on {len(task['error_samples'])} error samples)")
                        else:
                            new_verification_prompt_template = optimized_result
                            print(f"{agent_name}: verification_prompt_template optimization complete (based on {len(task['error_samples'])} error samples)")
                    else:
                        print(f"{agent_name}: optimization failed, unable to extract optimization result")
                except Exception as e:
                    agent_name = task["agent_name"]
                    print(f"{agent_name}: optimization failed: {e}")
                    traceback.print_exc()

        return new_generated_prompts, new_verification_prompt_template
    
    def evaluate_on_full_dataset(self, all_queries: List[Dict], all_ground_truths: List[Any], batch_size: int = 50) -> Tuple[int, int, float]:
        print(f"\n========== Starting evaluation of current prompts on full training set ({len(all_queries)} samples) ==========")
        total_correct = 0
        total_count = len(all_queries)
        
        for i in range(0, total_count, batch_size):
            batch_queries = all_queries[i:i+batch_size]
            batch_ground_truths = all_ground_truths[i:i+batch_size]
            batch_results = self.evaluate_batch(batch_queries, batch_ground_truths)
            
            for result in batch_results:
                if result.get("is_correct", False):
                    total_correct += 1
            
            print(f"  Evaluated {min(i+batch_size, total_count)}/{total_count} samples, current accuracy: {total_correct/min(i+batch_size, total_count):.4f}")
        
        accuracy = total_correct / total_count if total_count > 0 else 0
        print(f"========== Training set evaluation complete: {total_correct}/{total_count} = {accuracy:.4f} ({accuracy*100:.2f}%) ==========\n")
        return total_correct, total_count, accuracy


# ============================================================================

def run_evaluation():
    global OPTIMIZE_REWARD, OPTIMIZE_PUNISH, OPTIMIZE_BETTER, OPTIMIZE_WORSE, TASK_TYPE

    start_id = 100
    end_id = 300
    num_epochs = 6
    num_rounds = 3

    task_config = get_task_config(TASK_TYPE, end_id)
    TASK_TYPE = TASK_TYPE.replace("multi_","")
    evaluate_func = EVALUATION_FUNCTIONS.get(TASK_TYPE)

    all_queries = task_config["queries"][start_id:end_id]
    all_ground_truths = task_config["answers"][start_id:end_id]
    
    queries = list(all_queries)
    ground_truths = list(all_ground_truths)
    
    os.makedirs(os.path.dirname(f"output/{TASK_TYPE}"), exist_ok=True)
    
    model = MODEL

    evaluator = Annotator(
        model=model,
        api_key="afdca593-9da5-4570-9a26-3df4962c7b4f",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        task_type=TASK_TYPE,
        category_doc_path=task_config["requirement_doc_path"],
        eval_func=evaluate_func
    )
    
    doc_content = evaluator.doc_content
    generated_prompts = [doc_content, doc_content, doc_content]
    evaluator.set_generated_prompts(generated_prompts)
    
    print("Generated prompts set successfully")
    init_generated_prompts = copy.copy(evaluator.get_generated_prompts())
    init_verification_prompt_template = evaluator.get_verification_prompt_template()

    print("\n>>> Before optimization starts, evaluating initial prompts accuracy on training set:")
    init_correct, init_total, init_accuracy = evaluator.evaluate_on_full_dataset(
        all_queries, all_ground_truths, batch_size=50
    )
    print(f">>> Initial accuracy: {init_correct}/{init_total} = {init_accuracy:.4f} ({init_accuracy*100:.2f}%)\n")
    
    prev_after_correct, prev_after_total, prev_after_accuracy = init_correct, init_total, init_accuracy
    
    epoch_accuracies = []
    
    for epoch in range(num_epochs):
        output_file = f"output/{TASK_TYPE}/{TASK_TYPE}_multi_agents_prompt_optimization_with_{OUTPUT_FILE_PREFIX}_epoch{epoch}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs} starting")
        print(f"{'='*60}")
        
        before_correct, before_total, before_accuracy = prev_after_correct, prev_after_total, prev_after_accuracy
        print(f">>> Epoch {epoch+1} accuracy before optimization (reused): {before_correct}/{before_total} = {before_accuracy:.4f} ({before_accuracy*100:.2f}%)")
        
        combined = list(zip(queries, ground_truths))
        random.shuffle(combined)
        queries[:], ground_truths[:] = zip(*combined)
        
        results = []
        missing_or_error_indices = list(range(len(queries)))
        results = [None]*len(missing_or_error_indices)
        batch_size = 20
        
        for i in range(0, len(missing_or_error_indices), batch_size):
            batch_idxs = missing_or_error_indices[i:i+batch_size]
            batch_queries = [queries[idx] for idx in batch_idxs]
            batch_ground_truths = [ground_truths[idx] for idx in batch_idxs]
            print(f"---------------------- Starting optimization round {i//batch_size} -----------------------")
            
            try:
                evaluate_results = evaluator.evaluate_batch(batch_queries, batch_ground_truths)
                ori_num_correct = 0
                for evaluate_result in evaluate_results:
                    if evaluate_result["is_correct"]:
                        ori_num_correct += 1
                ori_generated_prompts = copy.copy(evaluator.get_generated_prompts())
                ori_verification_prompt_template = evaluator.get_verification_prompt_template()

                if ori_num_correct < len(batch_queries):
                    per_agent_errors = evaluator.collect_error_samples_per_agent(batch_queries, batch_ground_truths, evaluate_results)
                    total_errors = len(batch_queries) - ori_num_correct
                    print(f"System has {total_errors} error samples, starting SPO optimization")

                    for round_id in range(num_rounds):
                        print(f"Round {round_id} of this optimization starting")
                        new_generated_prompts, new_verification_prompt_template = evaluator.optimize_prompts(per_agent_errors)
                        evaluator.set_generated_prompts(new_generated_prompts)
                        evaluator.set_verification_prompt_template(new_verification_prompt_template)
                        eval_results_for_new_prompts = evaluator.evaluate_batch(batch_queries, batch_ground_truths)
                        new_num_correct = 0
                        for eval_result_for_new_prompts in eval_results_for_new_prompts:
                            if eval_result_for_new_prompts["is_correct"]:
                                new_num_correct += 1

                        if ori_num_correct > new_num_correct:
                            OPTIMIZE_WORSE += 1
                            print("Newly generated prompt performed worse, reverting to original prompt")
                            evaluator.set_generated_prompts(ori_generated_prompts)
                            evaluator.set_verification_prompt_template(ori_verification_prompt_template)
                        else:
                            if ori_num_correct < new_num_correct:
                                OPTIMIZE_BETTER += 1
                                print("Newly generated prompt is better than the original prompt")
                                evaluate_results = eval_results_for_new_prompts
                                ori_generated_prompts = copy.copy(new_generated_prompts)
                                ori_verification_prompt_template = new_verification_prompt_template
                                ori_num_correct = new_num_correct

                            if new_num_correct >= len(batch_queries):
                                print("This batch is all answered correctly, can be skipped")
                                break
                            if round_id >= num_rounds - 1:
                                continue
                            else:
                                per_agent_errors = evaluator.collect_error_samples_per_agent(batch_queries, batch_ground_truths, eval_results_for_new_prompts)
                else:
                    print("This batch samples all correct, no need for prompt optimization")

            except Exception as e:
                print(f"\nOptimization round {i//batch_size} failed: {e}")
                traceback.print_exc()
                for idx in batch_idxs:
                    results[idx] = {
                        "error": str(e), 
                        "query": json.dumps(queries[idx], ensure_ascii=False),
                        "ground_truth": ground_truths[idx],
                        "is_correct": False
                    }
                continue
            
            for batch_id, idx in enumerate(batch_idxs):
                results[idx] = {}
                results[idx]["annotation"] = evaluate_results[batch_id].get("annotation", "")
                results[idx]["is_correct"] = evaluate_results[batch_id]["is_correct"]
                results[idx]["query"] = json.dumps(queries[idx], ensure_ascii=False)
                results[idx]["ground_truth"] = ground_truths[idx]
                results[idx]["llm_interactions"] = evaluate_results[batch_id]["llm_interactions"]
                results[idx]["generated_prompts"] = ori_generated_prompts
                results[idx]["verification_prompt_template"] = ori_verification_prompt_template

        if OPTIMIZE_BETTER + OPTIMIZE_WORSE > 0:
            OPTIMIZE_REWARD = OPTIMIZE_WORSE / (OPTIMIZE_WORSE + OPTIMIZE_BETTER)
            OPTIMIZE_PUNISH = OPTIMIZE_BETTER / (OPTIMIZE_WORSE + OPTIMIZE_BETTER)
        
        batch_correct = 0
        for result in results:
            if result and result.get("is_correct", False):
                batch_correct += 1
        batch_accuracy = batch_correct / len(results) if results else 0
        print(f"\n>>> Epoch {epoch+1} batch optimization complete, batch accuracy: {batch_correct}/{len(results)} = {batch_accuracy:.4f} ({batch_accuracy*100:.2f}%)")

        print(f">>> Epoch {epoch+1} after optimization, testing current prompts accuracy on training set:")
        after_correct, after_total, after_accuracy = evaluator.evaluate_on_full_dataset(
            all_queries, all_ground_truths, batch_size=50
        )
        
        prev_after_correct, prev_after_total, prev_after_accuracy = after_correct, after_total, after_accuracy
        
        accuracy_change = after_accuracy - before_accuracy
        change_symbol = "↑" if accuracy_change > 0 else ("↓" if accuracy_change < 0 else "-")
        
        print(f">>> Epoch {epoch+1} accuracy after optimization: {after_correct}/{after_total} = {after_accuracy:.4f} ({after_accuracy*100:.2f}%)")
        print(f">>> Accuracy change: {change_symbol} {abs(accuracy_change):.4f} ({abs(accuracy_change)*100:.2f}%)")
        print(f">>> Current optimization successes: {OPTIMIZE_BETTER}, failures: {OPTIMIZE_WORSE}")
        
        epoch_accuracies.append({
            "epoch": epoch + 1,
            "before": {
                "correct": before_correct,
                "total": before_total,
                "accuracy": before_accuracy,
                "accuracy_percent": f"{before_accuracy*100:.2f}%"
            },
            "after": {
                "correct": after_correct,
                "total": after_total,
                "accuracy": after_accuracy,
                "accuracy_percent": f"{after_accuracy*100:.2f}%"
            },
            "change": {
                "absolute": accuracy_change,
                "percent": f"{accuracy_change*100:.2f}%",
                "symbol": change_symbol
            }
        })
        
        output_data = {
            "batch_results": results,
            "full_dataset_evaluation": {
                "before_optimization": {
                    "correct": before_correct,
                    "total": before_total,
                    "accuracy": before_accuracy,
                    "accuracy_percent": f"{before_accuracy*100:.2f}%"
                },
                "after_optimization": {
                    "correct": after_correct,
                    "total": after_total,
                    "accuracy": after_accuracy,
                    "accuracy_percent": f"{after_accuracy*100:.2f}%"
                },
                "change": {
                    "absolute": accuracy_change,
                    "percent": f"{accuracy_change*100:.2f}%",
                    "symbol": change_symbol
                }
            },
            "epoch_accuracies_so_far": epoch_accuracies,
            "init_generated_prompts": init_generated_prompts,
            "init_verification_prompt_template": init_verification_prompt_template,
            "current_generated_prompts": evaluator.get_generated_prompts(),
            "current_verification_prompt_template": evaluator.get_verification_prompt_template(),
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\n🎉 Epoch {epoch+1} complete! Results saved to: {output_file}")
        except Exception as e:
            print(f"\nSave failed: {e}")
    
    print("\n" + "="*70)
    print("All epochs accuracy summary on full training set:")
    print("="*70)
    print(f"  {'Epoch':<8} {'Before':<18} {'After':<18} {'Change':<12}")
    print("-"*70)
    for acc_info in epoch_accuracies:
        before_str = f"{acc_info['before']['accuracy']:.4f} ({acc_info['before']['accuracy_percent']})"
        after_str = f"{acc_info['after']['accuracy']:.4f} ({acc_info['after']['accuracy_percent']})"
        change_str = f"{acc_info['change']['symbol']} {abs(acc_info['change']['absolute']):.4f} ({acc_info['change']['percent']})"
        print(f"  {acc_info['epoch']:<8} {before_str:<18} {after_str:<18} {change_str:<12}")
    print("="*70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='MAS-SPO multi-agent prompt optimization')
    parser.add_argument('--task_type', type=str, required=True, help='Task type')
    parser.add_argument('--output_file_prefix', type=str, required=True, help='Output file prefix')
    parser.add_argument('--model', type=str, default='Qwen3-VL-235B-A22B-Instruct',
                        help='LLM model name (default: Qwen3-VL-235B-A22B-Instruct)')
    args = parser.parse_args()

    TASK_TYPE = args.task_type
    OUTPUT_FILE_PREFIX = args.output_file_prefix
    MODEL = args.model

    run_evaluation()
