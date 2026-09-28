"""
Unified single-agent annotation evaluation script.

Supports three modes via --mode:
  - cot (default): chain-of-thought prompting with optional best-epoch optimized doc
  - direct: no CoT, direct answer only
  - icl: in-context-learning, retrieves similar examples from a separate ICL pool

CoT and direct share the same SingleAnnotator code path and differ only in the
PROMPTS template used. ICL mode additionally enables example retrieval and the
ICLExamplePool. The image-similarity stack and pre-computed embeddings live in
`utils.multimodal_utils` and are loaded once per process at import time.
"""
import openai
import json
import re
import glob
import numpy as np
import sys
import os
import traceback
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.evaluation_functions import EVALUATION_FUNCTIONS
from utils.prompts import PROMPTS_EN as PROMPTS, OUTPUT_FORMATS
from utils.load_task import get_task_config
from utils.call_llm_api import call_llm_general
from utils.multimodal_utils import (
    extract_multimodal_data,
    format_multimodal_query,
    compute_multimodal_similarity,
    prompt_to_messages,
)


# ============================================================================
# ICL example pool
# ============================================================================

class ICLExamplePool:
    """ICL example pool: manages the example library and supports similarity retrieval."""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.examples: List[Dict] = []

    def initialize_from_dataset(self, queries: List[Dict], ground_truths: List[Any]):
        self.examples = [{"query": q, "ground_truth": gt} for q, gt in zip(queries, ground_truths)]
        print(f"[ICL Pool] Initialization complete, total {len(self.examples)} examples")

    def retrieve_similar_examples(self, query_data: Dict) -> List[Tuple[Dict, float]]:
        if not self.examples:
            return []
        scored = []
        for example in self.examples:
            similarity = compute_multimodal_similarity(query_data, example["query"])
            if similarity < 0.99:
                scored.append((example, similarity))
        if not scored:
            return []
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.top_k]

    def format_examples_context(self, examples_with_scores: List[Tuple[Dict, float]]) -> Tuple[str, List[str]]:
        if not examples_with_scores:
            return "", []
        context_parts = []
        all_sample_image_urls: List[str] = []
        for i, (example, similarity) in enumerate(examples_with_scores, 1):
            query_text, image_urls = extract_multimodal_data(example["query"])
            gt = example["ground_truth"]
            if isinstance(gt, dict) and "vote_distribution_detail" in gt:
                vote_dist = gt.get("vote_distribution_detail", [])
                if vote_dist:
                    gt_display = vote_dist[0]["answer"]
                else:
                    all_annotations = gt.get("all_annotations", [])
                    gt_display = all_annotations[0] if all_annotations else ""
            else:
                gt_display = gt
            query_with_images = format_multimodal_query(query_text, image_urls)
            all_sample_image_urls.extend(image_urls or [])
            context_parts.append(PROMPTS["icl_example_section"].format(
                idx=i,
                similarity=similarity,
                query_with_images=query_with_images,
                ground_truth=gt_display,
            ))
        return "\n\n".join(context_parts), all_sample_image_urls


# ============================================================================
# Unified single-agent annotator
# ============================================================================

MODE_PROMPT_KEYS = {
    "cot": "annotation_based_on_generated_prompt",
    "direct": "annotation_based_on_generated_prompt_direct",
    "icl": "annotation_based_on_generated_prompt",  # fallback when no ICL examples retrieved
}


class SingleAnnotator:
    """Single-agent annotator. Behaviour is controlled by `mode`:
      - cot: CoT-style annotation prompt
      - direct: answer-only annotation prompt
      - icl: ICL retrieval + ICL annotation prompt (falls back to cot template if no examples)
    """

    def __init__(self, model: str, api_key: str, base_url: str, category_doc_path: str,
                 eval_func: Callable, task_type: str, mode: str = "cot",
                 icl_top_k: int = 3, temperature: float = 0.1):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.eval_func = eval_func
        self.task_type = task_type
        self.mode = mode
        self.temperature = temperature
        with open(category_doc_path, 'r', encoding='utf-8') as f:
            self.doc_content = f.read()
        self.optimized_doc = ""
        self._local = threading.local()
        self.icl_pool: Optional[ICLExamplePool] = ICLExamplePool(top_k=icl_top_k) if mode == "icl" else None

    def set_optimized_doc(self, optimized_doc: str):
        self.optimized_doc = optimized_doc

    def initialize_icl_pool(self, queries: List[Dict], ground_truths: List[Any]):
        if self.icl_pool is None:
            raise RuntimeError("ICL pool is only available in mode='icl'")
        self.icl_pool.initialize_from_dataset(queries, ground_truths)

    @staticmethod
    def _extract_representative_gt(ground_truth: Any) -> Any:
        if isinstance(ground_truth, dict) and "vote_distribution_detail" in ground_truth:
            vote_dist = ground_truth.get("vote_distribution_detail", [])
            if vote_dist:
                return vote_dist[0]["answer"]
            all_annotations = ground_truth.get("all_annotations", [])
            if all_annotations:
                return all_annotations[0]
        return ground_truth

    def _evaluate_multi_annotation_detailed(self, prediction: str, ground_truth: Dict) -> Dict[str, Any]:
        all_annotations = ground_truth.get("all_annotations", [])
        vote_distribution_detail = ground_truth.get("vote_distribution_detail", [])

        if not all_annotations:
            return {
                "match_count": 0,
                "total_annotators": 0,
                "hit_rate": 0.0,
                "is_correct_hit2": False,
                "is_correct_hit1": False,
                "is_correct_majority": False,
                "agreement_level": "unknown"
            }

        total_annotators = len(all_annotations)
        match_count = sum(1 for ann in all_annotations if self.eval_func(prediction, ann))
        hit_rate = match_count / total_annotators if total_annotators > 0 else 0.0
        agreement_level = ground_truth.get("agreement_level", "unknown")

        is_correct_majority = False
        if vote_distribution_detail:
            top_count = vote_distribution_detail[0].get("count", 0)
            tied_answers = [v["answer"] for v in vote_distribution_detail if v.get("count", 0) == top_count]
            for top_answer in tied_answers:
                if self.eval_func(prediction, top_answer):
                    is_correct_majority = True
                    break

        return {
            "match_count": match_count,
            "total_annotators": total_annotators,
            "hit_rate": hit_rate,
            "is_correct_hit2": match_count >= 2,
            "is_correct_hit1": match_count >= 1,
            "is_correct_majority": is_correct_majority,
            "agreement_level": agreement_level,
        }

    def _build_annotation_prompt(self, doc: str, query_str: str, output_format: str,
                                  icl_context: str) -> Tuple[str, str]:
        """Return (annotation_prompt, step_name)."""
        if self.mode == "icl" and icl_context:
            prompt = PROMPTS["icl_annotation_prompt"].format(
                doc=doc, icl_context=icl_context, query=query_str, output_format=output_format,
            )
            return prompt, "annotation_with_icl"

        key = MODE_PROMPT_KEYS[self.mode]
        prompt = PROMPTS[key].format(
            generated_prompt=doc, query=query_str, output_format=output_format,
        )
        if self.mode == "icl":
            return prompt, "annotation_without_icl"
        step_name = "annotation_based_on_optimized_doc" if self.optimized_doc else "annotation_based_on_doc"
        return prompt, step_name

    def evaluate_single(self, query_dict: Dict, ground_truth: Any) -> Dict[str, Any]:
        query_text, image_urls = extract_multimodal_data(query_dict)
        query_str = format_multimodal_query(query_text, image_urls)
        llm_interactions = []
        self._local.interactions = llm_interactions

        icl_context = ""
        sample_image_urls: List[str] = []
        similar_examples: List[Tuple[Dict, float]] = []
        if self.mode == "icl" and self.icl_pool is not None:
            similar_examples = self.icl_pool.retrieve_similar_examples(query_dict)
            icl_context, sample_image_urls = self.icl_pool.format_examples_context(similar_examples)
            if icl_context:
                print(f"  [ICL] Retrieved {len(similar_examples)} similar examples")
                for i, (ex, sim) in enumerate(similar_examples):
                    ex_text, _ = extract_multimodal_data(ex["query"])
                    print(f"    - Example {i+1}: sim={sim:.3f}, query={ex_text[:50]}...")
            else:
                print(f"  [ICL] No similar examples retrieved")

        doc = self.optimized_doc if self.optimized_doc else self.doc_content
        output_format = OUTPUT_FORMATS.get(self.task_type, '```json\n{"label": "annotation result"}```')

        annotation_prompt, step_name = self._build_annotation_prompt(
            doc, query_str, output_format, icl_context
        )
        all_image_urls = sample_image_urls + (image_urls or [])

        annotation_messages = prompt_to_messages(
            annotation_prompt, image_urls=all_image_urls if all_image_urls else None
        )
        annotation_result = call_llm_general(
            annotation_messages, self.model, temperature=self.temperature
        )

        interaction_entry = {
            "step": step_name, "prompt": annotation_prompt, "response": annotation_result,
        }
        if self.mode == "icl":
            interaction_entry["icl_injected"] = bool(icl_context)
        self._local.interactions.append(interaction_entry)

        multi_annotation_metrics = None
        if isinstance(ground_truth, dict) and "all_annotations" in ground_truth:
            multi_annotation_metrics = self._evaluate_multi_annotation_detailed(annotation_result, ground_truth)
            is_correct = multi_annotation_metrics["is_correct_hit2"]
        else:
            is_correct = self.eval_func(annotation_result, ground_truth)

        llm_interactions = self._local.interactions.copy()
        del self._local.interactions

        result: Dict[str, Any] = {
            "annotation": annotation_result,
            "is_correct": is_correct,
            "llm_interactions": llm_interactions,
        }
        if multi_annotation_metrics:
            result["multi_annotation_metrics"] = multi_annotation_metrics
        if self.mode == "icl":
            result["context_injected"] = bool(icl_context)
            result["icl_examples"] = [
                {"query": extract_multimodal_data(ex["query"])[0], "similarity": sim}
                for ex, sim in similar_examples
            ]
        return result


# ============================================================================
# Best-epoch lookup
# ============================================================================

def find_best_epoch(task_type: str, output_file_prefix: str, mode: str
                    ) -> Tuple[Optional[int], Optional[Dict], float]:
    """Search optimization output files for the epoch with the best training accuracy.

    Search filename pattern depends on mode (preserved from the original scripts).
    Returns (None, None, 0.0) if no epoch beats the baseline.
    """
    if mode == "cot":
        search_pattern = (
            f"output/{task_type}/{task_type}_single_agent_doc_multimodal_optimization_with_"
            f"{output_file_prefix}_epoch*.json"
        )
    else:
        search_pattern = (
            f"output/{task_type}/{task_type}_single_agent_doc_optimization_with_"
            f"{output_file_prefix}_epoch*.json"
        )

    print(f"\nSearching for best epoch among all results...")
    epoch_files = glob.glob(search_pattern)
    if not epoch_files:
        print(f"No epoch result files found: {search_pattern}")
        return None, None, 0.0

    best_epoch = -1
    best_accuracy = 0.0
    best_data: Optional[Dict] = None
    baseline_accuracy: Optional[float] = None
    epoch_results = []

    for file_path in sorted(epoch_files):
        try:
            filename = os.path.basename(file_path)
            match = re.search(r'_epoch(\d+)\.json$', filename)
            if not match:
                continue
            epoch = int(match.group(1))
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            accuracy = 0.0
            if "full_dataset_evaluation" in data:
                eval_info = data["full_dataset_evaluation"]
                if "after_optimization" in eval_info:
                    accuracy = eval_info["after_optimization"].get("accuracy", 0.0)
                elif "accuracy" in eval_info:
                    accuracy = eval_info["accuracy"]
                if epoch == 0 and "before_optimization" in eval_info:
                    baseline_accuracy = eval_info["before_optimization"].get("accuracy", 0.0)

            epoch_results.append({"epoch": epoch, "file": file_path, "accuracy": accuracy})
            print(f"  Epoch {epoch}: training accuracy = {accuracy:.4f} ({accuracy*100:.2f}%)")
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                best_data = data
        except Exception as e:
            print(f"  Failed to read file {file_path}: {e}")
            continue

    if best_epoch < 0:
        print("No valid epoch results found")
        return None, None, 0.0

    if baseline_accuracy is not None:
        print(f"\nBaseline accuracy (epoch0 before_optimization): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
        if best_accuracy < baseline_accuracy:
            print(f"\n  All epochs underperform the baseline; optimization brought no improvement")
            print(f"   Best epoch accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
            print(f"   Baseline accuracy:   {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
            return None, None, 0.0

    print(f"\nBest Epoch: {best_epoch}, Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print("-" * 50)
    for info in sorted(epoch_results, key=lambda x: x["epoch"]):
        marker = " <-- best" if info["epoch"] == best_epoch else ""
        print(f"  Epoch {info['epoch']:2d}: {info['accuracy']:.4f} ({info['accuracy']*100:.2f}%){marker}")
    print("-" * 50)
    return best_epoch, best_data, best_accuracy


# ============================================================================
# Output filename helpers
# ============================================================================

def _build_output_filename(mode: str, kind: str, task_type: str,
                           output_file_prefix: Optional[str],
                           epoch_tag: Optional[str], icl_top_k: int) -> str:
    """Build the output JSON filename.

    kind: "best_epoch" | "specific_epoch" | "baseline"
    """
    base = f"output/{task_type}/{task_type}_evaluation_single_agent"
    mode_suffix = {"cot": "", "direct": "_direct", "icl": f"_icl"}[mode]
    icl_suffix = f"_k{icl_top_k}" if mode == "icl" else ""

    if kind == "baseline":
        if mode == "direct":
            return f"{base}_doc_direct_annotation_baseline.json"
        if mode == "icl":
            return f"{base}_icl_baseline_k{icl_top_k}.json"
        return f"{base}_doc_baseline.json"

    return f"{base}{mode_suffix}_{output_file_prefix}_{epoch_tag}{icl_suffix}.json"


# ============================================================================
# Main evaluation loop
# ============================================================================

def run_evaluation():
    start_time = time.time()

    if MODE == "icl":
        # ICL pool from [100, 300), test set from [300, 1300)
        icl_start_id, icl_end_id = 100, 300
        test_start_id, test_end_id = 300, 1300
    else:
        # cot / direct: test set only, no ICL pool
        icl_start_id = icl_end_id = 0
        test_start_id, test_end_id = 300, 1300

    max_workers = 5
    task_config = get_task_config(TASK_TYPE, test_end_id)
    evaluate_func = EVALUATION_FUNCTIONS.get(TASK_TYPE)

    evaluator = SingleAnnotator(
        model=MODEL,
        api_key="afdca593-9da5-4570-9a26-3df4962c7b4f",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        category_doc_path=task_config["requirement_doc_path"],
        eval_func=evaluate_func,
        task_type=TASK_TYPE,
        mode=MODE,
        icl_top_k=ICL_TOP_K,
        temperature=ANNOTATOR_TEMPERATURE,
    )

    if MODE == "icl":
        icl_queries = list(task_config["queries"][icl_start_id:icl_end_id])
        icl_ground_truths = list(task_config["answers"][icl_start_id:icl_end_id])
        evaluator.initialize_icl_pool(icl_queries, icl_ground_truths)

    queries = list(task_config["queries"][test_start_id:test_end_id])
    ground_truths = list(task_config["answers"][test_start_id:test_end_id])

    optimized_doc = ""
    best_epoch: Optional[int] = -1
    best_accuracy = 0.0
    mode_label = {"cot": "Single Agent", "direct": "Single Agent Direct", "icl": "Single Agent ICL"}[MODE]

    if USE_BEST_EPOCH_PROMPT and OUTPUT_FILE_PREFIX:
        best_epoch, best_data, best_accuracy = find_best_epoch(TASK_TYPE, OUTPUT_FILE_PREFIX, MODE)
        if best_data is not None:
            optimized_doc = best_data.get("current_doc", "")
            if optimized_doc:
                evaluator.set_optimized_doc(optimized_doc)
                print(f"\nUsing optimized rule doc from Epoch {best_epoch} (training accuracy: {best_accuracy:.4f})")
                print(f"============ {mode_label}: annotating with best-epoch optimized rule doc ===============")
                print(optimized_doc[:200] + "..." if len(optimized_doc) > 200 else optimized_doc)
                output_file = _build_output_filename(
                    MODE, "best_epoch", TASK_TYPE, OUTPUT_FILE_PREFIX,
                    f"best_epoch{best_epoch}_optimized_doc" if MODE != "icl" else f"best_epoch{best_epoch}",
                    ICL_TOP_K,
                )
            else:
                print("Warning: current_doc field not found in best-epoch data, falling back to original rule doc")
                print(f"============ {mode_label}: annotating with original rule doc ===============")
                output_file = _build_output_filename(MODE, "baseline", TASK_TYPE, None, None, ICL_TOP_K)
        else:
            # cot keeps the historical "exit" behaviour when no epoch beats baseline.
            # direct/icl fall back silently to baseline (matching their original behaviour).
            if MODE == "cot":
                print("\nAll epochs underperform the baseline, no need to retest. Exiting.")
                sys.exit(0)
            print(f"No usable best epoch, falling back to baseline")
            print(f"============ {mode_label}: annotating with original rule doc ===============")
            output_file = _build_output_filename(MODE, "baseline", TASK_TYPE, None, None, ICL_TOP_K)

    elif USE_OPTIMIZED_PROMPT and OUTPUT_FILE_PREFIX and TEST_EPOCH is not None:
        if MODE == "cot":
            optimized_doc_path = (
                f"output/{TASK_TYPE}/{TASK_TYPE}_single_agent_doc_multimodal_optimization_with_"
                f"{OUTPUT_FILE_PREFIX}_epoch{TEST_EPOCH}.json"
            )
        else:
            optimized_doc_path = (
                f"output/{TASK_TYPE}/{TASK_TYPE}_single_agent_doc_optimization_with_"
                f"{OUTPUT_FILE_PREFIX}_epoch{TEST_EPOCH}.json"
            )

        if os.path.exists(optimized_doc_path):
            with open(optimized_doc_path, "r", encoding='utf-8') as f:
                optimization_records = json.load(f)
            print(f"Read Optimized Doc From {optimized_doc_path}")
            optimized_doc = optimization_records.get("current_doc", "")
            if optimized_doc:
                evaluator.set_optimized_doc(optimized_doc)
                print(f"============ {mode_label}: annotating with optimized rule doc ===============")
                print(optimized_doc[:200] + "..." if len(optimized_doc) > 200 else optimized_doc)
                output_file = _build_output_filename(
                    MODE, "specific_epoch", TASK_TYPE, OUTPUT_FILE_PREFIX,
                    f"epoch{TEST_EPOCH}_optimized_doc" if MODE != "icl" else f"epoch{TEST_EPOCH}",
                    ICL_TOP_K,
                )
            else:
                print("Warning: current_doc field not found in file, falling back to original rule doc")
                print(f"============ {mode_label}: annotating with original rule doc ===============")
                output_file = _build_output_filename(MODE, "baseline", TASK_TYPE, None, None, ICL_TOP_K)
        else:
            print(f"Warning: optimized result file {optimized_doc_path} not found, falling back to original rule doc")
            print(f"============ {mode_label}: annotating with original rule doc ===============")
            output_file = _build_output_filename(MODE, "baseline", TASK_TYPE, None, None, ICL_TOP_K)

    else:
        print(f"============ {mode_label}: annotating with original rule doc ===============")
        output_file = _build_output_filename(MODE, "baseline", TASK_TYPE, None, None, ICL_TOP_K)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    all_run_accuracies: List[float] = []
    all_run_results: List[Dict[str, Any]] = []

    for run_idx in range(NUM_RUNS):
        print(f"\n{'='*50}")
        print(f"  Run {run_idx + 1}/{NUM_RUNS}")
        print(f"{'='*50}")

        results: List[Optional[Dict[str, Any]]] = [None] * len(queries)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(evaluator.evaluate_single, queries[idx], ground_truths[idx]): idx
                for idx in range(len(queries))
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results[idx] = result
                    results[idx]['ground_truth'] = ground_truths[idx]
                    results[idx]['query'] = json.dumps(queries[idx], ensure_ascii=False)
                    print(f"[Run {run_idx+1}] Sample {idx} annotation completed")
                except Exception as e:
                    print(f"\n[Run {run_idx+1}] Index {idx} failed: {e}")
                    traceback.print_exc()
                    results[idx] = {
                        "error": str(e),
                        "query": json.dumps(queries[idx], ensure_ascii=False),
                        "ground_truth": ground_truths[idx],
                        "is_correct": False,
                    }

        num_correct = 0
        multi_annotation_stats = {
            "total": 0,
            "by_agreement_level": {"consensus": 0, "split": 0, "ambiguous": 0},
            "hit2_correct": {"consensus": 0, "split": 0, "ambiguous": 0},
            "hit1_correct": {"consensus": 0, "split": 0, "ambiguous": 0},
            "majority_correct": {"consensus": 0, "split": 0, "ambiguous": 0},
        }

        for result in results:
            if result and result.get("is_correct", False):
                num_correct += 1
            if result and "multi_annotation_metrics" in result:
                metrics = result["multi_annotation_metrics"]
                level = metrics.get("agreement_level", "ambiguous")
                multi_annotation_stats["total"] += 1
                multi_annotation_stats["by_agreement_level"][level] += 1
                if metrics["is_correct_hit2"]:
                    multi_annotation_stats["hit2_correct"][level] += 1
                if metrics["is_correct_hit1"]:
                    multi_annotation_stats["hit1_correct"][level] += 1
                if metrics["is_correct_majority"]:
                    multi_annotation_stats["majority_correct"][level] += 1

        run_accuracy = num_correct / len(results) if results else 0
        all_run_accuracies.append(run_accuracy)
        all_run_results.append({"results": results, "multi_annotation_stats": multi_annotation_stats})

        if multi_annotation_stats["total"] > 0:
            print(f"\n[Run {run_idx+1}] ========== Multi-Annotation Detailed Statistics ==========")
            print(f"Total multi-annotation samples: {multi_annotation_stats['total']}")
            print(f"\nBy agreement level:")
            for level in ["consensus", "split", "ambiguous"]:
                count = multi_annotation_stats["by_agreement_level"][level]
                if count > 0:
                    hit2_acc = multi_annotation_stats["hit2_correct"][level] / count * 100
                    hit1_acc = multi_annotation_stats["hit1_correct"][level] / count * 100
                    maj_acc = multi_annotation_stats["majority_correct"][level] / count * 100
                    print(f"  {level}: {count} samples")
                    print(f"    - Hit@2 (>=2 annotators): {hit2_acc:.2f}%")
                    print(f"    - Hit@1 (>=1 annotator):  {hit1_acc:.2f}%")
                    print(f"    - Majority Vote:          {maj_acc:.2f}%")
            total = multi_annotation_stats["total"]
            total_hit2 = sum(multi_annotation_stats["hit2_correct"].values())
            total_hit1 = sum(multi_annotation_stats["hit1_correct"].values())
            total_maj = sum(multi_annotation_stats["majority_correct"].values())
            print(f"\nOverall accuracy:")
            print(f"  - Hit@2 (>=2 annotators): {total_hit2/total*100:.2f}% ({total_hit2}/{total})")
            print(f"  - Hit@1 (>=1 annotator):  {total_hit1/total*100:.2f}% ({total_hit1}/{total})")
            print(f"  - Majority Vote:          {total_maj/total*100:.2f}% ({total_maj}/{total})")
            print("==========================================\n")

        print(f"\n[Run {run_idx+1}] Annotation accuracy: {num_correct}/{len(results)} = {run_accuracy:.4f} ({run_accuracy*100:.2f}%)")

    avg_accuracy = float(np.mean(all_run_accuracies)) if all_run_accuracies else 0.0
    std_accuracy = float(np.std(all_run_accuracies)) if all_run_accuracies else 0.0

    print(f"\n{'='*60}")
    print(f"  Summary: {NUM_RUNS} Runs Average Accuracy")
    print(f"{'='*60}")
    for i, acc in enumerate(all_run_accuracies):
        print(f"  Run {i+1}: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  {'─'*40}")
    print(f"  Average: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    print(f"  Std Dev: {std_accuracy:.4f} ({std_accuracy*100:.2f}%)")
    print(f"{'='*60}")

    has_multi_annotation = any(rd["multi_annotation_stats"]["total"] > 0 for rd in all_run_results)
    per_run_hit2: List[float] = []
    per_run_hit1: List[float] = []
    per_run_majority: List[float] = []
    if has_multi_annotation:
        for rd in all_run_results:
            stats = rd["multi_annotation_stats"]
            total = stats["total"]
            if total > 0:
                per_run_hit2.append(sum(stats["hit2_correct"].values()) / total)
                per_run_hit1.append(sum(stats["hit1_correct"].values()) / total)
                per_run_majority.append(sum(stats["majority_correct"].values()) / total)

        print(f"\n{'='*60}")
        print(f"  Summary: {NUM_RUNS} Runs Average Multi-Annotation Metrics")
        print(f"{'='*60}")
        print(f"  {'':>6}  {'Hit@2':>10}  {'Hit@1':>10}  {'Majority':>10}")
        for i in range(len(per_run_hit2)):
            print(f"  Run {i+1}: {per_run_hit2[i]*100:>9.2f}%  {per_run_hit1[i]*100:>9.2f}%  {per_run_majority[i]*100:>9.2f}%")
        print(f"  {'─'*48}")
        print(f"  Avg:   {np.mean(per_run_hit2)*100:>9.2f}%  {np.mean(per_run_hit1)*100:>9.2f}%  {np.mean(per_run_majority)*100:>9.2f}%")
        print(f"  Std:   {np.std(per_run_hit2)*100:>9.2f}%  {np.std(per_run_hit1)*100:>9.2f}%  {np.std(per_run_majority)*100:>9.2f}%")
        print(f"{'='*60}")

    elapsed_time = time.time() - start_time
    print(f"\nTotal time elapsed: {elapsed_time:.1f}s ({elapsed_time/60:.2f}min)")

    avg_multi_annotation = None
    if has_multi_annotation:
        avg_multi_annotation = {
            "per_run_hit2": per_run_hit2,
            "per_run_hit1": per_run_hit1,
            "per_run_majority": per_run_majority,
            "average_hit2": float(np.mean(per_run_hit2)),
            "average_hit1": float(np.mean(per_run_hit1)),
            "average_majority": float(np.mean(per_run_majority)),
            "std_hit2": float(np.std(per_run_hit2)),
            "std_hit1": float(np.std(per_run_hit1)),
            "std_majority": float(np.std(per_run_majority)),
        }

    output_data = {
        "mode": MODE,
        "num_runs": NUM_RUNS,
        "per_run_accuracies": all_run_accuracies,
        "average_accuracy": avg_accuracy,
        "std_accuracy": std_accuracy,
        "average_multi_annotation_metrics": avg_multi_annotation,
        "runs": [],
        "doc_info": {
            "used_best_epoch": USE_BEST_EPOCH_PROMPT,
            "best_epoch": best_epoch if USE_BEST_EPOCH_PROMPT else None,
            "best_train_accuracy": best_accuracy if USE_BEST_EPOCH_PROMPT else None,
            "average_test_accuracy": avg_accuracy,
            "used_optimized_doc": bool(optimized_doc),
        },
        "elapsed_time_seconds": round(elapsed_time, 1),
    }
    if MODE == "icl":
        output_data["icl_top_k"] = ICL_TOP_K
        output_data["temperature"] = ANNOTATOR_TEMPERATURE

    for i, run_data in enumerate(all_run_results):
        output_data["runs"].append({
            "run_index": i + 1,
            "accuracy": all_run_accuracies[i],
            "test_results": run_data["results"],
            "multi_annotation_statistics": run_data["multi_annotation_stats"] if run_data["multi_annotation_stats"]["total"] > 0 else None,
        })

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nDone! Results saved to: {output_file}")
    except Exception as e:
        print(f"\nSave failed: {e}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified single-agent annotation evaluation')
    parser.add_argument('--mode', type=str, default='cot', choices=['cot', 'direct', 'icl'],
                        help='Annotation strategy (default: cot)')
    parser.add_argument('--task_type', type=str, required=True, help='Task type')
    parser.add_argument('--model', type=str, default="Qwen3-VL-235B-A22B-Instruct",
                        help='Model name to use (default: Qwen3-VL-235B-A22B-Instruct)')
    parser.add_argument('--epoch', type=int, default=None, help='Which epoch result to use')
    parser.add_argument('--output_file_prefix', type=str, default=None, help='Output file prefix')
    parser.add_argument('--use_best_epoch_prompt', action='store_true', default=False,
                        help='Automatically use the best-accuracy epoch optimized rule doc')
    parser.add_argument('--use_optimized_prompt', action='store_true', default=False,
                        help='Use the optimized rule doc from the specified epoch')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of runs per sample; final result is the average accuracy')
    # ICL-only options (ignored in other modes)
    parser.add_argument('--icl_top_k', type=int, default=2,
                        help='[icl] Number of similar examples to retrieve (default: 2)')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='[icl] Annotation temperature (default: 0.1). cot/direct always use 0.1.')

    args = parser.parse_args()

    MODE = args.mode
    TASK_TYPE = args.task_type
    MODEL = args.model
    OUTPUT_FILE_PREFIX = args.output_file_prefix
    TEST_EPOCH = args.epoch
    USE_BEST_EPOCH_PROMPT = args.use_best_epoch_prompt
    USE_OPTIMIZED_PROMPT = args.use_optimized_prompt
    NUM_RUNS = args.num_runs
    ICL_TOP_K = args.icl_top_k
    ANNOTATOR_TEMPERATURE = args.temperature if MODE == "icl" else 0.1

    if USE_BEST_EPOCH_PROMPT and USE_OPTIMIZED_PROMPT:
        print("Error: --use_best_epoch_prompt and --use_optimized_prompt cannot be used together")
        sys.exit(1)

    run_evaluation()
