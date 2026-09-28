import json
import re
import sys
import os
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from typing import Dict, List, Any, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.evaluation_functions import EVALUATION_FUNCTIONS
from utils.prompts import PROMPTS_EN as PROMPTS, OUTPUT_FORMATS
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
import argparse


# ============================================================================
# SingleAgentAnnotator: single-agent annotator (multimodal-aware)
# ============================================================================

class SingleAgentAnnotator:
    """Single-agent annotator: performs annotation and evaluation using a rule document; supports multimodal inputs"""

    def __init__(self, model: str, task_type: str, category_doc_path: str, eval_func: Callable):
        self.model = model
        self.eval_func = eval_func
        self.task_type = task_type
        with open(category_doc_path, 'r', encoding='utf-8') as f:
            self.doc_content = f.read()
        self.current_doc = self.doc_content
        self._local = threading.local()

    def get_current_doc(self):
        return self.current_doc

    def set_current_doc(self, doc: str):
        self.current_doc = doc

    @staticmethod
    def _extract_representative_gt(ground_truth: Any) -> Any:
        """
        Extract a representative answer from a multi-annotation GT (majority vote).
        For multi-annotation datasets, ground_truth is a dict containing all_annotations
        and vote_distribution_detail. For non-multi-annotation datasets, returns the
        original ground_truth as-is.
        """
        if isinstance(ground_truth, dict) and "vote_distribution_detail" in ground_truth:
            vote_dist = ground_truth.get("vote_distribution_detail", [])
            if vote_dist:
                return vote_dist[0]["answer"]
            all_annotations = ground_truth.get("all_annotations", [])
            if all_annotations:
                return all_annotations[0]
        return ground_truth

    def _evaluate_multi_annotation(self, prediction: str, ground_truth: Dict) -> Tuple[bool, int]:
        """
        Evaluate against a multi-annotation dataset: a prediction is counted as correct
        when it matches >=2 annotators.

        Args:
            prediction: model prediction
            ground_truth: multi-annotation ground_truth dict containing all_annotations

        Returns:
            (is_correct, match_count): correctness flag and number of matched annotators
        """
        all_annotations = ground_truth.get("all_annotations", [])
        if not all_annotations:
            return False, 0

        match_count = 0
        for ann in all_annotations:
            if self.eval_func(prediction, ann):
                match_count += 1

        is_correct = match_count >= 2
        return is_correct, match_count

    def _annotate(self, query_text: str, image_urls: List[str] = None) -> str:
        """Annotate using the current rule document (multimodal-aware)"""
        output_format = OUTPUT_FORMATS.get(self.task_type, '```json\n{"label": "annotation result"}```')
        query_str = format_multimodal_query(query_text, image_urls) if image_urls else query_text

        annotation_prompt = PROMPTS["annotation_based_on_generated_prompt"].format(
            generated_prompt=self.current_doc,
            query=query_str,
            output_format=output_format
        )
        annotation_messages = prompt_to_messages(annotation_prompt, image_urls=image_urls)
        annotation_result = call_llm_general(annotation_messages, self.model, temperature=0.1)
        return annotation_result

    def evaluate_single(self, query_dict: Dict, ground_truth: Any):
        """Evaluate a single sample (multimodal-aware)"""
        query_text, image_urls = extract_multimodal_data(query_dict)
        annotation_result = self._annotate(query_text, image_urls)

        if isinstance(ground_truth, dict) and "all_annotations" in ground_truth:
            is_correct, match_count = self._evaluate_multi_annotation(annotation_result, ground_truth)
        else:
            is_correct = self.eval_func(annotation_result, ground_truth)
            match_count = None

        result = {
            "query": query_text,
            "image_urls": image_urls,
            "annotation": annotation_result,
            "is_correct": is_correct,
            "ground_truth": ground_truth,
        }
        if match_count is not None:
            result["match_count"] = match_count
        return result

    def evaluate_batch(self, query_dicts: List, ground_truths: List) -> List[Dict]:
        """Batch evaluation"""
        results = [None] * len(query_dicts)
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
                    print(f"  Sample {idx} annotation completed")
                except Exception as e:
                    print(f"\n  Index {idx} failed: {e}")
                    traceback.print_exc()
                    results[idx] = {
                        "error": str(e),
                        "query": json.dumps(query_dicts[idx], ensure_ascii=False) if isinstance(query_dicts[idx], dict) else str(query_dicts[idx]),
                        "ground_truth": ground_truths[idx],
                        "is_correct": False,
                        "annotation": ""
                    }
        return results

    def evaluate_on_full_dataset(self, all_queries: List, all_ground_truths: List, batch_size: int = 50) -> Tuple[int, int, float]:
        """Evaluate the current rule document's accuracy on the entire training set"""
        print(f"\n========== Starting evaluation on full training set ({len(all_queries)} samples) ==========")
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
        print(f"========== Evaluation complete: {total_correct}/{total_count} = {accuracy:.4f} ({accuracy*100:.2f}%) ==========\n")
        return total_correct, total_count, accuracy


# ============================================================================
# VanillaSPOOptimizer: Vanilla SPO-style rule document optimizer (multimodal-aware)
# ============================================================================

class VanillaSPOOptimizer:
    """
    Vanilla SPO Optimizer:
    1. Collect error samples
    2. Feed error samples directly to the optimizer; the optimizer analyzes drawbacks
    3. Output in XML tag format (<analyse>, <modification>, <prompt>)
    4. Produce an optimized rule document
    """

    def __init__(self, model: str, task_type: str):
        self.model = model
        self.task_type = task_type

    def collect_error_samples(self, eval_results: List[Dict]) -> List[Dict]:
        """Collect error samples (no root-cause analysis)"""
        error_samples = []
        for idx, result in enumerate(eval_results):
            if not result.get("is_correct", False):
                gt_display = SingleAgentAnnotator._extract_representative_gt(result.get("ground_truth", ""))
                error_samples.append({
                    "query": result.get("query", ""),
                    "image_urls": result.get("image_urls", []),
                    "annotation": result.get("annotation", ""),
                    "ground_truth": gt_display
                })
        return error_samples

    def _build_error_samples_text(self, error_samples: List[Dict]) -> Tuple[str, List[str]]:
        """Build the textual representation of error samples; returns (text, list of all image URLs)"""
        text_parts = []
        all_image_urls = []
        for sid, sample in enumerate(error_samples):
            err_images = sample.get("image_urls", [])
            image_placeholder_str = ""
            if err_images:
                image_placeholder_str = "\n".join(["<image>" for _ in err_images]) + "\n"
                all_image_urls.extend(err_images)

            text_parts.append(
                f"Error sample {sid+1}:\n"
                f"{image_placeholder_str}"
                f"  Query to annotate: {sample['query']}\n"
                f"  Current output: {sample['annotation']}\n"
                f"  Expected correct answer: {sample['ground_truth']}"
            )
        return "\n\n".join(text_parts), all_image_urls

    def _build_golden_answers_text(self, error_samples: List[Dict]) -> str:
        """Build the textual representation of expected answers"""
        return "\n".join([
            f"Sample {sid+1}: {sample['ground_truth']}"
            for sid, sample in enumerate(error_samples)
        ])

    def optimize_doc(self, current_doc: str, error_samples: List[Dict]) -> Optional[str]:
        """
        Vanilla SPO-style rule document optimization.
        Feeds error samples directly to the optimizer; the optimizer analyzes drawbacks
        and produces an optimized rule document in XML tag format.
        """
        if not error_samples:
            print("No error samples, skipping optimization")
            return None

        error_text, error_image_urls = self._build_error_samples_text(error_samples)
        golden_text = self._build_golden_answers_text(error_samples)

        spo_optimize_prompt = f"""You are building an annotation rule document. Based on the given rule document, please reconstruct and optimize it.
You can add, modify, or delete rule content. Please include a single modification in XML tags in your reply.
During the optimization, you can incorporate any thinking models.
This is a rule document that performed excellently in a previous iteration. You must make further optimizations and improvements based on this document.
The modified rule document must differ from the provided example.

## Current Annotation Rule Document:
```
{current_doc}
```

## The execution result of this rule document (some error cases):
```
{error_text}
```

## The best answer we expect (for reference):
```
{golden_text}
```

Provide your analysis, optimization points, and the complete optimized rule document using the following XML format:
<analyse>Analyze what drawbacks exist in the results produced by the current rule document and how to improve them.</analyse>
<modification>Summarize the key points for improvement in one sentence</modification>
<prompt>Provide the complete optimized rule document here. It must be a full annotation rule document that can be directly used for annotation, not a summary or fragment.</prompt>
"""

        messages = prompt_to_messages(spo_optimize_prompt, image_urls=error_image_urls if error_image_urls else None)
        optimization_result = call_llm_general(
            messages,
            self.model,
            temperature=0.1,
            stream=True,
        )

        # Extract the optimized rule document from the XML tags
        return self._parse_optimized_doc(optimization_result)

    def _parse_optimized_doc(self, result: str) -> Optional[str]:
        """Parse the optimized rule document from the LLM output"""
        # First try extracting from the <prompt> tag
        prompt_match = re.search(r'<prompt>(.*?)</prompt>', result, re.DOTALL)
        if prompt_match:
            doc = prompt_match.group(1).strip()
            if doc:
                return doc

        # Fallback: try other markers
        for marker in ["Optimized rule document:", "Optimized Rule Document:", "optimized rule document:",
                       "优化后的规则文档：", "优化后的规则文档:"]:
            if marker in result:
                doc = result.split(marker)[-1].strip()
                if doc:
                    return doc

        return None


# ============================================================================
# Main function
# ============================================================================

def run_spo_optimization(args, task_config, evaluate_func, all_queries, all_ground_truths, queries, ground_truths):
    """Vanilla SPO optimization main loop"""
    model = args.model

    annotator = SingleAgentAnnotator(
        model=model,
        task_type=args.task_type,
        category_doc_path=task_config["requirement_doc_path"],
        eval_func=evaluate_func
    )
    optimizer = VanillaSPOOptimizer(model=model, task_type=args.task_type)

    init_doc = annotator.get_current_doc()
    optimize_better = 0
    optimize_worse = 0
    optimize_no_change = 0

    # Initial evaluation
    print("\n>>> Evaluating initial rule document:")
    init_correct, init_total, init_accuracy = annotator.evaluate_on_full_dataset(all_queries, all_ground_truths, batch_size=50)
    print(f">>> Initial accuracy: {init_correct}/{init_total} = {init_accuracy:.4f} ({init_accuracy*100:.2f}%)\n")

    prev_after_correct, prev_after_total, prev_after_accuracy = init_correct, init_total, init_accuracy
    epoch_accuracies = []

    for epoch in range(args.num_epochs):
        output_file = f"output/{args.task_type}/{args.task_type}_single_agent_doc_optimization_with_{args.output_file_prefix}_epoch{epoch}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Vanilla SPO Epoch {epoch+1}/{args.num_epochs} started")
        print(f"{'='*60}")

        before_correct, before_total, before_accuracy = prev_after_correct, prev_after_total, prev_after_accuracy

        # Shuffle data used for optimization
        combined = list(zip(queries, ground_truths))
        random.shuffle(combined)
        queries[:], ground_truths[:] = zip(*combined)

        results = [None] * len(queries)
        batch_size = 20
        num_rounds = args.num_rounds

        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i+batch_size]
            batch_ground_truths = ground_truths[i:i+batch_size]
            print(f"\n--- Batch {i//batch_size} started ---")

            try:
                # Evaluate the current batch
                eval_results = annotator.evaluate_batch(batch_queries, batch_ground_truths)
                ori_num_correct = sum(1 for r in eval_results if r.get("is_correct", False))
                ori_doc = annotator.get_current_doc()

                if ori_num_correct < len(batch_queries):
                    # Collect error samples (no root-cause analysis)
                    error_samples = optimizer.collect_error_samples(eval_results)
                    error_count = len(error_samples)
                    print(f"  Current batch accuracy: {ori_num_correct}/{len(batch_queries)}, {error_count} errors, starting Vanilla SPO optimization...")

                    # Vanilla SPO: multiple optimization rounds
                    for round_id in range(num_rounds):
                        print(f"  Round {round_id+1}/{num_rounds} optimization started")

                        # Run optimization directly on the error samples
                        new_doc = optimizer.optimize_doc(ori_doc, error_samples)

                        if new_doc:
                            annotator.set_current_doc(new_doc)
                            new_eval_results = annotator.evaluate_batch(batch_queries, batch_ground_truths)
                            new_num_correct = sum(1 for r in new_eval_results if r.get("is_correct", False))

                            if new_num_correct > ori_num_correct:
                                optimize_better += 1
                                print(f"  Vanilla SPO optimization succeeded: {ori_num_correct} -> {new_num_correct}")
                                eval_results = new_eval_results
                                ori_doc = new_doc
                                ori_num_correct = new_num_correct
                            elif new_num_correct < ori_num_correct:
                                optimize_worse += 1
                                print(f"  Vanilla SPO optimization reverted: {ori_num_correct} -> {new_num_correct}")
                                annotator.set_current_doc(ori_doc)
                            else:
                                optimize_no_change += 1
                                print(f"  Vanilla SPO optimization no change: {ori_num_correct} -> {new_num_correct}")
                                eval_results = new_eval_results
                                ori_doc = new_doc
                                ori_num_correct = new_num_correct

                            # If everything is correct, stop early
                            if ori_num_correct >= len(batch_queries):
                                print(f"  All samples in batch are correct, skipping remaining rounds")
                                break

                            # Re-collect error samples for the next optimization round
                            error_samples = optimizer.collect_error_samples(eval_results)
                            if not error_samples:
                                print(f"  No more error samples, optimization complete")
                                break
                        else:
                            print("  Vanilla SPO optimization failed to generate new document")
                            continue
                else:
                    print(f"  Batch all correct, no optimization needed")

                # Save batch results
                for batch_id, idx in enumerate(range(i, min(i+batch_size, len(queries)))):
                    results[idx] = {
                        "annotation": eval_results[batch_id].get("annotation", ""),
                        "is_correct": eval_results[batch_id]["is_correct"],
                        "query": eval_results[batch_id].get("query", ""),
                        "image_urls": eval_results[batch_id].get("image_urls", []),
                        "ground_truth": ground_truths[idx],
                        "current_doc": annotator.get_current_doc()
                    }

            except Exception as e:
                print(f"\n  Batch {i//batch_size} failed: {e}")
                traceback.print_exc()
                for idx in range(i, min(i+batch_size, len(queries))):
                    results[idx] = {
                        "error": str(e),
                        "query": json.dumps(queries[idx], ensure_ascii=False) if isinstance(queries[idx], dict) else str(queries[idx]),
                        "ground_truth": ground_truths[idx],
                        "is_correct": False
                    }

        # After the epoch, evaluate on the full training set
        print(f"\n>>> Epoch {epoch+1} post-optimization full evaluation:")
        after_correct, after_total, after_accuracy = annotator.evaluate_on_full_dataset(all_queries, all_ground_truths, batch_size=50)
        prev_after_correct, prev_after_total, prev_after_accuracy = after_correct, after_total, after_accuracy

        accuracy_change = after_accuracy - before_accuracy
        change_symbol = "↑" if accuracy_change > 0 else ("↓" if accuracy_change < 0 else "-")

        print(f">>> Epoch {epoch+1}: {before_accuracy*100:.2f}% -> {after_accuracy*100:.2f}% ({change_symbol}{abs(accuracy_change)*100:.2f}%)")
        print(f">>> Optimization succeeded: {optimize_better}, failed: {optimize_worse}, no change: {optimize_no_change}")

        epoch_accuracies.append({
            "epoch": epoch + 1,
            "before": {"correct": before_correct, "total": before_total, "accuracy": before_accuracy, "accuracy_percent": f"{before_accuracy*100:.2f}%"},
            "after": {"correct": after_correct, "total": after_total, "accuracy": after_accuracy, "accuracy_percent": f"{after_accuracy*100:.2f}%"},
            "change": {"absolute": accuracy_change, "percent": f"{accuracy_change*100:.2f}%", "symbol": change_symbol}
        })

        # Save results
        output_data = {
            "method": "vanilla_spo_single_agent",
            "batch_results": results,
            "full_dataset_evaluation": {
                "before_optimization": {"correct": before_correct, "total": before_total, "accuracy": before_accuracy, "accuracy_percent": f"{before_accuracy*100:.2f}%"},
                "after_optimization": {"correct": after_correct, "total": after_total, "accuracy": after_accuracy, "accuracy_percent": f"{after_accuracy*100:.2f}%"},
                "change": {"absolute": accuracy_change, "percent": f"{accuracy_change*100:.2f}%", "symbol": change_symbol}
            },
            "epoch_accuracies_so_far": epoch_accuracies,
            "init_doc": init_doc,
            "current_doc": annotator.get_current_doc(),
            "optimize_better": optimize_better,
            "optimize_worse": optimize_worse,
            "optimize_no_change": optimize_no_change,
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"  Results saved to: {output_file}")
        except Exception as e:
            print(f"  Save failed: {e}")

    _print_summary(epoch_accuracies, "Vanilla SPO Single Agent")


def _print_summary(epoch_accuracies, method_name):
    """Print accuracy summary across all epochs"""
    print(f"\n{'='*70}")
    print(f"{method_name} All Epoch Accuracy Summary:")
    print(f"{'='*70}")
    print(f"  {'Epoch':<8} {'Before':<18} {'After':<18} {'Change':<12}")
    print(f"{'-'*70}")
    for acc_info in epoch_accuracies:
        before_str = f"{acc_info['before']['accuracy']:.4f} ({acc_info['before']['accuracy_percent']})"
        after_str = f"{acc_info['after']['accuracy']:.4f} ({acc_info['after']['accuracy_percent']})"
        change_str = f"{acc_info['change']['symbol']} {abs(acc_info['change']['absolute']):.4f} ({acc_info['change']['percent']})"
        print(f"  {acc_info['epoch']:<8} {before_str:<18} {after_str:<18} {change_str:<12}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Vanilla SPO Single Agent for Annotation Rule Document Optimization (text + multimodal)')
    parser.add_argument('--task_type', type=str, required=True, help='Task type')
    parser.add_argument('--output_file_prefix', type=str, required=True, help='Output file prefix')
    parser.add_argument('--model', type=str, default='Qwen3-VL-235B-A22B-Instruct',
                        help='LLM model name (default: Qwen3-VL-235B-A22B-Instruct)')
    parser.add_argument('--num_epochs', type=int, default=6, help='Number of training epochs')
    parser.add_argument('--num_rounds', type=int, default=3, help='Number of optimization rounds per batch')
    parser.add_argument('--start_id', type=int, default=100, help='Data start ID')
    parser.add_argument('--end_id', type=int, default=300, help='Data end ID')
    args = parser.parse_args()

    task_config = get_task_config(args.task_type, args.end_id)
    evaluate_func = EVALUATION_FUNCTIONS.get(args.task_type)

    if evaluate_func is None:
        raise ValueError(f"Unsupported task type: {args.task_type}")

    all_queries = task_config["queries"][args.start_id:args.end_id]
    all_ground_truths = task_config["answers"][args.start_id:args.end_id]

    queries = all_queries.copy()
    ground_truths = all_ground_truths.copy()

    os.makedirs(f"output/{args.task_type}", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Method: Vanilla SPO Single Agent")
    print(f"Task: {args.task_type}")
    print(f"Data range: [{args.start_id}, {args.end_id})")
    print(f"Epochs: {args.num_epochs}")
    print(f"Rounds per batch: {args.num_rounds}")
    print(f"{'='*60}")

    run_spo_optimization(args, task_config, evaluate_func, all_queries, all_ground_truths, queries, ground_truths)


if __name__ == "__main__":
    main()
