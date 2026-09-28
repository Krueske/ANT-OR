import openai
import json
import re
import sys
import os
import traceback
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple, Callable
from collections import Counter
import threading

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.evaluation_functions import EVALUATION_FUNCTIONS, extract_label_from_annotation
from utils.prompts import PROMPTS_EN as PROMPTS, OUTPUT_FORMATS
from utils.load_task import get_task_config
from utils.call_llm_api import call_llm_general
from utils.multimodal_utils import (
    extract_multimodal_data,
    format_multimodal_query,
    compute_multimodal_similarity,
    prompt_to_messages,
)

# Multimodal similarity weights (used by compute_multimodal_similarity below)
TEXT_SIMILARITY_WEIGHT = 0.5
IMAGE_SIMILARITY_WEIGHT = 0.5

# =============================================================================
# ICL example pool management
# =============================================================================

class ICLExamplePool:
    """ICL example pool: manages examples and supports similarity-based retrieval"""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.examples = []  # list of examples; each item has query and ground_truth

    def initialize_from_dataset(self, queries: List[Dict], ground_truths: List[Any]):
        """Initialize the example pool from a dataset"""
        self.examples = []
        for q, gt in zip(queries, ground_truths):
            self.examples.append({
                "query": q,
                "ground_truth": gt
            })
        print(f"[ICL Pool] Initialized with {len(self.examples)} examples")

    def retrieve_similar_examples(self, query_data: Dict) -> List[Tuple[Dict, float]]:
        """
        Retrieve top-K examples most similar to the current query.

        Returns:
            [(example, similarity_score), ...] sorted by similarity in descending order
        """
        if not self.examples:
            return []

        scored_examples = []
        for example in self.examples:
            similarity = compute_multimodal_similarity(query_data, example["query"])
            # Exclude exact duplicates (similarity >= 0.99)
            if similarity < 0.99:
                scored_examples.append((example, similarity))

        if not scored_examples:
            return []

        scored_examples.sort(key=lambda x: x[1], reverse=True)
        return scored_examples[:self.top_k]

    def format_examples_context(self, examples_with_scores: List[Tuple[Dict, float]]) -> Tuple[str, List[str]]:
        """
        Format examples into a context string, supporting interleaved text/image layout.

        Returns:
            (context_str, sample_image_urls): formatted context string and list of example image URLs
        """
        if not examples_with_scores:
            return "", []

        context_parts = []
        all_sample_image_urls = []

        for i, (example, similarity) in enumerate(examples_with_scores, 1):
            query_dict = example["query"]
            query_text, image_urls = extract_multimodal_data(query_dict)
            gt = example["ground_truth"]

            # Build display value for ground_truth
            if isinstance(gt, dict) and "vote_distribution_detail" in gt:
                vote_dist = gt.get("vote_distribution_detail", [])
                if vote_dist:
                    gt_display = vote_dist[0]["answer"]
                else:
                    all_annotations = gt.get("all_annotations", [])
                    gt_display = all_annotations[0] if all_annotations else ""
            else:
                gt_display = gt

            # Build interleaved query display with image placeholders
            # format_multimodal_query interleaves <image> placeholders with the query text
            query_with_images = format_multimodal_query(query_text, image_urls)
            all_sample_image_urls.extend(image_urls or [])

            example_section = f"""[Example {i}] (similarity: {similarity:.3f})
Query: {query_with_images}
Ground Truth: {gt_display}
"""
            context_parts.append(example_section)

        return "\n\n".join(context_parts), all_sample_image_urls


# =============================================================================
# Multimodal utility functions
# =============================================================================

DEFAULT_VERIFICATION_PROMPT_TEMPLATE = PROMPTS["verification_with_logic_scoring"]


# =============================================================================
# Agent class definitions
# =============================================================================

class AnnotatorAgent:
    """Single annotator agent: performs one annotation call with its own prompt; supports ICL example injection"""

    def __init__(self, agent_name: str, model: str, api_key: str, base_url: str,
                 task_type: str, temperature: float = 0.1):
        self.agent_name = agent_name
        self.model = model
        self.task_type = task_type
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.prompt_template = ""

    def set_prompt(self, prompt_template: str):
        """Set this agent's annotation prompt."""
        self.prompt_template = prompt_template

    def annotate(self, query_str: str, image_urls: List[str] = None,
                 icl_context: str = "", sample_image_urls: List[str] = None) -> Tuple[str, Dict]:
        """
        Perform one annotation call; supports ICL example injection.

        Args:
            query_str: query text
            image_urls: list of query image URLs
            icl_context: ICL example context string
            sample_image_urls: list of example image URLs
        """
        sample_image_urls = sample_image_urls or []

        # Build the prompt with ICL examples
        if icl_context:
            annotation_prompt = f"""You are a professional annotation expert. Please annotate the following Query according to the rules below.

### Annotation Rules:
{self.prompt_template}

### Reference Examples:
{icl_context}

### Query to Annotate:
{query_str}

### Please output in the following format:
Reasoning process: [Step-by-step analysis]
Annotation result: {OUTPUT_FORMATS[self.task_type]}
"""
        else:
            annotation_prompt = PROMPTS["annotation_based_on_generated_prompt"].format(
                generated_prompt=self.prompt_template,
                query=query_str,
                output_format=OUTPUT_FORMATS[self.task_type]
            )

        # Merge all image URLs: example images first, then the current query images
        all_image_urls = sample_image_urls + (image_urls or [])

        annotation_messages = prompt_to_messages(annotation_prompt, image_urls=all_image_urls if all_image_urls else None)
        annotation_result = call_llm_general(
            annotation_messages, self.model,
            temperature=self.temperature,
            app_name="experiment_on_prompt_optimization"
        )

        interaction = {
            "step": f"annotation_{self.agent_name}",
            "prompt": annotation_prompt,
            "response": annotation_result,
            "temperature": self.temperature,
            "icl_injected": bool(icl_context)
        }
        return annotation_result, interaction


class AggregationAgent:
    """Aggregation agent: aggregates results from multiple AnnotatorAgents via hard majority vote."""

    def __init__(self, agent_name: str = "aggregator"):
        self.agent_name = agent_name

    def aggregate(self, annotation_results: List[str], task_type: str) -> Tuple[str, Dict]:
        """Perform hard majority vote; returns the final result and vote info."""
        if task_type == "entity_extraction":
            return self._aggregate_entity_extraction(annotation_results)

        labels = []
        label_to_raw = {}
        empty_count = 0

        for raw_result in annotation_results:
            label = extract_label_from_annotation(raw_result, task_type)
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
                "vote_distribution": {},
                "winning_label": "",
                "winning_count": 0,
                "total_votes": total_votes,
                "confidence": 0.0,
                "margin": 0,
                "is_unanimous": False,
                "is_three_way_tie": False,
                "empty_results_count": empty_count
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
            "winning_label": winning_label,
            "winning_count": winning_count,
            "total_votes": total_votes,
            "confidence": winning_count / total_votes,
            "margin": winning_count - runner_up_count,
            "is_unanimous": winning_count == total_votes,
            "is_three_way_tie": is_three_way_tie,
            "empty_results_count": empty_count
        }

        final_result = label_to_raw.get(winning_label, annotation_results[0] if annotation_results else "")
        return final_result, vote_info

    def _aggregate_entity_extraction(self, annotation_results: List[str]) -> Tuple[str, Dict]:
        """Aggregation for entity_extraction: uses entity-set equality."""
        total_votes = len(annotation_results)

        result_sets = []
        for result in annotation_results:
            entity_set = self._extract_entity_set(result)
            result_sets.append((result, entity_set))

        clusters = {}
        for result, entity_set in result_sets:
            if not entity_set:
                key = frozenset()
            else:
                key = entity_set

            if key in clusters:
                rep_result, count = clusters[key]
                clusters[key] = (rep_result, count + 1)
            else:
                clusters[key] = (result, 1)

        cluster_list = [(entity_set, rep_result, count) for entity_set, (rep_result, count) in clusters.items()]
        cluster_list.sort(key=lambda x: x[2], reverse=True)

        if not cluster_list:
            return annotation_results[0] if annotation_results else "", {
                "vote_distribution": {},
                "winning_label": "",
                "winning_count": 0,
                "total_votes": total_votes,
                "confidence": 0.0,
                "margin": 0,
                "is_unanimous": False,
                "is_three_way_tie": False,
                "aggregation_method": "entity_set_equality"
            }

        winning_entity_set, winning_result, winning_count = cluster_list[0]
        runner_up_count = cluster_list[1][2] if len(cluster_list) > 1 else 0

        vote_dist = {}
        for entity_set, rep_result, count in cluster_list:
            label = "|".join(sorted(entity_set)) if entity_set else ""
            vote_dist[label] = count

        winning_label = "|".join(sorted(winning_entity_set)) if winning_entity_set else ""

        is_unanimous = winning_count == total_votes
        is_three_way_tie = len(cluster_list) == 3 and winning_count == 1

        vote_info = {
            "vote_distribution": vote_dist,
            "winning_label": winning_label,
            "winning_count": winning_count,
            "total_votes": total_votes,
            "confidence": winning_count / total_votes,
            "margin": winning_count - runner_up_count,
            "is_unanimous": is_unanimous,
            "is_three_way_tie": is_three_way_tie,
            "aggregation_method": "entity_set_equality",
            "num_clusters": len(cluster_list)
        }

        return winning_result, vote_info

    def _extract_entity_set(self, annotation: str) -> frozenset:
        """Extract entity set from an entity_extraction result (used for set comparison)."""
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
            if isinstance(data, dict):
                entities = data.get("entities", [])
            elif isinstance(data, list):
                entities = data
            else:
                return frozenset()

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


class VerificationAgent:
    """Verification Agent: Check reasoning logic consistency when annotators disagree."""

    def __init__(self, agent_name: str, model: str, api_key: str, base_url: str,
                 task_type: str, doc_content: str, temperature: float = 0.1):
        self.agent_name = agent_name
        self.model = model
        self.task_type = task_type
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.verification_prompt_template = DEFAULT_VERIFICATION_PROMPT_TEMPLATE

    def set_verification_prompt_template(self, template: str):
        self.verification_prompt_template = template

    def get_verification_prompt_template(self):
        return self.verification_prompt_template

    def verify(self, query_str: str, annotation_results: List[str],
               annotator_names: List[str], vote_info: Dict,
               image_urls: List[str] = None,
               icl_context: str = "", sample_image_urls: List[str] = None) -> Tuple[str, Dict, Dict]:
        """Check reasoning logic and select the best result (supports ICL injection)"""
        sample_image_urls = sample_image_urls or []

        annotations_text = ""
        for i, (name, result) in enumerate(zip(annotator_names, annotation_results)):
            label = extract_label_from_annotation(result, self.task_type)
            annotations_text += f"\n=== {name} (label: {label}) ===\n{result}\n"

        is_three_way_tie = vote_info.get("is_three_way_tie", False)
        if is_three_way_tie:
            vote_text = "Vote: Three-way tie (1:1:1), no majority"
        else:
            vote_text = f"Majority vote: {vote_info.get('winning_label', 'N/A')} ({vote_info.get('winning_count', 0)}/{vote_info.get('total_votes', 0)})"

        icl_section = ""
        if icl_context:
            icl_section = f"""### Reference Examples:
{icl_context}

"""

        verification_prompt = f"""{self.verification_prompt_template}

{icl_section}### Query to Verify:
{query_str}

Annotations:{annotations_text}
Vote Info: {vote_text}

Output format:
```json
{{
  "reason": ["logic error description for each annotator"],
  "result": ["annotator1/annotator2/annotator3"]
}}
```"""

        # Image URL order: example images (if any) first, then the current query images.
        # This matches the order of <image> placeholders in the prompt: example placeholders first,
        # then the current query placeholders.
        all_image_urls = sample_image_urls + (image_urls or [])
        verification_messages = prompt_to_messages(verification_prompt, image_urls=all_image_urls if all_image_urls else None)
        verification_result = call_llm_general(
            verification_messages, self.model,
            temperature=self.temperature,
            app_name="experiment_on_prompt_optimization"
        )

        interaction = {
            "step": "verification",
            "prompt": verification_prompt,
            "response": verification_result,
            "temperature": self.temperature,
            "icl_injected": bool(icl_context)
        }

        return self._parse_verification_result(
            verification_result, annotation_results, annotator_names, vote_info, interaction
        )

    def _parse_verification_result(self, verification_result: str, annotation_results: List[str],
                                   annotator_names: List[str], vote_info: Dict,
                                   interaction: Dict) -> Tuple[str, Dict, Dict]:
        """Parse verification result and determine final result."""
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', verification_result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = verification_result

            parsed = json.loads(json_str)
            selected_agent = parsed.get("result", ["majority_vote"])[0] if isinstance(parsed.get("result"), list) else parsed.get("result", "majority_vote")
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
                "selected_agent": selected_agent,
                "reason": reason,
                "was_verified": True
            }

        except Exception as e:
            print(f"  [VerificationAgent] Parse failed: {e}, fallback to majority vote")
            final_result = self._get_majority_result(annotation_results, vote_info)
            verification_info = {
                "selected_agent": "majority_vote_fallback",
                "reason": [f"Parse error: {str(e)}"],
                "was_verified": False
            }

        return final_result, verification_info, interaction

    def _get_majority_result(self, annotation_results: List[str], vote_info: Dict) -> str:
        """Get the result corresponding to majority vote."""
        winning_label = vote_info.get("winning_label", "")
        for result in annotation_results:
            label = extract_label_from_annotation(result, self.task_type)
            if label == winning_label:
                return result
        return annotation_results[0] if annotation_results else ""


# =============================================================================
# ICL Multi-Agent Annotator
# =============================================================================

class ICLMultiAgentAnnotator:
    """
    ICL Multi-Agent Annotator: coordinates multiple AnnotatorAgents and an AggregationAgent
    to complete annotation tasks, with optional ICL example injection.

    Architecture:
    - 3 AnnotatorAgents (using the same or different prompts) annotate in parallel,
      each receiving the same retrieved examples.
    - 1 AggregationAgent aggregates results via hard majority vote.
    - 1 VerificationAgent runs verification when annotators disagree.
    """

    def __init__(self, model: str, api_key: str, base_url: str, category_doc_path: str,
                 eval_func: Callable, task_type: str, num_annotators: int = 3,
                 annotator_temperature: float = 0.1,
                 enable_verification: bool = True,
                 verification_temperature: float = 0.1,
                 icl_top_k: int = 3):
        self.model = model
        self.eval_func = eval_func
        self.task_type = task_type
        self.num_annotators = num_annotators
        self.enable_verification = enable_verification
        self.icl_top_k = icl_top_k

        with open(category_doc_path, 'r', encoding='utf-8') as f:
            self.doc_content = f.read()

        # Initialize AnnotatorAgents
        self.annotators: List[AnnotatorAgent] = []
        for i in range(num_annotators):
            agent = AnnotatorAgent(
                agent_name=f"annotator{i+1}",
                model=model,
                api_key=api_key,
                base_url=base_url,
                task_type=task_type,
                temperature=annotator_temperature
            )
            self.annotators.append(agent)

        # Initialize AggregationAgent
        self.aggregator = AggregationAgent(agent_name="aggregator")

        # Initialize VerificationAgent
        self.verifier = VerificationAgent(
            agent_name="verifier",
            model=model,
            api_key=api_key,
            base_url=base_url,
            task_type=task_type,
            doc_content=self.doc_content,
            temperature=verification_temperature
        )

        # Initialize ICL example pool
        self.icl_pool = ICLExamplePool(top_k=icl_top_k)

        self.prompts: List[str] = []
        self._local = threading.local()

    def set_verification_prompt_template(self, template: str):
        self.verifier.set_verification_prompt_template(template)

    def get_verification_prompt_template(self):
        return self.verifier.get_verification_prompt_template()

    def set_prompts(self, prompts: List[str]):
        """Set prompts for all annotators."""
        self.prompts = prompts
        if len(prompts) == 1 and self.num_annotators > 1:
            for annotator in self.annotators:
                annotator.set_prompt(prompts[0])
        else:
            for i, annotator in enumerate(self.annotators):
                if i < len(prompts):
                    annotator.set_prompt(prompts[i])
                else:
                    annotator.set_prompt(prompts[-1])

    def initialize_icl_pool(self, queries: List[Dict], ground_truths: List[Any]):
        """Initialize the ICL example pool"""
        self.icl_pool.initialize_from_dataset(queries, ground_truths)

    def _evaluate_single_annotation(self, prediction: str, ground_truth: Any) -> Tuple[bool, int]:
        """Evaluate a single annotation against a multi-annotation dataset."""
        if isinstance(ground_truth, dict) and "all_annotations" in ground_truth:
            all_annotations = ground_truth.get("all_annotations", [])
            if not all_annotations:
                return False, 0
            match_count = sum(1 for ann in all_annotations if self.eval_func(prediction, ann))
            is_correct = match_count >= 2
            return is_correct, match_count
        else:
            is_correct = self.eval_func(prediction, ground_truth)
            return is_correct, 1 if is_correct else 0

    def evaluate_single(self, query_dict: Dict, ground_truth: Any):
        """Evaluate a single sample with multi-agent annotation (ICL-aware)"""
        query_text, image_urls = extract_multimodal_data(query_dict)
        query_str = format_multimodal_query(query_text, image_urls)
        llm_interactions = []
        self._local.interactions = llm_interactions

        # Step 1: retrieve similar examples
        similar_examples = self.icl_pool.retrieve_similar_examples(query_dict)
        icl_context, sample_image_urls = self.icl_pool.format_examples_context(similar_examples)
        icl_injected = bool(icl_context)

        if icl_injected:
            print(f"  [ICL] Retrieved {len(similar_examples)} similar examples")
            for i, (ex, sim) in enumerate(similar_examples):
                ex_text, _ = extract_multimodal_data(ex["query"])
                print(f"    - Example {i+1}: sim={sim:.3f}, query={ex_text[:50]}...")
        else:
            print(f"  [ICL] No similar examples retrieved")

        # Step 2: run AnnotatorAgents in parallel (all receive the same ICL examples)
        annotation_results = [None] * self.num_annotators
        agent_correctness = {}

        print(f"[evaluate_single] Launching {self.num_annotators} AnnotatorAgents in parallel...")

        with ThreadPoolExecutor(max_workers=self.num_annotators) as executor:
            future_to_idx = {}
            for idx, annotator in enumerate(self.annotators):
                future = executor.submit(
                    annotator.annotate,
                    query_str,
                    image_urls if image_urls else None,
                    icl_context,
                    sample_image_urls
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                agent_name = f"annotator{idx+1}"
                try:
                    annotation_result, interaction = future.result()
                    annotation_results[idx] = annotation_result
                    self._local.interactions.append(interaction)

                    is_annotator_correct, _ = self._evaluate_single_annotation(annotation_result, ground_truth)
                    agent_correctness[agent_name] = {
                        "is_correct": is_annotator_correct,
                        "annotation": annotation_result
                    }
                    print(f"  {agent_name}: annotation done, correct={is_annotator_correct}")
                except Exception as e:
                    print(f"  {agent_name}: annotation failed: {e}")
                    annotation_results[idx] = ""
                    agent_correctness[agent_name] = {"is_correct": False, "annotation": ""}

        # Step 3: AggregationAgent runs hard majority vote
        final_result, vote_info = self.aggregator.aggregate(annotation_results, self.task_type)
        self._local.interactions.append({
            "step": "aggregation",
            "prompt": "",
            "response": f"Hard majority vote result: {vote_info['winning_label']}",
            "vote_info": vote_info
        })

        # Step 4: if results disagree and verification is enabled, call VerificationAgent
        verification_info = None
        was_verified = False

        if self.enable_verification and not vote_info.get("is_unanimous", False):
            print(f"  [Verification] Inconsistent results detected, calling VerificationAgent...")
            annotator_names = [f"annotator{i+1}" for i in range(self.num_annotators)]
            final_result, verification_info, verification_interaction = self.verifier.verify(
                query_str=query_str,
                annotation_results=annotation_results,
                annotator_names=annotator_names,
                vote_info=vote_info,
                image_urls=image_urls if image_urls else None,
                icl_context=icl_context,
                sample_image_urls=sample_image_urls
            )
            self._local.interactions.append(verification_interaction)
            was_verified = verification_info.get("was_verified", False)
            selected_agent = verification_info.get("selected_agent", "unknown")
            print(f"  [Verification] Done, selected agent: {selected_agent}, parsed successfully: {was_verified}")

        # Step 5: evaluate the final result
        if isinstance(ground_truth, dict) and "all_annotations" in ground_truth:
            all_annotations = ground_truth.get("all_annotations", [])
            match_count = sum(1 for ann in all_annotations if self.eval_func(final_result, ann))
            is_correct = match_count >= 2
            print(f"  Aggregator: multi-annotation eval, matched {match_count} annotators, correct={is_correct}")
        else:
            is_correct = self.eval_func(final_result, ground_truth)
            print(f"  Aggregator: eval done, correct={is_correct}")

        agent_correctness["aggregator"] = {
            "is_correct": is_correct,
            "annotation": final_result
        }

        llm_interactions = self._local.interactions.copy()
        if hasattr(self._local, 'interactions'):
            del self._local.interactions

        # Assemble the return value
        result = {
            "query": query_str,
            "annotation": final_result,
            "is_correct": is_correct,
            "llm_interactions": llm_interactions,
            "annotation_results": annotation_results,
            "ground_truth": ground_truth,
            "vote_info": vote_info,
            "agent_correctness": agent_correctness,
            "context_injected": icl_injected,
            "per_agent_context_injected": {f"annotator{i+1}": icl_injected for i in range(self.num_annotators)},
            "relevant_dimensions": [],
            "was_verified": was_verified,
            "verification_info": verification_info,
            "icl_examples": [
                {
                    "query": extract_multimodal_data(ex["query"])[0],
                    "similarity": sim
                } for ex, sim in similar_examples
            ] if similar_examples else []
        }

        return result


# =============================================================================
# Main
# =============================================================================

def run_evaluation():
    global TASK_TYPE
    start_time = time.time()

    # ICL example pool range: 100-300
    icl_start_id = 100
    icl_end_id = 300
    # Test set range: 300-1300
    test_start_id = 300
    test_end_id = 1300

    max_workers = 3
    task_config = get_task_config(TASK_TYPE, test_end_id)
    TASK_TYPE = TASK_TYPE.replace("multi_", "")
    evaluate_func = EVALUATION_FUNCTIONS.get(TASK_TYPE)

    evaluator = ICLMultiAgentAnnotator(
        model=MODEL,
        api_key="afdca593-9da5-4570-9a26-3df4962c7b4f",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        category_doc_path=task_config["requirement_doc_path"],
        eval_func=evaluate_func,
        task_type=TASK_TYPE,
        num_annotators=NUM_ANNOTATORS,
        annotator_temperature=ANNOTATOR_TEMPERATURE,
        enable_verification=ENABLE_VERIFICATION,
        verification_temperature=VERIFICATION_TEMPERATURE,
        icl_top_k=ICL_TOP_K
    )

    # Initialize ICL example pool (100-300); if ICL is disabled, leave the pool empty:
    # retrieve_similar_examples returns [] and AnnotatorAgent.annotate falls back to the
    # standard prompt template when icl_context is empty, which is equivalent to plain CoT.
    if USE_ICL:
        icl_queries = list(task_config["queries"][icl_start_id:icl_end_id])
        icl_ground_truths = list(task_config["answers"][icl_start_id:icl_end_id])
        evaluator.initialize_icl_pool(icl_queries, icl_ground_truths)
    else:
        print("[ICL] disabled, running in plain CoT mode")

    # Test set (300-1300)
    queries = list(task_config["queries"][test_start_id:test_end_id])
    ground_truths = list(task_config["answers"][test_start_id:test_end_id])

    generated_prompts = []
    best_epoch = -1
    best_accuracy = 0.0
    selected_prompt = ""

    if USE_BEST_EPOCH_PROMPT and OUTPUT_FILE_PREFIX:
        # Find the best epoch prompt
        import glob
        search_pattern = f"output/{TASK_TYPE}/{TASK_TYPE}_multi_agents_prompt_optimization_with_{OUTPUT_FILE_PREFIX}_epoch*.json"
        epoch_files = glob.glob(search_pattern)

        if epoch_files:
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

                    if accuracy >= best_accuracy:
                        best_accuracy = accuracy
                        best_epoch = epoch
                        best_data = data

                except Exception as e:
                    continue

            if best_data is not None:
                if "current_generated_prompts" in best_data:
                    generated_prompts = best_data["current_generated_prompts"]
                elif "init_generated_prompts" in best_data:
                    generated_prompts = best_data["init_generated_prompts"]

                evaluator.set_prompts(generated_prompts)

                if "current_verification_prompt_template" in best_data:
                    evaluator.set_verification_prompt_template(best_data["current_verification_prompt_template"])
                    print(f"Loaded optimized verification_prompt_template from best epoch")

                print(f"\nUsing Epoch {best_epoch} prompt (train accuracy: {best_accuracy:.4f})")
            else:
                print("No best epoch data found, exiting")
                sys.exit(1)

            selected_prompt = generated_prompts[PROMPT_INDEX] if len(generated_prompts) > PROMPT_INDEX else generated_prompts[0]
            print(f"============ {'ICL' if USE_ICL else 'CoT'}: best epoch prompt ===============")
            print(f"Selected prompt index: {PROMPT_INDEX}")
            print(selected_prompt[:300] + "..." if len(selected_prompt) > 300 else selected_prompt)
            method_tag = f"icl_k{ICL_TOP_K}" if USE_ICL else "cot"
            output_file = f"output/{TASK_TYPE}/{TASK_TYPE}_evaluation_{method_tag}_{OUTPUT_FILE_PREFIX}_best_epoch{best_epoch}_n{NUM_ANNOTATORS}_t{ANNOTATOR_TEMPERATURE}.json"

        else:
            print("No epoch files found, exiting")
            sys.exit(1)

    elif USE_OPTIMIZED_PROMPT and OUTPUT_FILE_PREFIX and TEST_EPOCH is not None:
        optimized_prompt_path = f"output/{TASK_TYPE}/{TASK_TYPE}_multi_agents_prompt_optimization_with_{OUTPUT_FILE_PREFIX}_epoch{str(TEST_EPOCH)}.json"
        if os.path.exists(optimized_prompt_path):
            with open(optimized_prompt_path, "r", encoding='utf-8') as f:
                data = json.load(f)
            print(f"Read Optimized Prompt From {optimized_prompt_path}")

            if "current_generated_prompts" in data:
                generated_prompts = data["current_generated_prompts"]
            elif "init_generated_prompts" in data:
                generated_prompts = data["init_generated_prompts"]

            evaluator.set_prompts(generated_prompts)

            if "current_verification_prompt_template" in data:
                evaluator.set_verification_prompt_template(data["current_verification_prompt_template"])
                print(f"Loaded optimized verification_prompt_template from epoch {TEST_EPOCH}")

            selected_prompt = generated_prompts[PROMPT_INDEX] if len(generated_prompts) > PROMPT_INDEX else generated_prompts[0]
        else:
            print(f"Optimized prompt file not found: {optimized_prompt_path}")
            sys.exit(1)

        print(f"============ {'ICL' if USE_ICL else 'CoT'}: specified epoch prompt ===============")
        print(f"Selected prompt index: {PROMPT_INDEX}")
        print(selected_prompt[:300] + "..." if len(selected_prompt) > 300 else selected_prompt)
        method_tag = f"icl_k{ICL_TOP_K}" if USE_ICL else "cot"
        output_file = f"output/{TASK_TYPE}/{TASK_TYPE}_evaluation_{method_tag}_{OUTPUT_FILE_PREFIX}_epoch{TEST_EPOCH}_n{NUM_ANNOTATORS}_t{ANNOTATOR_TEMPERATURE}.json"

    else:
        # Baseline: directly use the original rule document as the prompt
        category_doc_path = task_config["requirement_doc_path"]
        with open(category_doc_path, 'r', encoding='utf-8') as f:
            selected_prompt = f.read()
        evaluator.set_prompts([selected_prompt]*3)
        print(f"Loaded baseline prompt from rule document: {category_doc_path}")

        print(f"============ {'ICL' if USE_ICL else 'CoT'}: baseline prompt (from rule doc) ===============")
        print(selected_prompt[:300] + "..." if len(selected_prompt) > 300 else selected_prompt)
        method_tag = f"icl_k{ICL_TOP_K}" if USE_ICL else "cot"
        output_file = f"output/{TASK_TYPE}/{TASK_TYPE}_evaluation_{method_tag}_baseline_{OUTPUT_FILE_PREFIX}_n{NUM_ANNOTATORS}_t{ANNOTATOR_TEMPERATURE}.json"

    icl_status = f"ON (top_k={ICL_TOP_K})" if USE_ICL else "OFF"
    print(f"\nMulti-Agent Annotation config: num_annotators={NUM_ANNOTATORS}, temperature={ANNOTATOR_TEMPERATURE}, verification={'ON' if ENABLE_VERIFICATION else 'OFF'}, icl={icl_status}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results = [None] * len(queries)

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
                results[idx]['query'] = json.dumps(queries[idx], ensure_ascii=False)

                vote_info = result.get("vote_info", {})
                conf = vote_info.get("confidence", 0)
                was_verified = result.get('was_verified', False)
                agent_corr = result.get('agent_correctness', {})
                annotator_correct_count = sum(1 for k, v in agent_corr.items() if k.startswith('annotator') and v.get('is_correct', False))
                icl_injected = result.get('context_injected', False)
                verified_str = " [V]" if was_verified else ""
                icl_str = " [ICL]" if icl_injected else ""
                print(f"Sample {idx} done: correct={result['is_correct']}, "
                      f"vote={vote_info.get('winning_label','?')} ({vote_info.get('winning_count',0)}/{vote_info.get('total_votes',0)}, "
                      f"conf={conf:.2f}, annotators_correct={annotator_correct_count}/{NUM_ANNOTATORS}){verified_str}{icl_str}")
            except Exception as e:
                print(f"\nIndex {idx} failed: {e}")
                traceback.print_exc()
                results[idx] = {
                    "error": str(e),
                    "query": json.dumps(queries[idx], ensure_ascii=False),
                    "ground_truth": ground_truths[idx],
                    "is_correct": False
                }

    # Accuracy stats
    num_correct = 0
    vote_confidence_stats = {"unanimous": 0, "high_conf": 0, "low_conf": 0}
    icl_injection_stats = {"injected": 0, "not_injected": 0}

    for result in results:
        if result.get("is_correct", False):
            num_correct += 1

        vote_info = result.get("vote_info", {})
        if vote_info.get("is_unanimous"):
            vote_confidence_stats["unanimous"] += 1
        elif vote_info.get("confidence", 0) >= 0.6:
            vote_confidence_stats["high_conf"] += 1
        else:
            vote_confidence_stats["low_conf"] += 1

        if result.get("context_injected", False):
            icl_injection_stats["injected"] += 1
        else:
            icl_injection_stats["not_injected"] += 1

    test_accuracy = num_correct / len(results) if results else 0

    # Print vote confidence distribution
    total_samples = len(results)
    method_label = "ICL Multi-Agent" if USE_ICL else "CoT Multi-Agent"
    print(f"\n========== {method_label} Vote Statistics ==========")
    print(f"Unanimous ({NUM_ANNOTATORS}/{NUM_ANNOTATORS}): {vote_confidence_stats['unanimous']} ({vote_confidence_stats['unanimous']/total_samples*100:.1f}%)")
    print(f"High confidence (>=60%):   {vote_confidence_stats['high_conf']} ({vote_confidence_stats['high_conf']/total_samples*100:.1f}%)")
    print(f"Low confidence (<60%):     {vote_confidence_stats['low_conf']} ({vote_confidence_stats['low_conf']/total_samples*100:.1f}%)")
    if USE_ICL:
        print(f"ICL Injection:             {icl_injection_stats['injected']}/{total_samples} ({icl_injection_stats['injected']/total_samples*100:.1f}%)")
    print(f"=====================================================")

    print(f"\nAnnotation accuracy: {num_correct}/{len(results)} = {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    elapsed_time = time.time() - start_time
    elapsed_min = elapsed_time / 60
    print(f"\nTotal time elapsed: {elapsed_time:.1f}s ({elapsed_min:.2f}min)")

    output_data = {
        "test_results": results,
        "config": {
            "method": "icl_multi_agent_annotation" if USE_ICL else "cot_multi_agent_annotation",
            "num_annotators": NUM_ANNOTATORS,
            "annotator_temperature": ANNOTATOR_TEMPERATURE,
            "enable_verification": ENABLE_VERIFICATION,
            "verification_temperature": VERIFICATION_TEMPERATURE,
            "prompt_index": PROMPT_INDEX,
            "used_best_epoch": USE_BEST_EPOCH_PROMPT,
            "best_epoch": best_epoch if USE_BEST_EPOCH_PROMPT else None,
            "best_train_accuracy": best_accuracy if USE_BEST_EPOCH_PROMPT else None,
            "test_accuracy": test_accuracy,
            "selected_prompt": selected_prompt,
            "use_icl": USE_ICL,
            "icl_top_k": ICL_TOP_K if USE_ICL else None,
            "icl_range": f"{icl_start_id}-{icl_end_id}" if USE_ICL else None,
            "test_range": f"{test_start_id}-{test_end_id}"
        },
        "vote_confidence_stats": vote_confidence_stats,
        "icl_injection_stats": icl_injection_stats,
        "elapsed_time_seconds": round(elapsed_time, 1)
    }

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nDone! Results saved to: {output_file}")
    except Exception as e:
        print(f"\nSave failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Agent annotation evaluation (CoT, optional ICL)')
    parser.add_argument('--task_type', type=str, required=True, help='Task type')
    parser.add_argument('--model', type=str, default='Qwen3-VL-235B-A22B-Instruct',
                       help='LLM model name (default: Qwen3-VL-235B-A22B-Instruct)')
    parser.add_argument('--epoch', type=int, default=None, help='Which epoch result to use')
    parser.add_argument('--output_file_prefix', type=str, default=None, help='Output file prefix')
    parser.add_argument('--use_best_epoch_prompt', action='store_true', default=False,
                       help='Auto-select best epoch prompt from training')
    parser.add_argument('--use_optimized_prompt', action='store_true', default=False,
                       help='Use specified epoch optimized prompt')
    parser.add_argument('--num_annotators', type=int, default=3,
                       help='Number of annotator agents (default: 3)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for annotator agents (default: 0.1)')
    parser.add_argument('--enable_verification', action='store_true', default=True,
                       help='Enable verification agent when annotators disagree (default: True)')
    parser.add_argument('--verification_temperature', type=float, default=0.1,
                       help='Temperature for verification agent (default: 0.1)')
    parser.add_argument('--prompt_index', type=int, default=0,
                       help='Which prompt to use from generated_prompts list (default: 0)')
    parser.add_argument('--use_icl', action='store_true', default=False,
                       help='Enable in-context-learning (retrieve similar examples). Default: False (plain CoT).')
    parser.add_argument('--icl_top_k', type=int, default=2,
                       help='Number of similar examples to retrieve for ICL (default: 2, only used when --use_icl)')

    args = parser.parse_args()

    TASK_TYPE = args.task_type
    MODEL = args.model
    OUTPUT_FILE_PREFIX = args.output_file_prefix
    TEST_EPOCH = args.epoch
    USE_BEST_EPOCH_PROMPT = args.use_best_epoch_prompt
    USE_OPTIMIZED_PROMPT = args.use_optimized_prompt
    NUM_ANNOTATORS = args.num_annotators
    ANNOTATOR_TEMPERATURE = args.temperature
    ENABLE_VERIFICATION = args.enable_verification
    VERIFICATION_TEMPERATURE = args.verification_temperature
    PROMPT_INDEX = args.prompt_index
    USE_ICL = args.use_icl
    ICL_TOP_K = args.icl_top_k

    if USE_BEST_EPOCH_PROMPT and USE_OPTIMIZED_PROMPT:
        print("Error: --use_best_epoch_prompt and --use_optimized_prompt cannot be used together")
        sys.exit(1)

    run_evaluation()
