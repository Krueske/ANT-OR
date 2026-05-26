import openai
import json
import re
import numpy as np
import pandas as pd
import sys
import os
import traceback
import copy
import random
import time
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple, Optional, Callable

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.evaluation_functions import EVALUATION_FUNCTIONS, extract_label_from_annotation
from utils.prompts import PROMPTS_EN as PROMPTS, OUTPUT_FORMATS
from utils.prompts import build_contrastive_context, build_dimension_sample_prompt, build_verification_prompt
from utils.load_task import get_task_config
from utils.call_llm_api import call_llm_general
from utils.samples_retriever import get_or_create_embedding, cosine_similarity
from utils.token_counter import TokenCounter
from collections import Counter
import threading

token_counter = TokenCounter()

CONTEXT_INJECTION_TOP_K = 2
USE_CONTRASTIVE_CONTEXT = True   


class SampleQualityScorer:
    def __init__(self, step_size: float = 0.1):
        self.step_size = step_size
        self.sample_scores = {}
        self.usage_history = []
        self.global_correct_count = 0
        self.global_incorrect_count = 0
        
    def get_global_correct_rate(self) -> float:
        total = self.global_correct_count + self.global_incorrect_count
        if total == 0:
            return 0.5
        return self.global_correct_count / total
    
    def update_global_stats(self, was_correct: bool):
        if was_correct:
            self.global_correct_count += 1
        else:
            self.global_incorrect_count += 1
    
    def get_score(self, query_str: str) -> float:
        if query_str not in self.sample_scores:
            return 1.0
        return self.sample_scores[query_str]['score']
    
    def compute_weighted_similarity(self, query_str: str, 
                                     base_similarity: float) -> float:
        quality_score = self.get_score(query_str)
        return base_similarity * quality_score
    
    def get_sample_stats(self, query_str: str = None) -> Dict:
        if query_str:
            if query_str not in self.sample_scores:
                return {'score': 1.0, 'usage_count': 0}
            return self.sample_scores[query_str].copy()
        
        if not self.sample_scores:
            return {'total_samples': 0, 'avg_score': 1.0}
        
        scores = [s['score'] for s in self.sample_scores.values()]
        usage_counts = [s['usage_count'] for s in self.sample_scores.values()]
        
        return {
            'total_samples': len(self.sample_scores),
            'avg_score': np.mean(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'total_usage': sum(usage_counts),
            'avg_usage_per_sample': np.mean(usage_counts) if usage_counts else 0
        }
    
    def export_state(self) -> Dict:
        return {
            'sample_scores': self.sample_scores,
            'usage_history': self.usage_history[-1000:],
        }
    
    def import_state(self, state: Dict):
        self.sample_scores = state.get('sample_scores', {})
        self.usage_history = state.get('usage_history', [])


class SampleSelector:
    def __init__(self, strategy: str = "deterministic", temperature: float = 0.1):
        self.strategy = strategy
        self.temperature = temperature
        self.selection_counts = {}
    
    def select(self, candidates: List[Tuple[Dict, float]], 
               use_sampling: bool = True) -> Tuple[Optional[Dict], float]:
        if not candidates:
            return None, 0.0
        
        best = max(candidates, key=lambda x: x[1])
        self._update_selection_count(best[0])
        return best
    
    def _update_selection_count(self, sample: Dict):
        query_str = sample.get('query', '')
        if query_str:
            self.selection_counts[query_str] = self.selection_counts.get(query_str, 0) + 1
    
    def reset_counts(self):
        self.selection_counts = {}


def call_llm_tracked(messages, model, temperature=0.1, step_name="unknown", app_name="experiment_on_prompt_optimization"):
    result = call_llm_general(messages, model, temperature=temperature, app_name=app_name)
    token_counter.record(messages, result, step_name=step_name)
    return result


def prompt_to_messages(prompt):
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content":  prompt
        }
    ]


class ContrastiveExamplePool:
    def __init__(self, task_type: str, output_file_prefix: str):
        self.task_type = task_type
        self.output_file_prefix = output_file_prefix
        self.all_samples = []
        self.correct_samples = []
        self.incorrect_samples = []
        self.historical_badcases = []
        self._evaluated_flags = {}
        
        self.agent_names = ["annotator1", "annotator2", "annotator3", "verifier"]
        self.agent_correct_samples = {agent: [] for agent in self.agent_names}
        self.agent_incorrect_samples = {agent: [] for agent in self.agent_names}
        self._agent_evaluated_flags = {agent: {} for agent in self.agent_names}
        
        self.dimension_sample_index = {}
        self.sample_dimension_tags = {}
        
        self.sample_scorer = SampleQualityScorer()
        self.sample_selector = SampleSelector(strategy="deterministic", temperature=0.1)
        self._injected_samples_cache = {}
        self.use_sampling = False
    
    def _extract_representative_gt(self, ground_truth: Any) -> Any:
        if isinstance(ground_truth, dict) and "vote_distribution_detail" in ground_truth:
            vote_dist = ground_truth.get("vote_distribution_detail", [])
            if vote_dist:
                return vote_dist[0]["answer"]
            all_annotations = ground_truth.get("all_annotations", [])
            if all_annotations:
                return all_annotations[0]
        return ground_truth

    def initialize_from_dataset(self, queries: List[Dict], ground_truths: List[Any]):
        self.all_samples = []
        for q, gt in zip(queries, ground_truths):
            query_str = json.dumps(q, ensure_ascii=False)
            self.all_samples.append({
                "query": query_str,
                "ground_truth": gt,
                "query_dict": q
            })
    
    def load_historical_badcases(self):
        badcases_file = f"cases/{self.task_type}/badcases_{self.output_file_prefix}.jsonl"
        self.historical_badcases = []
        if os.path.exists(badcases_file):
            with open(badcases_file, "r", encoding="utf-8") as f:
                for line in f:
                    self.historical_badcases.append(json.loads(line, strict=False))
        print(f"[Pool] Loaded badcases: {len(self.historical_badcases)}")
    
    def load_from_checkpoint(self, checkpoint_data: Dict):
        if "pool_state" not in checkpoint_data:
            return
        
        pool_state = checkpoint_data["pool_state"]
        
        if "correct_samples" in pool_state:
            self.correct_samples = pool_state["correct_samples"]
        
        if "incorrect_samples" in pool_state:
            self.incorrect_samples = pool_state["incorrect_samples"]
        
        if "agent_correct_samples" in pool_state:
            loaded_agent_correct = pool_state["agent_correct_samples"]
            for agent in self.agent_names:
                if agent in loaded_agent_correct:
                    self.agent_correct_samples[agent] = loaded_agent_correct[agent]
        
        if "agent_incorrect_samples" in pool_state:
            loaded_agent_incorrect = pool_state["agent_incorrect_samples"]
            for agent in self.agent_names:
                if agent in loaded_agent_incorrect:
                    self.agent_incorrect_samples[agent] = loaded_agent_incorrect[agent]
        
        if "dimension_sample_index" in pool_state:
            self.dimension_sample_index = pool_state["dimension_sample_index"]

        if "sample_dimension_tags" in pool_state:
            self.sample_dimension_tags = pool_state["sample_dimension_tags"]

        if "sample_scorer_state" in pool_state:
            self.sample_scorer.import_state(pool_state["sample_scorer_state"])
    
    def get_agent_correct_samples(self, agent_name: str) -> List[Dict]:
        if agent_name not in self.agent_names:
            return []
        return self.agent_correct_samples.get(agent_name, [])
    
    def get_agent_incorrect_samples(self, agent_name: str) -> List[Dict]:
        if agent_name not in self.agent_names:
            return []
        return self.agent_incorrect_samples.get(agent_name, [])
    
    def get_agent_contrastive_context(self, agent_name: str, query_str: str,
                                       relevant_dimensions: List[Dict] = None) -> Tuple[str, List[Dict], List[Dict], Dict[str, float]]:
        """
        Returns:
            (context_str, correct_samples_list, incorrect_samples_list, similarity_scores)
        """
        is_system = (agent_name == "system")
        top_k = CONTEXT_INJECTION_TOP_K

        if is_system or agent_name == "verifier":
            correct_candidates = self.correct_samples
        else:
            correct_candidates = self.get_agent_correct_samples(agent_name)

        correct_matches = self._retrieve_top_k_matches_with_score(
            query_str, correct_candidates, top_k, relevant_dimensions, exclude_query=query_str
        )

        if is_system:
            incorrect_candidates = self.historical_badcases if self.historical_badcases else self.incorrect_samples
        else:
            incorrect_candidates = [
                bc for bc in self.historical_badcases
                if bc.get("agent_name") == agent_name
            ]
            if not incorrect_candidates:
                incorrect_candidates = self.get_agent_incorrect_samples(agent_name)

        incorrect_matches = self._retrieve_top_k_matches_with_score(
            query_str, incorrect_candidates, top_k, relevant_dimensions, exclude_query=query_str
        )

        correct_samples_list = [m[0] for m in correct_matches]
        incorrect_samples_list = [m[0] for m in incorrect_matches]

        similarity_scores = {
            'correct': correct_matches[0][1] if correct_matches else 0.0,
            'incorrect': incorrect_matches[0][1] if incorrect_matches else 0.0,
            'correct_all': [m[1] for m in correct_matches],
            'incorrect_all': [m[1] for m in incorrect_matches]
        }

        context_str = self._format_contrastive_context_multi(
            correct_samples_list, incorrect_samples_list, similarity_scores, relevant_dimensions
        )

        return context_str, correct_samples_list, incorrect_samples_list, similarity_scores

    def _retrieve_top_k_matches_with_score(self, query_str: str, candidates: List[Dict],
                                            top_k: int = 3,
                                            relevant_dimensions: List[Dict] = None,
                                            exclude_query: str = None) -> List[Tuple[Dict, float, float]]:
        """
        Returns:
            [(sample, similarity_score, weighted_score), ...] Sort by weighted_score in descending order
        """
        if not candidates:
            return []

        query_embedding = get_or_create_embedding(query_str)
        if query_embedding == 0:
            return [(candidates[0], 0.0, 0.0)] if candidates else []

        enhanced_queries = [query_str]
        if relevant_dimensions:
            dim_keywords = []
            for dim in relevant_dimensions:
                for key in ["meta_rule", "decision_boundary", "applicable_boundary"]:
                    text = dim.get(key, "")
                    if text:
                        dim_keywords.append(text)
            if dim_keywords:
                enhanced_queries.extend(dim_keywords)

        scored_candidates = []
        for cand in candidates:
            cand_query = cand.get("query", "")
            if not cand_query:
                continue
            if exclude_query and cand_query == exclude_query:
                continue

            cand_embedding = get_or_create_embedding(cand_query)
            if cand_embedding == 0:
                continue

            scores = [cosine_similarity(get_or_create_embedding(q), cand_embedding) for q in enhanced_queries]
            avg_similarity = sum(scores) / len(scores)

            weighted_score = self.sample_scorer.compute_weighted_similarity(cand_query, avg_similarity)

            scored_candidates.append((cand, avg_similarity, weighted_score))

        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        return scored_candidates[:top_k]

    def _format_contrastive_context_multi(self, correct_samples: List[Dict],
                                           incorrect_samples: List[Dict],
                                           similarity_scores: Dict = None,
                                           relevant_dimensions: List[Dict] = None) -> str:
        return build_contrastive_context(
            correct_samples=correct_samples,
            incorrect_samples=incorrect_samples,
            similarity_scores=similarity_scores,
            relevant_dimensions=relevant_dimensions,
            include_images=False,
            prompts_dict=PROMPTS
        )
    
    def _retrieve_best_match_with_score(self, query_str: str, candidates: List[Dict], 
                                        relevant_dimensions: List[Dict] = None,
                                        exclude_query: str = None,
                                        use_quality_score: bool = True) -> Tuple[Optional[Dict], float, float]:
        query_embedding = get_or_create_embedding(query_str)
        if query_embedding == 0:
            return (candidates[0] if candidates else None), 0.0, 0.0
        
        enhanced_queries = [query_str]
        if relevant_dimensions:
            dim_keywords = []
            for dim in relevant_dimensions:
                for key in ["meta_rule", "decision_boundary", "applicable_boundary"]:
                    text = dim.get(key, "")
                    if text:
                        dim_keywords.append(text)
            if dim_keywords:
                enhanced_queries.extend(dim_keywords)
        
        best_match = None
        best_sim_score = -1.0
        best_weighted_score = -1.0
        
        for cand in candidates:
            cand_query = cand.get("query", "")
            if not cand_query:
                continue
            
            if exclude_query and cand_query == exclude_query:
                continue
            
            cand_embedding = get_or_create_embedding(cand_query)
            if cand_embedding == 0:
                continue
            
            scores = [cosine_similarity(get_or_create_embedding(q), cand_embedding) for q in enhanced_queries]
            avg_similarity = sum(scores) / len(scores)
            
            if use_quality_score:
                weighted_score = self.sample_scorer.compute_weighted_similarity(cand_query, avg_similarity)
            else:
                weighted_score = avg_similarity
            
            if weighted_score > best_weighted_score:
                best_weighted_score = weighted_score
                best_sim_score = avg_similarity
                best_match = cand
        
        return best_match, best_sim_score, best_weighted_score
    
    
    def find_dimensions_for_sample(self, sample_query: str, dimension_manager: 'DimensionManager' = None) -> List[Dict]:
        if not sample_query or sample_query not in self.sample_dimension_tags:
            return []
        
        dim_names = self.sample_dimension_tags[sample_query]
        dimensions = []
        
        for dim_name in dim_names:
            if dim_name in self.dimension_sample_index:
                if dimension_manager and dim_name in dimension_manager.intent_hypotheses:
                    dimensions.append(dimension_manager.intent_hypotheses[dim_name])
                else:
                    sample_info = self.dimension_sample_index[dim_name].get(sample_query, {})
                    dimensions.append({
                        'name': dim_name,
                        'is_correct': sample_info.get('is_correct', True)
                    })
        
        return dimensions


class DimensionManager:
    def __init__(self, task_type: str, output_file_prefix: str, model: str = None):
        self.task_type = task_type
        self.output_file_prefix = output_file_prefix
        self.model = model
        self.intent_hypotheses = {}
        self._similarity_history = []
        self._history_max_size = 1000
        self._load_intent_hypotheses()
    
    def _load_intent_hypotheses(self):
        path = f"cases/{self.task_type}/intent_hypotheses_{self.output_file_prefix}.json"
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.intent_hypotheses = json.load(f)
    
    def load_from_checkpoint(self, checkpoint_data: Dict):
        if "intent_hypotheses" in checkpoint_data:
            checkpoint_dims = checkpoint_data["intent_hypotheses"]
            if checkpoint_dims:
                merged_dims = self.intent_hypotheses.copy()
                merged_dims.update(checkpoint_dims)
                self.intent_hypotheses = merged_dims
                for dim_name, dim_info in self.intent_hypotheses.items():
                    evidence = dim_info.get("evidence_count", 0)
                    rule_intent = dim_info.get("rule_intent", "")
                    print(f"  - {dim_name}: evidence_count={evidence}, intent={rule_intent}...")
        
        if "dimension_manager_state" in checkpoint_data:
            dm_state = checkpoint_data["dimension_manager_state"]
            if "similarity_history" in dm_state:
                self._similarity_history = dm_state["similarity_history"]
    

class Annotator:
    def __init__(self, model: str, api_key: str, base_url: str, task_type: str,
                 category_doc_path: str, eval_func: Callable, output_file_prefix: str):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.eval_func = eval_func
        self.task_type = task_type
        self.output_file_prefix = output_file_prefix

        with open(category_doc_path, 'r', encoding='utf-8') as f:
            self.doc_content = f.read()

        self.generated_prompts = []
        self.verification_prompt_template = PROMPTS["default_verification_prompt_template"]
        self._local = threading.local()
        self.pool = ContrastiveExamplePool(task_type, output_file_prefix)
        self.dimension_manager = DimensionManager(task_type, output_file_prefix, model)
    
    def set_generated_prompts(self, generated_prompts: List[str]):
        self.generated_prompts = generated_prompts

    def set_verification_prompt_template(self, template: str):
        self.verification_prompt_template = template

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
        match_count = 0
        for ann in all_annotations:
            if self.eval_func(prediction, ann):
                match_count += 1

        hit_rate = match_count / total_annotators if total_annotators > 0 else 0.0
        agreement_level = ground_truth.get("agreement_level", "unknown")
        is_correct_majority = False
        if vote_distribution_detail:
            max_votes = vote_distribution_detail[0]["count"] if vote_distribution_detail else 0
            top_answers = []
            for item in vote_distribution_detail:
                if item["count"] == max_votes:
                    top_answers.append(item["answer"])
                else:
                    break

            for top_answer in top_answers:
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
            "agreement_level": agreement_level
        }
    
    def _build_dimension_sample_prompt(self, base_prompt: str, query: str,
                                       contrastive_context: str,
                                       sample_dimensions: List[Dict]) -> str:
        return build_dimension_sample_prompt(
            base_prompt=base_prompt,
            query=query,
            contrastive_context=contrastive_context,
            sample_dimensions=sample_dimensions,
            output_format=OUTPUT_FORMATS.get(self.task_type, ""),
            prompts_dict=PROMPTS
        )
    
    def _annotate_single(self, query_str: str, prompt: str,
                         contrastive_context: str, sample_dimensions: List[Dict], 
                         agent_name: str) -> str:
        sleep_time = random.random() * 0.5
        time.sleep(sleep_time)
        
        has_context = bool(contrastive_context) or len(sample_dimensions) > 0
        
        if has_context:
            annotation_prompt = self._build_dimension_sample_prompt(
                base_prompt=prompt,
                query=query_str,
                contrastive_context=contrastive_context,
                sample_dimensions=sample_dimensions
            )
        else:
            annotation_prompt = PROMPTS["annotation_based_on_generated_prompt"].format(
                generated_prompt=prompt,
                query=query_str,
                output_format=OUTPUT_FORMATS[TASK_TYPE]
            )
        
        annotation_messages = prompt_to_messages(annotation_prompt)
        annotation_result = call_llm_tracked(annotation_messages, self.model, temperature=0.1,
                                             step_name=f"annotation_{agent_name}",
                                             app_name="experiment_on_prompt_optimization")
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
                contrastive_context: str = "") -> Tuple[str, Dict]:
        annotations_text = ""
        for i, (name, result) in enumerate(zip(annotator_names, annotation_results)):
            label = extract_label_from_annotation(result, self.task_type)
            annotations_text += f"\n=== {name} (label: {label}) ===\n{result}\n"

        is_three_way_tie = vote_info.get("is_three_way_tie", False)
        if is_three_way_tie:
            vote_text = "Vote: Three-way tie (1:1:1), no majority"
        else:
            vote_text = f"Majority vote: {vote_info.get('winning_label', 'N/A')} ({vote_info.get('winning_count', 0)}/{vote_info.get('total_votes', 0)})"

        sample_section = ""
        if contrastive_context:
            sample_section = f"\n\nSimilar Sample Reference:\n{contrastive_context}"

        verification_prompt = build_verification_prompt(
            verification_prompt_template=self.verification_prompt_template,
            query_str=query_str,
            annotations_text=annotations_text,
            vote_text=vote_text,
            sample_section=sample_section,
            prompts_dict=PROMPTS
        )

        verification_messages = prompt_to_messages(verification_prompt)
        verification_result = call_llm_tracked(verification_messages, self.model, temperature=0.1,
                                               step_name="verification",
                                               app_name="experiment_on_prompt_optimization")

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
    
    def evaluate_single(self, query_dict: Dict, ground_truth: Any):
        query_str = json.dumps(query_dict, ensure_ascii=False)
        llm_interactions = []
        self._local.interactions = llm_interactions

        agent_injection_packages = {}
        all_agent_names = [f"annotator{i+1}" for i in range(len(self.generated_prompts))] + ["verifier"]

        print(f"[evaluate_single] Retrieve examples → Find dimensions → Direct injection (no LLM judgment):")

        for agent_name in all_agent_names:
            context, correct_samples_list, incorrect_samples_list, sim_scores = self.pool.get_agent_contrastive_context(
                agent_name, query_str, relevant_dimensions=None
            )

            sample_dimensions = []
            if agent_name != "verifier":
                existing_names = set()
                for cs in correct_samples_list:
                    dims = self.pool.find_dimensions_for_sample(
                        cs.get('query', ''),
                        dimension_manager=self.dimension_manager
                    )
                    for d in dims:
                        if d.get('name') not in existing_names:
                            sample_dimensions.append(d)
                            existing_names.add(d.get('name'))
                for ics in incorrect_samples_list:
                    dims = self.pool.find_dimensions_for_sample(
                        ics.get('query', ''),
                        dimension_manager=self.dimension_manager
                    )
                    for d in dims:
                        if d.get('name') not in existing_names:
                            sample_dimensions.append(d)
                            existing_names.add(d.get('name'))

            has_samples = len(correct_samples_list) > 0 or len(incorrect_samples_list) > 0
            should_inject = has_samples
            reason = f"Direct injection: {len(correct_samples_list)} correct + {len(incorrect_samples_list)} incorrect samples" if has_samples else "No samples retrieved"

            agent_injection_packages[agent_name] = {
                'should_inject': should_inject,
                'reason': reason,
                'context': context if should_inject else "",
                'samples': {
                    "correct": correct_samples_list[0] if correct_samples_list else None,
                    "incorrect": incorrect_samples_list[0] if incorrect_samples_list else None,
                    "correct_list": correct_samples_list,
                    "incorrect_list": incorrect_samples_list
                } if should_inject else None,
                'dimensions': sample_dimensions if should_inject else [],
                'sim_scores': sim_scores
            }
        
        annotation_results = [None] * len(self.generated_prompts)
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_idx = {}
            for idx, prompt in enumerate(self.generated_prompts):
                agent_name = f"annotator{idx+1}"
                pkg = agent_injection_packages[agent_name]
                future = executor.submit(
                    self._annotate_single, query_str, prompt, 
                    pkg['context'], pkg['dimensions'], agent_name
                )
                future_to_idx[future] = idx
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                agent_name = f"annotator{idx+1}"
                try:
                    annotation_result = future.result()
                    annotation_results[idx] = annotation_result
                    
                    has_context = agent_injection_packages[agent_name]['should_inject']
                    step_name = f"annotation_{agent_name}_with_context" if has_context else f"annotation_{agent_name}_without_context"
                    self._local.interactions.append({"step": step_name, "prompt": "", "response": annotation_result})
                    print(f"{agent_name}: Annotation success")
                except Exception as e:
                    print(f"{agent_name}: Annotation failed: {e}")
                    annotation_results[idx] = ""
        
        # Hard Majority Vote
        final_result, vote_info = self._hard_majority_vote(annotation_results)
        self._local.interactions.append({
            "step": "aggregation", "prompt": "",
            "response": f"Hard majority vote result: {vote_info['winning_label']}",
            "vote_info": vote_info
        })

        # VerificationAgent
        verification_info = None
        was_verified = False

        if not vote_info.get("is_unanimous", False):
            print(f"  [Verification] Calling VerificationAgent...")
            verifier_pkg = agent_injection_packages["verifier"]
            verifier_context = verifier_pkg['context'] if verifier_pkg['should_inject'] else ""
            annotator_names = [f"annotator{i+1}" for i in range(len(self.generated_prompts))]
            final_result, verification_info = self._verify(
                query_str=query_str,
                annotation_results=annotation_results,
                annotator_names=annotator_names,
                vote_info=vote_info,
                contrastive_context=verifier_context
            )
            was_verified = verification_info.get("was_verified", False)
            selected_agent = verification_info.get("selected_agent", "unknown")
            self._local.interactions.append({
                "step": "verification", "prompt": "",
                "response": f"Verification selected: {selected_agent}, reason: {verification_info.get('reason', [])}"
            })
            print(f"  [Verification] Success, selected: {selected_agent}")


        multi_annotation_metrics = None
        if isinstance(ground_truth, dict) and "all_annotations" in ground_truth:
            multi_annotation_metrics = self._evaluate_multi_annotation_detailed(final_result, ground_truth)
            is_correct = multi_annotation_metrics["is_correct_hit2"]
        else:
            is_correct = self.eval_func(final_result, ground_truth)

        llm_interactions = self._local.interactions.copy()
        if hasattr(self._local, 'interactions'):
            del self._local.interactions

        agent_inject_decisions = {k: v['should_inject'] for k, v in agent_injection_packages.items()}
        agent_injected_samples = {k: v['samples'] for k, v in agent_injection_packages.items()}
        all_dim_names = set()
        for pkg in agent_injection_packages.values():
            for d in pkg.get('dimensions', []):
                all_dim_names.add(d.get('name', ''))

        result = {
            "query": query_str,
            "annotation": final_result,
            "is_correct": is_correct,
            "llm_interactions": llm_interactions,
            "annotation_results": annotation_results,
            "ground_truth": ground_truth,
            "context_injected": any(agent_inject_decisions.values()),
            "per_agent_context_injected": agent_inject_decisions,
            "agent_injected_samples": agent_injected_samples,
            "relevant_dimensions": list(all_dim_names),
            "vote_info": vote_info,
            "was_verified": was_verified,
            "verification_info": verification_info,
        }
        if multi_annotation_metrics:
            result["multi_annotation_metrics"] = multi_annotation_metrics
        return result
    
    def evaluate_batch(self, query_dicts: List[Dict], ground_truths: List[str]):
        results = [None] * len(query_dicts)
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_index = {
                executor.submit(self.evaluate_single, query_dicts[idx], ground_truths[idx]): idx
                for idx in range(len(query_dicts))
            }
            
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results[idx] = result
                    print(f"ID {idx} annotation complete")
                except Exception as e:
                    print(f"\nID {idx} failed: {e}")
                    traceback.print_exc()
                    results[idx] = {
                        "error": str(e),
                        "query": json.dumps(query_dicts[idx], ensure_ascii=False),
                        "ground_truth": ground_truths[idx],
                        "is_correct": False,
                        "llm_interactions": [],
                        "annotation": "",
                        "annotation_results": [],
                        "context_injected": False,
                        "per_agent_context_injected": {},
                        "vote_info": {},
                        "was_verified": False,
                        "verification_info": None,
                    }
        
        return results


def find_best_epoch(task_type: str, output_file_prefix: str) -> Tuple[int, Dict, float]:
    search_pattern = f"output/{task_type}/{task_type}_multi_agents_prompt_optimization_with_{output_file_prefix}_epoch*.json"
    epoch_files = glob.glob(search_pattern)
    
    if not epoch_files:
        print(f"No epoch file found: {search_pattern}")
        return None, None, 0.0
    
    best_epoch = -1
    best_accuracy = 0.0
    best_data = None
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
            
            epoch_results.append((epoch, accuracy, file_path, data))
            
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                best_data = data
                
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    if epoch_results:
        print(f"\nFound {len(epoch_results)} epoch results:")
        for epoch, accuracy, file_path, _ in sorted(epoch_results):
            marker = " <-- Best" if epoch == best_epoch else ""
            print(f"  Epoch {epoch}: {accuracy:.4f} ({accuracy*100:.2f}%){marker}")
    
    if best_data is not None:
        print(f"\nSelected Epoch {best_epoch} (accuracy: {best_accuracy:.4f})")
    
    return best_epoch, best_data, best_accuracy


def run_evaluation():
    global TASK_TYPE
    start_id = 300
    end_id = 1300
    
    task_config = get_task_config(TASK_TYPE, end_id)
    TASK_TYPE = TASK_TYPE.replace("multi_","")
    evaluate_func = EVALUATION_FUNCTIONS.get(TASK_TYPE)
    
    evaluator = Annotator(
        model="DeepSeek-V3.1-Terminus",
        api_key="",
        base_url="",
        task_type=TASK_TYPE,
        category_doc_path=task_config["requirement_doc_path"],
        eval_func=evaluate_func,
        output_file_prefix=OUTPUT_FILE_PREFIX
    )
    
    all_queries = task_config["queries"][start_id:end_id]
    all_ground_truths = task_config["answers"][start_id:end_id]
    evaluator.pool.initialize_from_dataset(all_queries, all_ground_truths)
    evaluator.pool.load_historical_badcases()
    
    queries = task_config["queries"][start_id:end_id]
    ground_truths = task_config["answers"][start_id:end_id]
    
    if use_best_epoch:
        best_epoch, best_data, best_accuracy = find_best_epoch(TASK_TYPE, OUTPUT_FILE_PREFIX)
        
        if best_data is not None:
            if "current_generated_prompts" in best_data:
                evaluator.set_generated_prompts(best_data["current_generated_prompts"])
                print(f"Loaded best epoch {best_epoch} prompts ({len(best_data['current_generated_prompts'])} prompts)")
            
            if "current_verification_prompt_template" in best_data:
                evaluator.set_verification_prompt_template(best_data["current_verification_prompt_template"])
                print("Loaded best epoch verification prompt template")
            elif "current_judge_rules" in best_data:
                evaluator.set_verification_prompt_template(best_data["current_judge_rules"])
                print("Loaded best epoch judge rules as verification prompt template (legacy)")

            if "intent_hypotheses" in best_data:
                evaluator.dimension_manager.load_from_checkpoint(best_data)
            
            if "pool_state" in best_data:
                evaluator.pool.load_from_checkpoint(best_data)
            
            output_file = f"output/{TASK_TYPE}/{TASK_TYPE}_evaluation_multi_agents_{OUTPUT_FILE_PREFIX}_best_epoch{best_epoch}_v0406_sample_injection{CONTEXT_INJECTION_TOP_K}.json"
        else:
            print("""
Best epoch data not found, will use default method""")
    
    if not use_best_epoch and use_optimized_prompt:
        optimized_prompt_path = f"output/{TASK_TYPE}/{TASK_TYPE}_multi_agents_prompt_optimization_with_{OUTPUT_FILE_PREFIX}_epoch9.json"
        
        if os.path.exists(optimized_prompt_path):
            with open(optimized_prompt_path, "r", encoding='utf-8') as f:
                optimization_data = json.load(f)
            
            if "current_generated_prompts" in optimization_data:
                evaluator.set_generated_prompts(optimization_data["current_generated_prompts"])
                print(f"Loaded optimized prompts ({len(optimization_data['current_generated_prompts'])} prompts)")
            
            if "current_verification_prompt_template" in optimization_data:
                evaluator.set_verification_prompt_template(optimization_data["current_verification_prompt_template"])
                print("Loaded optimized verification prompt template")
            elif "current_judge_rules" in optimization_data:
                evaluator.set_verification_prompt_template(optimization_data["current_judge_rules"])
                print("Loaded optimized judge rules as verification prompt template (legacy)")
            
            if "intent_hypotheses" in optimization_data:
                evaluator.dimension_manager.load_from_checkpoint(optimization_data)
            
            if "pool_state" in optimization_data:
                evaluator.pool.load_from_checkpoint(optimization_data)
            
            output_file = f"output/{TASK_TYPE}/{TASK_TYPE}_evaluation_multi_agents_{OUTPUT_FILE_PREFIX}_optimized_v0406_sample_injection{CONTEXT_INJECTION_TOP_K}.json"
        else:
            print(f"Warning: Optimized prompt file not found: {optimized_prompt_path}")
    
    if not use_best_epoch and not use_optimized_prompt:
        init_prompt_path = f"output/{TASK_TYPE}/{TASK_TYPE}_init_prompts.json"
        if os.path.exists(init_prompt_path):
            with open(init_prompt_path, "r", encoding='utf-8') as f:
                init_data = json.load(f)
            evaluator.set_generated_prompts(init_data["generated_prompts"])
            print(f"Loaded initial prompts")
        else:
            print("Warning: No prompts found, please provide optimized or initial prompts")
            return
        
        output_file = f"output/{TASK_TYPE}/{TASK_TYPE}_evaluation_multi_agents_{OUTPUT_FILE_PREFIX}_v0406_sample_injection{CONTEXT_INJECTION_TOP_K}.json"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\nStarting evaluation of {len(queries)} samples...")
    results = evaluator.evaluate_batch(queries, ground_truths)
    
    for idx, result in enumerate(results):
        result['query'] = json.dumps(queries[idx], ensure_ascii=False)
        result['ground_truth'] = ground_truths[idx]
    
    num_correct = 0
    multi_annotation_stats = {
        "total": 0,
        "by_agreement_level": {"consensus": 0, "split": 0, "ambiguous": 0},
        "hit2_correct": {"consensus": 0, "split": 0, "ambiguous": 0},
        "hit1_correct": {"consensus": 0, "split": 0, "ambiguous": 0},
        "majority_correct": {"consensus": 0, "split": 0, "ambiguous": 0}
    }

    for result in results:
        if result.get("is_correct", False):
            num_correct += 1

        if "multi_annotation_metrics" in result:
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

    accuracy = num_correct / len(results) if results else 0

    if multi_annotation_stats["total"] > 0:
        print(f"\n========== Multi-annotation Dataset Detailed Statistics ==========")
        print(f"Total multi-annotation samples: {multi_annotation_stats['total']}")

        print(f"\nDistribution by agreement level:")
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
        print(f"==========================================\n")

    print(f"\nAnnotation accuracy: {num_correct}/{len(results)} = {accuracy:.4f} ({accuracy*100:.2f}%)")

    token_counter.print_summary()

    context_injected_count = sum(1 for r in results if r.get("context_injected", False))
    print(f"Context injection ratio: {context_injected_count}/{len(results)} = {context_injected_count/len(results)*100:.2f}%")
    
    agent_injection_counts = {}
    for r in results:
        if r:
            per_agent = r.get("per_agent_context_injected", {})
            for agent_name, injected in per_agent.items():
                if agent_name not in agent_injection_counts:
                    agent_injection_counts[agent_name] = {"injected": 0, "total": 0}
                agent_injection_counts[agent_name]["total"] += 1
                if injected:
                    agent_injection_counts[agent_name]["injected"] += 1
    if agent_injection_counts:
        agent_stats = ", ".join([
            f"{name}: {counts['injected']}/{counts['total']}"
            for name, counts in sorted(agent_injection_counts.items())
        ])
        print(f"Per-agent injection stats: {agent_stats}")
    
    per_agent_summary = {}
    for name, counts in agent_injection_counts.items():
        per_agent_summary[name] = {
            "injected": counts["injected"],
            "total": counts["total"],
            "percent": f"{counts['injected']/counts['total']*100:.2f}%" if counts["total"] > 0 else "0%"
        }
    
    output_data = {
        "results": results,
        "summary": {
            "total": len(results),
            "correct": num_correct,
            "accuracy": accuracy,
            "accuracy_percent": f"{accuracy*100:.2f}%",
            "context_injected_count": context_injected_count,
            "context_injected_percent": f"{context_injected_count/len(results)*100:.2f}%",
            "per_agent_injection": per_agent_summary
        },
        "config": {
            "use_contrastive_context": USE_CONTRASTIVE_CONTEXT
        },
        "multi_annotation_statistics": multi_annotation_stats if multi_annotation_stats["total"] > 0 else None,
        "token_usage": token_counter.get_summary()
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nDone! Results saved to: {output_file}")
    except Exception as e:
        print(f"\nSave failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-agent Annotation Evaluation Script')
    parser.add_argument('--task_type', type=str, required=True, help='Task type')
    parser.add_argument('--output_file_prefix', type=str, required=True, help='Output file prefix')
    parser.add_argument('--use_best_epoch_prompt', action='store_true', default=False,
                       help='Automatically find and load the best performing epoch')
    parser.add_argument('--use_optimized_prompt', action='store_true', default=False,
                       help='Use optimized prompt (epoch9)')
    parser.add_argument('--use_contrastive_context', action='store_true', default=True,
                       help='Enable contrastive example context injection')
    
    args = parser.parse_args()
    
    TASK_TYPE = args.task_type
    OUTPUT_FILE_PREFIX = args.output_file_prefix
    use_best_epoch = args.use_best_epoch_prompt
    use_optimized_prompt = args.use_optimized_prompt
    USE_CONTRASTIVE_CONTEXT = args.use_contrastive_context
    
    run_evaluation()
