import openai
import json
import re
import numpy as np
import pandas as pd
import sys
import os
import requests
import traceback
import time
import copy
import hashlib
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple, Optional, Callable

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.evaluation_functions import EVALUATION_FUNCTIONS, extract_label_from_annotation
from collections import Counter
from utils.prompts import PROMPTS_EN as PROMPTS, OUTPUT_FORMATS
from utils.prompts import build_dimension_sample_prompt, build_verification_prompt
from utils.load_task import get_task_config
from utils.error_reason_analysis import (
    identify_all_error_agents,
    identify_errors_with_contrastive_analysis_en as identify_errors_with_contrastive_analysis,
    find_contrastive_pairs_for_error
)
from utils.call_llm_api import call_llm_general
from utils.samples_retriever import (
    get_or_create_embedding,
    cosine_similarity
)
from utils.token_counter import TokenCounter
import threading

token_counter = TokenCounter()


def call_llm_tracked(messages, model, temperature=0.1, stream=False, step_name="unknown", app_name="experiment_on_prompt_optimization"):
    result = call_llm_general(messages, model, temperature=temperature, stream=stream, app_name=app_name)
    token_counter.record(messages, result, step_name=step_name)
    return result


TASK_TYPE = sys.argv[1]
OUTPUT_FILE_PREFIX = sys.argv[2]

RESUME_EPOCH = None
if len(sys.argv) > 3:
    for i, arg in enumerate(sys.argv[3:]):
        if arg == "--resume-epoch" and i + 4 < len(sys.argv):
            try:
                RESUME_EPOCH = int(sys.argv[i + 4])
                print(f"Resuming training from epoch {RESUME_EPOCH}")
            except ValueError:
                print(f"Warning: Invalid --resume-epoch value, starting training from scratch")

OPTIMIZE_REWARD = 0.1
OPTIMIZE_PUNISH = 0.1
OPTIMIZE_BETTER = 0
OPTIMIZE_WORSE = 0

CONTEXT_INJECTION_TOP_K = 2     
POOL_REFRESH_BATCH_SIZE = 50      


def _normalize_gt_label(ground_truth: Any) -> str:
    if isinstance(ground_truth, dict):
        majority = ground_truth.get("majority_answer")
        if majority is not None:
            return str(majority).strip()
        vote_dist = ground_truth.get("vote_distribution_detail", [])
        if vote_dist:
            return str(vote_dist[0].get("answer", "")).strip()
        all_annos = ground_truth.get("all_annotations", [])
        if all_annos:
            return str(all_annos[0]).strip()
        return str(ground_truth).strip()
    elif isinstance(ground_truth, list):
        if ground_truth:
            return str(ground_truth[0]).strip()
        return ""
    else:
        return str(ground_truth).strip()

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
    
    def update_score(self, query_str: str, was_helpful: bool, 
                    annotation_changed: bool = False):
        if query_str not in self.sample_scores:
            self.sample_scores[query_str] = {
                'score': 1.0,
                'correct_count': 0,
                'incorrect_count': 0,
                'usage_count': 0,
                'last_used': time.time()
            }
        
        record = self.sample_scores[query_str]
        record['usage_count'] += 1
        record['last_used'] = time.time()

        global_correct_rate = self.get_global_correct_rate()

        scarcity_factor = 1.0 - global_correct_rate

        if was_helpful:
            adjusted_step = self.step_size * scarcity_factor
            record['score'] = min(2.0, record['score'] + adjusted_step)
            record['correct_count'] += 1
            self.update_global_stats(True)
        else:
            penalty_factor = 1.0 - scarcity_factor 
            adjusted_step = self.step_size * penalty_factor
            record['score'] = max(0.0, record['score'] - adjusted_step)
            record['incorrect_count'] += 1
            self.update_global_stats(False)

        self.usage_history.append({
            'query': query_str,
            'was_helpful': was_helpful,
            'annotation_changed': annotation_changed,
            'timestamp': time.time(),
            'new_score': record['score'],
            'global_correct_rate': global_correct_rate,
            'scarcity_factor': scarcity_factor
        })
        
        if len(self.usage_history) > 10000:
            self.usage_history = self.usage_history[-5000:]
    
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
    def __init__(self, strategy: str = "softmax", temperature: float = 1.0):
        self.strategy = strategy
        self.temperature = temperature
        self.selection_counts = {}
    
    def select(self, candidates: List[Tuple[Dict, float]], 
               use_sampling: bool = True) -> Tuple[Optional[Dict], float]:
        if not candidates:
            return None, 0.0
        
        if len(candidates) == 1 or not use_sampling:
            best = max(candidates, key=lambda x: x[1])
            self._update_selection_count(best[0])
            return best
        
        if self.strategy == "softmax":
            return self._softmax_sample(candidates)
        elif self.strategy == "ucb":
            return self._ucb_sample(candidates)
        else:
            best = max(candidates, key=lambda x: x[1])
            self._update_selection_count(best[0])
            return best
    
    def _softmax_sample(self, candidates: List[Tuple[Dict, float]]) -> Tuple[Optional[Dict], float]:
        scores = np.array([c[1] for c in candidates])
        
        scores = scores - np.max(scores)
        
        exp_scores = np.exp(scores / self.temperature)
        probs = exp_scores / np.sum(exp_scores)
        
        selected_idx = np.random.choice(len(candidates), p=probs)
        selected = candidates[selected_idx]
        
        self._update_selection_count(selected[0])
        return selected
    
    def _ucb_sample(self, candidates: List[Tuple[Dict, float]], 
                    exploration_factor: float = 2.0) -> Tuple[Optional[Dict], float]:
        total_selections = sum(self.selection_counts.values()) + 1
        
        ucb_scores = []
        for sample, score in candidates:
            query_str = sample.get('query', '')
            count = self.selection_counts.get(query_str, 0) + 1
            ucb = score + exploration_factor * np.sqrt(2 * np.log(total_selections) / count)
            ucb_scores.append(ucb)
        
        best_idx = np.argmax(ucb_scores)
        selected = candidates[best_idx]
        
        self._update_selection_count(selected[0])
        return selected
    
    def _update_selection_count(self, sample: Dict):
        query_str = sample.get('query', '')
        if query_str:
            self.selection_counts[query_str] = self.selection_counts.get(query_str, 0) + 1
    
    def set_temperature(self, temperature: float):
        self.temperature = max(0.1, temperature)
    
    
    def get_exploration_stats(self) -> Dict:
        if not self.selection_counts:
            return {'total_selections': 0, 'unique_samples': 0}
        
        counts = list(self.selection_counts.values())
        return {
            'total_selections': sum(counts),
            'unique_samples': len(counts),
            'avg_selections_per_sample': np.mean(counts),
            'max_selections': max(counts),
            'min_selections': min(counts),
            'current_temperature': self.temperature
        }
    
    def reset_counts(self):
        self.selection_counts = {}


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
    
    def __init__(self, task_type: str, output_file_prefix: str, mode: str = "train"):
        self.task_type = task_type
        self.output_file_prefix = output_file_prefix
        self.mode = mode 
        self.all_samples = [] 
        self.correct_samples = [] 
        self.incorrect_samples = [] 
        self.historical_badcases = [] 
        self._evaluated_flags = {} 

        self.agent_names = ["annotator1", "annotator2", "annotator3", "verifier"]
        self.agent_correct_samples = {
            agent: [] for agent in self.agent_names
        }  # agent -> list of samples where this agent is correct
        self.agent_incorrect_samples = {
            agent: [] for agent in self.agent_names
        }  # agent -> list of samples where this agent is incorrect
        self._agent_evaluated_flags = {
            agent: {} for agent in self.agent_names
        }  # agent -> {query -> is_correct}

        self.sample_scorer = SampleQualityScorer()
        self.sample_selector = SampleSelector(strategy="softmax", temperature=1.0)
        self._injected_samples_cache = {}  # query_str -> {correct_sample, incorrect_sample, timestamp}
        # train=True, test=False
        self.use_sampling = (mode == "train")
        
        self.dimension_sample_index = {}  # dimension_name -> {query_str -> sample_info}
        self.sample_dimension_tags = {}   # query_str -> [dimension_names]
    
    def set_mode(self, mode: str):
        self.mode = mode
        self.use_sampling = (mode == "train")
        print(f"[ContrastiveExamplePool] Switched to {mode} mode, sampling={'on' if self.use_sampling else 'off'}")

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
    
    def refresh(self, annotator, batch_size: int = POOL_REFRESH_BATCH_SIZE):
        print(f"\n[Pool] Starting to refresh sample pool (total {len(self.all_samples)} samples) ...")
        self.correct_samples = []
        self.incorrect_samples = []
        self._evaluated_flags = {}
        
        for agent in self.agent_names:
            self.agent_correct_samples[agent] = []
            self.agent_incorrect_samples[agent] = []
            self._agent_evaluated_flags[agent] = {}
        
        for i in range(0, len(self.all_samples), batch_size):
            batch = self.all_samples[i:i+batch_size]
            batch_queries = [s["query_dict"] for s in batch]
            batch_ground_truths = [s["ground_truth"] for s in batch]
            
            results = annotator.evaluate_batch(batch_queries, batch_ground_truths)
            
            for sample, result in zip(batch, results):
                query_str = sample["query"]
                is_correct = result.get("is_correct", False)
                self._evaluated_flags[query_str] = is_correct
                representative_gt = self._extract_representative_gt(sample["ground_truth"])

                if is_correct:
                    self.correct_samples.append({
                        "query": query_str,
                        "ground_truth": representative_gt,
                        "annotation": result.get("annotation", "")
                    })
                else:
                    self.incorrect_samples.append({
                        "query": query_str,
                        "ground_truth": representative_gt,
                        "annotation": result.get("annotation", "")
                    })

                agent_correctness = result.get("agent_correctness", {})
                if agent_correctness:
                    agent_results = {}
                    for agent_name, info in agent_correctness.items():
                        agent_results[agent_name] = {
                            "is_correct": info.get("is_correct", False),
                            "annotation": info.get("annotation", ""),
                            "ground_truth": representative_gt
                        }
                    self.update_agent_evaluation(query_str, agent_results)
            
            print(f"[Pool] Refreshed {min(i+batch_size, len(self.all_samples))}/{len(self.all_samples)} samples, "
                  f"current correct: {len(self.correct_samples)}, incorrect: {len(self.incorrect_samples)}")
        
        agent_stats = ", ".join([f"{agent}: {len(self.agent_correct_samples[agent])} correct/{len(self.agent_incorrect_samples[agent])} incorrect" 
                                  for agent in self.agent_names])
        print(f"[Pool] Agent-level stats: {agent_stats}")
        
        print(f"[Pool] Refresh complete. Correct samples: {len(self.correct_samples)}, incorrect samples: {len(self.incorrect_samples)}")
    
    def get_contrastive_context(self, query_str: str, relevant_dimensions: List[Dict] = None) -> Tuple[str, List[Dict], List[Dict], Dict[str, float]]:
        return self.get_agent_contrastive_context("system", query_str, relevant_dimensions)
    
    def get_agent_contrastive_context(self, agent_name: str, query_str: str,
                                       relevant_dimensions: List[Dict] = None) -> Tuple[str, List[Dict], List[Dict], Dict[str, float]]:
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

        cache_key = query_str if is_system else f"{agent_name}:{query_str}"
        self._injected_samples_cache[cache_key] = {
            'correct_sample': correct_samples_list[0] if correct_samples_list else None,
            'incorrect_sample': incorrect_samples_list[0] if incorrect_samples_list else None,
            'correct_samples_list': correct_samples_list,
            'incorrect_samples_list': incorrect_samples_list,
            'timestamp': time.time(),
            'agent_name': agent_name if not is_system else None
        }

        context_str = self._format_contrastive_context_multi(
            correct_samples_list, incorrect_samples_list, similarity_scores, relevant_dimensions
        )

        return context_str, correct_samples_list, incorrect_samples_list, similarity_scores

    def _retrieve_top_k_matches_with_score(self, query_str: str, candidates: List[Dict],
                                            top_k: int = 3,
                                            relevant_dimensions: List[Dict] = None,
                                            exclude_query: str = None) -> List[Tuple[Dict, float, float]]:
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

            weighted_score = self.sample_scorer.compute_weighted_similarity(
                cand_query, avg_similarity
            )

            scored_candidates.append((cand, avg_similarity, weighted_score))

        if not scored_candidates:
            return []

        if self.use_sampling and len(scored_candidates) > top_k:
            weights = np.array([w for _, _, w in scored_candidates])
            weights = weights - np.max(weights)
            probs = np.exp(weights / self.sample_selector.temperature)
            probs = probs / np.sum(probs)
            selected_indices = np.random.choice(
                len(scored_candidates), size=min(top_k, len(scored_candidates)),
                replace=False, p=probs
            )
            result = [scored_candidates[i] for i in selected_indices]
            result.sort(key=lambda x: x[2], reverse=True)
            return result
        else:
            scored_candidates.sort(key=lambda x: x[2], reverse=True)
            return scored_candidates[:top_k]

    def _format_contrastive_context_multi(self, correct_samples: List[Dict],
                                           incorrect_samples: List[Dict],
                                           similarity_scores: Dict = None,
                                           relevant_dimensions: List[Dict] = None) -> str:
        if similarity_scores is None:
            similarity_scores = {}

        context_parts = []

        dim_explanation = ""
        if relevant_dimensions:
            dim_names = [d.get('name', '') for d in relevant_dimensions[:3] if d.get('name')]
            if dim_names:
                dim_explanation = f" (Associated dimensions: {', '.join(dim_names)})"

        # Format correct samples
        correct_sims = similarity_scores.get('correct_all', [])
        for i, sample in enumerate(correct_samples):
            sim_score = correct_sims[i] if i < len(correct_sims) else 0.0
            quality_score = self.sample_scorer.get_score(sample.get('query', ''))
            quality_tag = " [High Quality]" if quality_score > 0.7 else " [Medium Quality]" if quality_score > 0.4 else " [Low Quality]"
            gt_display = self._extract_representative_gt(sample.get("ground_truth", ""))
            correct_section = PROMPTS["contrastive_sample_correct"].format(
                idx=i+1,
                dim_explanation=dim_explanation,
                quality_tag=quality_tag,
                sim_text=f" (similarity: {sim_score:.3f})",
                image_placeholder="",
                query=sample.get("query", ""),
                annotation=sample.get("annotation", ""),
                ground_truth=gt_display
            )
            context_parts.append(correct_section)

        # Format incorrect samples
        incorrect_sims = similarity_scores.get('incorrect_all', [])
        for i, sample in enumerate(incorrect_samples):
            sim_score = incorrect_sims[i] if i < len(incorrect_sims) else 0.0
            quality_score = self.sample_scorer.get_score(sample.get('query', ''))
            quality_tag = " [High Quality Warning]" if quality_score > 0.7 else " [Medium Quality]" if quality_score > 0.4 else " [Low Quality]"
            gt_display = self._extract_representative_gt(sample.get("ground_truth", ""))
            error_section = PROMPTS["contrastive_sample_incorrect"].format(
                idx=i+1,
                dim_explanation=dim_explanation,
                quality_tag=quality_tag,
                sim_text=f" (similarity: {sim_score:.3f})",
                image_placeholder="",
                query=sample.get("query", ""),
                annotation=sample.get("annotation", sample.get("error_output", "")),
                ground_truth=gt_display,
                error_reason=sample.get("error_reason", "Unknown")
            )
            context_parts.append(error_section)

        # Add contrastive analysis guidance
        if correct_samples and incorrect_samples:
            context_parts.append(PROMPTS["contrastive_analysis_guidance"].format(
                num_correct=len(correct_samples),
                num_incorrect=len(incorrect_samples)
            ))
        elif correct_samples:
            context_parts.append(PROMPTS["contrastive_correct_only_guidance"].format(
                num_correct=len(correct_samples)
            ))
        elif incorrect_samples:
            context_parts.append(PROMPTS["contrastive_incorrect_only_guidance"].format(
                num_incorrect=len(incorrect_samples)
            ))

        return "\n\n".join(context_parts)
    
    
    def get_agent_correct_samples(self, agent_name: str) -> List[Dict]:
        if agent_name not in self.agent_names:
            return []
        return self.agent_correct_samples.get(agent_name, [])
    
    def get_agent_incorrect_samples(self, agent_name: str) -> List[Dict]:
        if agent_name not in self.agent_names:
            return []
        return self.agent_incorrect_samples.get(agent_name, [])
    
    def is_agent_correct_on_query(self, agent_name: str, query_str: str) -> Optional[bool]:
        if agent_name not in self.agent_names:
            return None
        return self._agent_evaluated_flags.get(agent_name, {}).get(query_str)
    
    def update_agent_evaluation(self, query_str: str, agent_results: Dict[str, Dict]):
        for agent_name, result in agent_results.items():
            if agent_name not in self.agent_names:
                continue
            
            is_correct = result.get("is_correct", False)
            self._agent_evaluated_flags[agent_name][query_str] = is_correct
            
            self.agent_correct_samples[agent_name] = [
                s for s in self.agent_correct_samples[agent_name] 
                if s.get("query") != query_str
            ]
            self.agent_incorrect_samples[agent_name] = [
                s for s in self.agent_incorrect_samples[agent_name] 
                if s.get("query") != query_str
            ]
            
            sample = {
                "query": query_str,
                "annotation": result.get("annotation", ""),
                "ground_truth": self._extract_representative_gt(result.get("ground_truth", ""))
            }
            
            if is_correct:
                self.agent_correct_samples[agent_name].append(sample)
            else:
                self.agent_incorrect_samples[agent_name].append(sample)
    
    def update_sample_quality_feedback(self, query_str: str, was_helpful: bool, 
                                       agent_name: str = None,
                                       annotation_changed: bool = False):
        cache_key = f"{agent_name}:{query_str}" if agent_name else query_str
        
        if cache_key not in self._injected_samples_cache:
            if cache_key in self._injected_samples_cache:
                injected_info = self._injected_samples_cache[cache_key]
            else:
                return
        else:
            injected_info = self._injected_samples_cache[cache_key]
        
        correct_sample = injected_info.get('correct_sample')
        if correct_sample:
            sample_query = correct_sample.get('query', '')
            if sample_query:
                self.sample_scorer.update_score(
                    query_str=sample_query,
                    was_helpful=was_helpful,
                    annotation_changed=annotation_changed
                )
                new_score = self.sample_scorer.get_score(sample_query)
                print(f"  [Sample Quality Update] Correct sample score updated: {new_score:.3f} (helpful={was_helpful})")
        
        incorrect_sample = injected_info.get('incorrect_sample')
        if incorrect_sample:
            sample_query = incorrect_sample.get('query', '')
            if sample_query:
                self.sample_scorer.update_score(
                    query_str=sample_query,
                    was_helpful=was_helpful,
                    annotation_changed=annotation_changed
                )
                new_score = self.sample_scorer.get_score(sample_query)
                print(f"  [Sample Quality Update] Incorrect sample score updated: {new_score:.3f} (helpful={was_helpful})")
        
        if len(self._injected_samples_cache) > 200:
            sorted_cache = sorted(
                self._injected_samples_cache.items(),
                key=lambda x: x[1].get('timestamp', 0),
                reverse=True
            )
            self._injected_samples_cache = dict(sorted_cache[:100])
    
    def get_sample_quality_stats(self) -> Dict:
        return self.sample_scorer.get_sample_stats()
    
    def update_from_batch_results(self, batch_queries: List[Dict], batch_results: List[Dict]):
        for q_dict, result in zip(batch_queries, batch_results):
            query_str = json.dumps(q_dict, ensure_ascii=False)
            is_correct = result.get("is_correct", False)
            representative_gt = self._extract_representative_gt(result.get("ground_truth", ""))

            self.correct_samples = [s for s in self.correct_samples if s["query"] != query_str]
            self.incorrect_samples = [s for s in self.incorrect_samples if s["query"] != query_str]

            if is_correct:
                self.correct_samples.append({
                    "query": query_str,
                    "ground_truth": representative_gt,
                    "annotation": result.get("annotation", "")
                })
            else:
                self.incorrect_samples.append({
                    "query": query_str,
                    "ground_truth": representative_gt,
                    "annotation": result.get("annotation", "")
                })

            self._evaluated_flags[query_str] = is_correct

            agent_correctness = result.get("agent_correctness", {})
            if agent_correctness:
                agent_results = {}
                for agent_name, info in agent_correctness.items():
                    agent_results[agent_name] = {
                        "is_correct": info.get("is_correct", False),
                        "annotation": info.get("annotation", ""),
                        "ground_truth": representative_gt
                    }
                self.update_agent_evaluation(query_str, agent_results)
    
    
    def tag_sample_with_dimensions(self, sample_query: str, dimensions: List[Dict], 
                                   is_correct: bool = True, agent_name: str = None):
        if not dimensions:
            return
        
        if sample_query not in self.sample_dimension_tags:
            self.sample_dimension_tags[sample_query] = []
        
        for dim in dimensions:
            dim_name = dim.get('name', '')
            if not dim_name:
                continue
            
            if dim_name not in self.sample_dimension_tags[sample_query]:
                self.sample_dimension_tags[sample_query].append(dim_name)
            
            if dim_name not in self.dimension_sample_index:
                self.dimension_sample_index[dim_name] = {}
            
            sample_info = {
                'query': sample_query,
                'is_correct': is_correct,
                'agent_name': agent_name,
                'dimensions': [d.get('name', '') for d in dimensions]
            }
            self.dimension_sample_index[dim_name][sample_query] = sample_info
    
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


# =============================================================================
# =============================================================================

class DimensionManager:
    
    def __init__(self, model: str, doc_content: str, task_type: str, output_file_prefix: str, pool=None):
        self.model = model
        self.doc_content = doc_content
        self.task_type = task_type
        self.output_file_prefix = output_file_prefix
        self.pool = pool
        self.intent_hypotheses = {}
        self._similarity_history = []
        self._history_max_size = 500
        self._load_intent_hypotheses()

        for dim in self.intent_hypotheses.values():
            dim.setdefault("success_rate", 0.5)
            dim.setdefault("total_injections", 0)
    
    def _save_intent_hypotheses(self):
        path = f"cases/{self.task_type}/intent_hypotheses_{self.output_file_prefix}.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.intent_hypotheses, f, ensure_ascii=False, indent=2)
    
    def _load_intent_hypotheses(self):
        path = f"cases/{self.task_type}/intent_hypotheses_{self.output_file_prefix}.json"
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.intent_hypotheses = json.load(f)
            print(f"[DimensionManager] Loaded {len(self.intent_hypotheses)} historical dimensions")
    
    
    def _get_dimension_embedding_text(self, dim: Dict) -> str:
        parts = [
            dim.get("meta_rule", ""),
            dim.get("rule_intent", ""),
            dim.get("decision_boundary", ""),
            dim.get("applicable_boundary", "")
        ]
        return " ".join([p for p in parts if p])
    
    def update_dimension_outcomes(self, dimension_names: List[str], is_correct: bool):
        for dim_name in dimension_names:
            if dim_name not in self.intent_hypotheses:
                continue
            dim = self.intent_hypotheses[dim_name]
            total = dim.get("total_injections", 0) + 1
            current_sr = dim.get("success_rate", 0.5)
            new_sr = (current_sr * (total - 1) + (1.0 if is_correct else 0.0)) / total
            dim["success_rate"] = new_sr
            dim["total_injections"] = total
    
    def _update_similarity_history(self, top_score: float):
        self._similarity_history.append(top_score)
        if len(self._similarity_history) > self._history_max_size:
            self._similarity_history.pop(0)
    
    def mine_contrastive_pairs(self, evaluate_results: List[Dict], threshold: float = 0.2,
                                correct_samples_pool: List[Dict] = None, top_k_per_error: int = 2,
                                agent_correct_samples: Dict[str, List[Dict]] = None):
        if correct_samples_pool is not None:
            correct = correct_samples_pool
        else:
            correct = [r for r in evaluate_results if r.get('is_correct', False)]

        errors = [r for r in evaluate_results if not r.get('is_correct', False)]

        if not correct or not errors:
            return []

        triples = []

        for err in errors:
            err_query = err.get('query', '')
            err_embedding = get_or_create_embedding(err_query)
            err_gt_label = _normalize_gt_label(err.get('ground_truth', ''))

            error_agents = err.get('error_agents', [])
            agent_specific_correct = None

            if agent_correct_samples and error_agents:
                agent_specific_correct = []
                seen_queries = set()
                for agent_info in error_agents:
                    agent_name = agent_info.get('智能体名称', agent_info.get('agent_name', '')).lower().strip()
                    if agent_name in agent_correct_samples:
                        for sample in agent_correct_samples.get(agent_name, []):
                            q = sample.get('query', '')
                            if q and q not in seen_queries:
                                seen_queries.add(q)
                                agent_specific_correct.append(sample)

                if agent_specific_correct:
                    print(f"  [mine_contrastive_pairs] Error sample '{err_query[:30]}...' (GT={err_gt_label}) "
                          f"using {len(agent_specific_correct)} agent-specific correct samples")

            search_pool = agent_specific_correct if agent_specific_correct else correct

            candidates = []
            for corr in search_pool:
                corr_query = corr.get('query', '')
                if corr_query == err_query:
                    continue
                corr_embedding = get_or_create_embedding(corr_query)
                if corr_embedding == 0:
                    continue

                similarity = cosine_similarity(err_embedding, corr_embedding)
                if similarity > threshold and similarity < 0.99:
                    corr_gt = _normalize_gt_label(corr.get('ground_truth', ''))
                    candidates.append((corr, similarity, corr_gt))

            if len(candidates) < top_k_per_error and search_pool is not correct:
                fallback_threshold = max(threshold - 0.1, 0.4)
                existing_queries = {c[0].get('query', '') for c in candidates}
                for corr in correct:
                    corr_query = corr.get('query', '')
                    if corr_query == err_query:
                        continue
                    if corr_query in existing_queries:
                        continue
                    corr_embedding = get_or_create_embedding(corr_query)
                    if corr_embedding == 0:
                        continue
                    similarity = cosine_similarity(err_embedding, corr_embedding)
                    if similarity > fallback_threshold and similarity < 0.99:
                        corr_gt = _normalize_gt_label(corr.get('ground_truth', ''))
                        candidates.append((corr, similarity, corr_gt))

            if len(candidates) < 1:
                continue

            candidates.sort(key=lambda x: x[1], reverse=True)

            if len(candidates) >= 2:
                corr1, sim1, gt1 = candidates[0]
                corr2, sim2, gt2 = candidates[1]
                avg_sim = (sim1 + sim2) / 2
                triples.append((corr1, corr2, err, avg_sim))
                print(f"  Error sample '{err_query[:50]}...' (GT={err_gt_label}) "
                      f"triple: corr1_GT={gt1}, corr2_GT={gt2} (sim={sim1:.3f}, {sim2:.3f})")

            elif len(candidates) == 1:
                corr1, sim1, gt1 = candidates[0]
                triples.append((corr1, None, err, sim1))
                print(f"  Error sample '{err_query[:50]}...' (GT={err_gt_label}) "
                      f"only found 1 candidate, building pair (sim={sim1:.3f})")

        triples.sort(key=lambda x: x[3], reverse=True)
        print(f"Contrastive intent mining complete: found {len(triples)} contrastive groups (from {len(errors)} error samples)")
        return triples
    
    def _analyze_single_triple(self, triple: Tuple, existing_dims_text: str) -> Optional[Dict]:
        corr1, corr2, err, avg_sim = triple

        case_text = f"\n[Contrastive Group]\n"
        case_text += f"Sample A (Correctly judged):\n"
        case_text += f"  Query: {corr1.get('query', '')}\n"
        case_text += f"  Model output: {corr1.get('annotation', '')}\n"
        case_text += f"  Correct answer: {self.pool._extract_representative_gt(corr1.get('ground_truth', ''))}\n"

        if corr2 is not None:
            case_text += f"\nSample B (Correctly judged):\n"
            case_text += f"  Query: {corr2.get('query', '')}\n"
            case_text += f"  Model output: {corr2.get('annotation', '')}\n"
            case_text += f"  Correct answer: {self.pool._extract_representative_gt(corr2.get('ground_truth', ''))}\n"

        case_text += f"\nSample C (Incorrectly judged):\n"
        case_text += f"  Query: {err.get('query', '')}\n"
        case_text += f"  Model output: {err.get('annotation', '')}\n"
        case_text += f"  Correct answer: {self.pool._extract_representative_gt(err.get('ground_truth', ''))}\n"

        prompt = PROMPTS["analyze_contrastive_triple"].format(
            doc_content=self.doc_content,
            existing_dims_text=existing_dims_text,
            case_text=case_text
        )

        messages = prompt_to_messages(prompt)
        response = call_llm_tracked(messages, self.model, temperature=0.1, step_name="mine_contrastive_pairs")

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response

            parsed = json.loads(json_str)
            parsed['_case_text'] = case_text
            return parsed
        except Exception as e:
            print(f"  Failed to analyze single triple: {e}")
            return None
    
    def induct_dimensions(self, triples: List[Tuple]) -> List[Dict]:
        if len(triples) < 1:
            return []
        
        existing_dims_text = self._format_existing_dimensions()
        
        print(f"Starting parallel analysis of {len(triples)} contrastive triples...")
        analysis_results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_triple = {
                executor.submit(self._analyze_single_triple, triple, existing_dims_text): triple
                for triple in triples[:5]
            }

            for future in as_completed(future_to_triple):
                triple = future_to_triple[future]
                try:
                    result = future.result()
                    if result:
                        analysis_results.append((result, triple))
                        corr1, corr2, err, avg_sim = triple
                        print(f"  Triple analysis complete: error='{err.get('query', '')[:30]}...'")
                except Exception as e:
                    print(f"  Failed to analyze triple: {e}")

        print(f"Parallel analysis complete: obtained {len(analysis_results)} valid analysis results")

        merged_count = 0
        created_count = 0

        for analysis, source_triple in analysis_results:
            case_text = analysis.pop('_case_text', '')
            dim_name = analysis.get("meta_rule", "").strip()
            if not dim_name:
                continue

            matched_existing = None
            for existing_name in self.intent_hypotheses.keys():
                if dim_name.lower().replace(" ", "") == existing_name.lower().replace(" ", ""):
                    matched_existing = existing_name
                    break
            
            if matched_existing:
                self.intent_hypotheses[matched_existing]["evidence_count"] =\
                    self.intent_hypotheses[matched_existing].get("evidence_count", 0) + 1
                existing_surface = self.intent_hypotheses[matched_existing].get("surface_difference", "")
                new_surface = analysis.get("surface_difference", "")
                self.intent_hypotheses[matched_existing]["surface_difference"] =\
                    self._merge_dimension_descriptions(existing_surface, new_surface)
                
                existing_fields = {
                    "rule_intent": self.intent_hypotheses[matched_existing].get("rule_intent", ""),
                    "decision_boundary": self.intent_hypotheses[matched_existing].get("decision_boundary", ""),
                    "applicable_boundary": self.intent_hypotheses[matched_existing].get("applicable_boundary", "")
                }
                
                has_existing_data = any(existing_fields.values())
                has_new_data = any([
                    analysis.get("rule_intent", ""),
                    analysis.get("decision_boundary", ""),
                    analysis.get("applicable_boundary", "")
                ])
                
                if has_existing_data and has_new_data:
                    refined_fields = self._refine_dimension_fields(
                        existing_fields,
                        analysis,
                        matched_existing,
                        case_text
                    )
                    self.intent_hypotheses[matched_existing]["rule_intent"] = refined_fields["rule_intent"]
                    self.intent_hypotheses[matched_existing]["decision_boundary"] = refined_fields["decision_boundary"]
                    self.intent_hypotheses[matched_existing]["applicable_boundary"] = refined_fields["applicable_boundary"]
                    print(f"  Dimension '{matched_existing}' fields refined and updated")
                
                merged_count += 1
            else:
                new_dim = {
                    "name": dim_name,
                    "meta_rule": dim_name,
                    "surface_difference": analysis.get("surface_difference", ""),
                    "decision_boundary": analysis.get("decision_boundary", ""),
                    "rule_intent": analysis.get("rule_intent", ""),
                    "applicable_boundary": analysis.get("applicable_boundary", ""),
                    "evidence_count": 1,
                    "tested": False,
                    "success_rate": 0.5,
                }
                self.intent_hypotheses[dim_name] = new_dim
                created_count += 1

            actual_dim_name = matched_existing if matched_existing else dim_name
            dim_data = self.intent_hypotheses.get(actual_dim_name)
            if dim_data and self.pool:
                corr1, corr2, err, avg_sim = source_triple
                if corr1 and corr1.get('query'):
                    self.pool.tag_sample_with_dimensions(
                        sample_query=corr1['query'],
                        dimensions=[dim_data],
                        is_correct=True
                    )
                if corr2 and corr2.get('query'):
                    self.pool.tag_sample_with_dimensions(
                        sample_query=corr2['query'],
                        dimensions=[dim_data],
                        is_correct=True
                    )
                if err and err.get('query'):
                    self.pool.tag_sample_with_dimensions(
                        sample_query=err['query'],
                        dimensions=[dim_data],
                        is_correct=False
                    )

        active_dims_count = sum(1 for d in self.intent_hypotheses.values()
                               if d.get("evidence_count", 0) >= 2)

        print(f"Induction complete: {len(analysis_results)} analysis results, {merged_count} merged, {created_count} new, "
              f"total {len(self.intent_hypotheses)} dimensions ({active_dims_count} meet evidence threshold)")
        
        self._save_intent_hypotheses()
        return list(self.intent_hypotheses.values())
    
    def _refine_dimension_fields(self, existing_fields: Dict, new_analysis: Dict, meta_rule: str, case_text: str) -> Dict:
        existing_intent = existing_fields.get("rule_intent", "")
        existing_boundary = existing_fields.get("decision_boundary", "")
        existing_applicable = existing_fields.get("applicable_boundary", "")
        
        new_intent = new_analysis.get("rule_intent", "")
        new_boundary = new_analysis.get("decision_boundary", "")
        new_applicable = new_analysis.get("applicable_boundary", "")
        
        if not any([existing_intent, existing_boundary, existing_applicable]):
            return {
                "rule_intent": new_intent,
                "decision_boundary": new_boundary,
                "applicable_boundary": new_applicable
            }
        
        prompt = PROMPTS["refine_dimension_fields"].format(
            doc_content=self.doc_content,
            existing_intent=existing_intent or "None yet",
            existing_boundary=existing_boundary or "None yet",
            existing_applicable=existing_applicable or "None yet",
            case_text=case_text,
            new_intent=new_intent or "None yet",
            new_boundary=new_boundary or "None yet",
            new_applicable=new_applicable or "None yet"
        )
        
        messages = prompt_to_messages(prompt)
        response = call_llm_tracked(messages, self.model, temperature=0.1, step_name="induct_dimensions")
        
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
            
            parsed = json.loads(json_str)
            return {
                "rule_intent": parsed.get("rule_intent", existing_intent),
                "decision_boundary": parsed.get("decision_boundary", existing_boundary),
                "applicable_boundary": parsed.get("applicable_boundary", existing_applicable)
            }
        except Exception as e:
            print(f"  Failed to parse refined fields, falling back to simple merge: {e}")
            return {
                "rule_intent": existing_intent,
                "decision_boundary": existing_boundary,
                "applicable_boundary": existing_applicable
            }
    
    def _merge_dimension_descriptions(self, existing_desc: str, new_desc: str) -> str:
        if not existing_desc:
            return new_desc
        if not new_desc:
            return existing_desc
        return existing_desc if len(existing_desc) >= len(new_desc) else new_desc
    
    def _format_existing_dimensions(self) -> str:
        if not self.intent_hypotheses:
            return "No historical discriminative dimensions yet."

        dims_text = ""
        for i, (name, info) in enumerate(self.intent_hypotheses.items(), 1):
            meta_rule = info.get("meta_rule", "")
            rule_intent = info.get("rule_intent", "")
            decision_boundary = info.get("decision_boundary", "")
            applicable_boundary = info.get("applicable_boundary", "")
            evidence = info.get("evidence_count", 0)
            dims_text += f"{i}. {name}\n"
            if meta_rule:
                dims_text += f"   Meta-rule: {meta_rule}\n"
            if rule_intent:
                dims_text += f"   Rule intent: {rule_intent}\n"
            if applicable_boundary:
                dims_text += f"   Applicable boundary: {applicable_boundary}\n"
            if decision_boundary:
                dims_text += f"   Decision boundary: {decision_boundary}\n"
            dims_text += f"   Evidence count: {evidence}\n\n"
        return dims_text.strip()


# =============================================================================
# =============================================================================

class Annotator:
    
    def __init__(self, model: str, api_key: str, base_url: str, task_type: str, category_doc_path: str, eval_func: Callable):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.eval_func = eval_func
        self.task_type = task_type
        with open(category_doc_path, 'r', encoding='utf-8') as f:
            self.doc_content = f.read()
        self.agent_names = ["annotator1", "annotator2", "annotator3", "verifier"]
        self.generated_prompt = ""
        self.verification_prompt_template = PROMPTS["verification_with_logic_scoring"]
        self.generated_prompts = []
        self._local = threading.local()

        self.pool = ContrastiveExamplePool(task_type, OUTPUT_FILE_PREFIX)
        self.dimension_manager = DimensionManager(model, self.doc_content, task_type, OUTPUT_FILE_PREFIX, pool=self.pool)
    
    def get_generated_prompts(self):
        return self.generated_prompts
    
    def get_verification_prompt_template(self):
        return self.verification_prompt_template
    
    def set_generated_prompts(self, generated_prompts: List[str]):
        self.generated_prompts = generated_prompts
    
    def set_verification_prompt_template(self, template: str):
        self.verification_prompt_template = template
    
    def _annotate_single(self, query_str: str, prompt: str,
                         contrastive_context: str, injected_samples: Dict,
                         sample_dimensions: List[Dict], agent_name: str) -> Tuple[str, bool]:
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
        step_name = f"annotation_{agent_name}" if agent_name else "annotation"
        annotation_result = call_llm_tracked(annotation_messages, self.model, temperature=0.1, step_name=step_name)
        
        sample_referenced = False
        
        return annotation_result, sample_referenced
    
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
        verification_result = call_llm_tracked(
            verification_messages, self.model,
            temperature=0.1, step_name="verification"
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

    def _evaluate_multi_annotation(self, prediction: str, ground_truth: Dict) -> Tuple[bool, int]:
        all_annotations = ground_truth.get("all_annotations", [])
        if not all_annotations:
            return False, 0

        match_count = 0
        for ann in all_annotations:
            if self.eval_func(prediction, ann):
                match_count += 1

        is_correct = match_count >= 2
        return is_correct, match_count

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
            reason = f"直接注入: {len(correct_samples_list)}正确+{len(incorrect_samples_list)}错误样例" if has_samples else "未检索到样例"

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

            action = "Inject" if should_inject else "Skip"
            dim_names = [d.get('name', '') for d in sample_dimensions[:2]]
            dim_info = f", related dimensions: {dim_names}" if dim_names else ""
            print(f"  [{agent_name}] {action} ({len(correct_samples_list)} correct + {len(incorrect_samples_list)} incorrect){dim_info}")

        inject_summary = ", ".join([f"{k}={'Y' if v['should_inject'] else 'N'}" for k, v in agent_injection_packages.items()])
        print(f"[evaluate_single] Injection decisions: {inject_summary}")

        annotation_results = [None] * len(self.generated_prompts)
        annotator_correctness = {}
        agent_sample_referenced = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_idx = {}
            for idx, prompt in enumerate(self.generated_prompts):
                agent_name = f"annotator{idx+1}"
                pkg = agent_injection_packages[agent_name]
                future = executor.submit(
                    self._annotate_single, query_str, prompt,
                    pkg['context'], pkg['samples'], pkg['dimensions'],
                    agent_name
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                agent_name = f"annotator{idx+1}"
                try:
                    annotation_result, sample_referenced = future.result()
                    annotation_results[idx] = annotation_result
                    agent_sample_referenced[agent_name] = sample_referenced

                    if isinstance(ground_truth, dict) and "all_annotations" in ground_truth:
                        is_annotator_correct, _ = self._evaluate_multi_annotation(annotation_result, ground_truth)
                    else:
                        is_annotator_correct = self.eval_func(annotation_result, ground_truth)
                    annotator_correctness[agent_name] = {
                        "is_correct": is_annotator_correct,
                        "annotation": annotation_result
                    }

                    has_context = agent_injection_packages[agent_name]['should_inject']
                    step_name = f"annotation_{agent_name}_with_context" if has_context else f"annotation_{agent_name}_without_context"
                    self._local.interactions.append({"step": step_name, "prompt": "", "response": annotation_result})
                    ref_status = f", sample referenced: {sample_referenced}" if has_context else ""
                    print(f"{agent_name}: Annotation completed {'(with context)' if has_context else ''}, correctness: {is_annotator_correct}{ref_status}")
                except Exception as e:
                    print(f"{agent_name}: Annotation failed: {e}")
                    annotation_results[idx] = ""
                    agent_sample_referenced[agent_name] = False
                    annotator_correctness[agent_name] = {
                        "is_correct": False,
                        "annotation": ""
                    }

        final_result, vote_info = self._hard_majority_vote(annotation_results)
        self._local.interactions.append({
            "step": "aggregation", "prompt": "",
            "response": f"Hard majority vote result: {vote_info['winning_label']}",
            "vote_info": vote_info
        })

        verification_info = None
        was_verified = False

        if not vote_info.get("is_unanimous", False):
            print(f"  [Verification] Inconsistent results detected, calling VerificationAgent...")
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
            print(f"  [Verification] Completed, selected: {selected_agent}")

        if isinstance(ground_truth, dict) and "all_annotations" in ground_truth:
            is_correct, match_count = self._evaluate_multi_annotation(final_result, ground_truth)
            print(f"  Multi-annotation evaluation: matched {match_count} annotators, correct: {is_correct}")
        else:
            is_correct = self.eval_func(final_result, ground_truth)

        llm_interactions = self._local.interactions.copy()
        if hasattr(self._local, 'interactions'):
            del self._local.interactions

        agent_results_for_pool = {}
        for agent_name, info in annotator_correctness.items():
            agent_results_for_pool[agent_name] = {
                "is_correct": info["is_correct"],
                "annotation": info["annotation"],
                "ground_truth": ground_truth
            }
        self.pool.update_agent_evaluation(query_str, agent_results_for_pool)

        for agent_name, pkg in agent_injection_packages.items():
            if pkg['should_inject']:
                agent_correct = annotator_correctness.get(agent_name, {}).get("is_correct", False)
                self.pool.update_sample_quality_feedback(
                    query_str=query_str,
                    was_helpful=agent_correct,
                    agent_name=agent_name
                )

        agent_inject_decisions = {k: v['should_inject'] for k, v in agent_injection_packages.items()}
        agent_injected_samples = {k: v['samples'] for k, v in agent_injection_packages.items()}
        agent_similarity_scores = {k: v['sim_scores'] for k, v in agent_injection_packages.items()}
        all_dim_names = set()
        for pkg in agent_injection_packages.values():
            for d in pkg.get('dimensions', []):
                all_dim_names.add(d.get('name', ''))

        return {
            "query": query_str,
            "annotation": final_result,
            "is_correct": is_correct,
            "llm_interactions": llm_interactions,
            "annotation_results": annotation_results,
            "ground_truth": ground_truth,
            "context_injected": any(agent_inject_decisions.values()),
            "per_agent_context_injected": agent_inject_decisions,
            "agent_injected_samples": agent_injected_samples,
            "agent_similarity_scores": agent_similarity_scores,
            "agent_sample_referenced": agent_sample_referenced,
            "relevant_dimensions": list(all_dim_names),
            "agent_correctness": annotator_correctness,
            "vote_info": vote_info,
            "was_verified": was_verified,
            "verification_info": verification_info,
        }
    
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
                    print(f"Data {idx} annotation completed")
                except Exception as e:
                    print(f"\nIndex {idx} failed: {e}")
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
                        "agent_correctness": {},
                        "vote_info": {},
                        "was_verified": False,
                        "verification_info": None,
                    }
        
        return results

    def collect_error_samples_per_agent(self, query_dicts: List[Dict], ground_truths: List[Any],
                                        evaluate_results: List[Dict]) -> Dict[str, List[Dict]]:
        num_annotators = len(self.generated_prompts)
        per_agent_errors = {f"annotator{i+1}": [] for i in range(num_annotators)}
        per_agent_errors["verifier"] = []

        for idx, (query_dict, ground_truth, evaluate_result) in enumerate(zip(query_dicts, ground_truths, evaluate_results)):
            if evaluate_result.get("is_correct", False):
                continue

            query_str = json.dumps(query_dict, ensure_ascii=False)

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
                    })

        for agent_name, samples in per_agent_errors.items():
            if samples:
                print(f"  {agent_name}: Assigned {len(samples)} error samples")

        return per_agent_errors

    def do_contrastive_mining(self, evaluate_results: List[Dict]):
        if len(evaluate_results) < 1:
            return

        mining_pool = self._build_mining_pool(evaluate_results)
        print(f"Using mixed correct sample pool for contrastive intent mining (total {len(mining_pool)}) ...")

        agent_correct_samples = {}
        for agent_name in self.pool.agent_names:
            agent_correct_samples[agent_name] = self.pool.get_agent_correct_samples(agent_name)
            print(f"  {agent_name} correct samples: {len(agent_correct_samples[agent_name])}")

        pairs = self.dimension_manager.mine_contrastive_pairs(
            evaluate_results, threshold=0.2, correct_samples_pool=mining_pool,
            agent_correct_samples=agent_correct_samples
        )
        if len(pairs) >= 1:
            print(f"Found {len(pairs)} contrastive pairs, starting discriminative dimension induction...")
            self.dimension_manager.induct_dimensions(pairs)

    def convert_llm_interactions_to_attribution_type(self, query: str, ground_truth: str, is_correct: bool, llm_interactions: List):
        processed_for_attribution = {}
        processed_for_attribution["history"] = []
        annotator_num = 0
        for llm_interaction in llm_interactions:
            step = llm_interaction.get("step", "")
            if "annotation" in step:
                processed_for_attribution["history"].append({"content": llm_interaction["response"], "role": f"Annotator{annotator_num+1}"})
                annotator_num += 1
            elif "aggregation" in step:
                vote_info = llm_interaction.get("vote_info", {})
                vote_summary = f"Hard majority vote result: {vote_info.get('winning_label', 'unknown')}, votes: {vote_info.get('vote_counts', {})}, unanimous: {vote_info.get('is_unanimous', False)}"
                processed_for_attribution["history"].append({"content": vote_summary, "role": "Aggregator"})
            elif "verification" in step:
                processed_for_attribution["history"].append({"content": llm_interaction["response"], "role": "Verifier"})
        question = PROMPTS["attribution_question"].format(doc_content=self.doc_content, query=query)
        processed_for_attribution["question"] = question
        processed_for_attribution["ground_truth"] = ground_truth
        processed_for_attribution["is_corrected"] = is_correct
        return processed_for_attribution
    
    def error_reason_analysis(self, query: str, ground_truth: str, is_correct: bool, llm_interactions: List):
        processed_for_attribution = self.convert_llm_interactions_to_attribution_type(
            query=query, ground_truth=ground_truth, is_correct=is_correct, llm_interactions=llm_interactions
        )
        attribution_result = identify_all_error_agents(processed_for_attribution, is_handcrafted=True, model=self.model, max_tokens=4096, token_counter=token_counter)
        return attribution_result
    
    def error_reason_analysis_with_contrastive(self, query: str, ground_truth: str, is_correct: bool,
                                                llm_interactions: List, correct_samples_pool: List[Dict] = None):
        processed_for_attribution = self.convert_llm_interactions_to_attribution_type(
            query=query, ground_truth=ground_truth, is_correct=is_correct, llm_interactions=llm_interactions
        )

        representative_gt = self.pool._extract_representative_gt(ground_truth)
        error_sample = {
            'query': query,
            'ground_truth': representative_gt,
            'annotation': llm_interactions[-1]['response'] if llm_interactions else '',
            'full_entry': f"Query to annotate:{query}\nModel output:{llm_interactions[-1]['response'] if llm_interactions else ''}\nCorrect answer:{representative_gt}"
        }

        contrastive_pairs = []
        if correct_samples_pool:
            contrastive_pairs = find_contrastive_pairs_for_error(
                error_sample=error_sample,
                correct_samples_pool=correct_samples_pool,
                similarity_threshold=0.2,
                top_k=3
            )
            print(f"Found {len(contrastive_pairs)} contrastive samples")

        attribution_result = identify_errors_with_contrastive_analysis(
            data=processed_for_attribution,
            contrastive_pairs=contrastive_pairs,
            is_handcrafted=True,
            model=self.model,
            max_tokens=8192,
            token_counter=token_counter
        )

        return attribution_result
    
    def do_attributions(self, query_dicts: List[Dict], ground_truths: List[str], evaluate_results: List[Dict],
                        use_contrastive_analysis: bool = True):
        global OPTIMIZE_BETTER, OPTIMIZE_WORSE, OPTIMIZE_REWARD, OPTIMIZE_PUNISH
        
        need_optimized = False
        attribution_results = []
        error_queries = []
        
        if len(evaluate_results) >= 1:
            mining_pool = self._build_mining_pool(evaluate_results)
            print(f"Using mixed correct sample pool for contrastive intent mining (batch latest + pool history, total {len(mining_pool)}) ...")
            
            agent_correct_samples = {}
            for agent_name in self.pool.agent_names:
                agent_correct_samples[agent_name] = self.pool.get_agent_correct_samples(agent_name)
                print(f"  {agent_name} correct samples: {len(agent_correct_samples[agent_name])}")
            
            pairs = self.dimension_manager.mine_contrastive_pairs(
                evaluate_results, threshold=0.2, correct_samples_pool=mining_pool,
                agent_correct_samples=agent_correct_samples
            )
            if len(pairs) >= 1:
                print(f"Found {len(pairs)} contrastive pairs, starting discriminative dimension induction...")
                self.dimension_manager.induct_dimensions(pairs)

        correct_samples_pool = None
        if use_contrastive_analysis:
            seen_queries = set()
            merged_pool = []
            for agent_name in self.pool.agent_names:
                for sample in self.pool.get_agent_correct_samples(agent_name):
                    q = sample.get('query', '')
                    if q and q not in seen_queries:
                        seen_queries.add(q)
                        merged_pool.append(sample)
            if merged_pool:
                correct_samples_pool = merged_pool
                gt_dist = {}
                for s in correct_samples_pool:
                    g = _normalize_gt_label(s.get('ground_truth', ''))
                    gt_dist[g] = gt_dist.get(g, 0) + 1
                print(f"Using pool agent-level correct sample pool for contrastive analysis: {len(correct_samples_pool)} samples, GT distribution: {gt_dist}")
            else:
                correct_samples_pool = []
                for r in evaluate_results:
                    if r.get('is_correct', False):
                        raw_outputs = r.get('annotation_results', [])
                        agent_outputs_with_names = [
                            {'agent_name': f'Annotator{i+1}', 'output': output}
                            for i, output in enumerate(raw_outputs)
                        ]
                        correct_samples_pool.append({
                            'query': r.get('query', ''),
                            'annotation': r.get('annotation', ''),
                            'ground_truth': self.pool._extract_representative_gt(r.get('ground_truth', '')),
                            'agent_outputs': agent_outputs_with_names
                        })
                print(f"Pool is empty, falling back to current batch to build correct sample pool: {len(correct_samples_pool)} samples")
        
        tasks = []
        for idx, (query_dict, ground_truth, evaluate_result) in enumerate(zip(query_dicts, ground_truths, evaluate_results)):
            query_str = json.dumps(query_dict, ensure_ascii=False)
            if not evaluate_result["is_correct"]:
                tasks.append({
                    'idx': idx,
                    'query_str': query_str,
                    'ground_truth': self.pool._extract_representative_gt(ground_truth),
                    'evaluate_result': evaluate_result
                })
        
        max_workers = min(10, len(tasks)) if tasks else 1
        results_map = {}
        
        def run_error_analysis(task):
            if use_contrastive_analysis and correct_samples_pool:
                return self.error_reason_analysis_with_contrastive(
                    query=task['query_str'],
                    ground_truth=task['ground_truth'],
                    is_correct=task['evaluate_result']["is_correct"],
                    llm_interactions=task['evaluate_result']["llm_interactions"],
                    correct_samples_pool=correct_samples_pool
                )
            else:
                return self.error_reason_analysis(
                    query=task['query_str'],
                    ground_truth=task['ground_truth'],
                    is_correct=task['evaluate_result']["is_correct"],
                    llm_interactions=task['evaluate_result']["llm_interactions"]
                )
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(run_error_analysis, task): task
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    attribution_result = future.result()
                    results_map[task['idx']] = {
                        'attribution_result': attribution_result,
                        'error_query': {
                            "query": task['query_str'],
                            "ground_truth": task['ground_truth'],
                            "annotation": task['evaluate_result']["annotation"]
                        }
                    }
                    if attribution_result is not None:
                        print(f"Data {task['idx']} error attribution completed")
                    else:
                        print(f"Data {task['idx']} error attribution failed: returned None")
                except Exception as e:
                    print(f"\nIndex {task['idx']} error attribution failed: {e}")
                    traceback.print_exc()
        
        for idx in sorted(results_map.keys()):
            result = results_map[idx]
            if result['attribution_result'] is not None:
                attribution_results.append(result['attribution_result'])
                error_queries.append(result['error_query'])
        
        print("Error attribution completed, ready for optimization")
        
        agent_names = self.agent_names
        agents_attributions = {}
        for agent_name in agent_names:
            agents_attributions[agent_name] = {
                "failure_reasons": "",
                "failure_count": 0,
                "failure_samples": "",
                "error_dimension_names": set()
            }
        
        for error_query, attribution_result in zip(error_queries, attribution_results):
            if attribution_result is None:
                continue
            
            error_agents_list = attribution_result.get("error_agents", [])
            
            for agent_info in error_agents_list:
                agent_name = agent_info.get("智能体名称", agent_info.get("agent_name", "")).lower().strip()
                if agent_name in agent_names:
                    error_output = agent_info.get('错误输出', agent_info.get('error_output', ''))
                    error_reason = agent_info.get('错误原因', agent_info.get('error_reason', ''))

                    error_sample = f"Query to annotate:{error_query['query']}\nError output:{error_output}\nCorrect answer:{error_query['ground_truth']}"

                    count = agents_attributions[agent_name]["failure_count"]
                    agents_attributions[agent_name]["failure_reasons"] += f"Error reason {count+1}:\n{error_reason}\n\n"
                    agents_attributions[agent_name]["failure_samples"] += f"Error sample {count+1}:\n{error_sample}\n\n"
                    agents_attributions[agent_name]["failure_count"] += 1

                    related_dims = self.pool.find_dimensions_for_sample(
                        error_query['query'],
                        dimension_manager=self.dimension_manager
                    )
                    for dim in related_dims:
                        dim_name = dim.get('name', '')
                        if dim_name:
                            agents_attributions[agent_name]["error_dimension_names"].add(dim_name)

                    need_optimized = True
        
        for error_query, attribution_result in zip(error_queries, attribution_results):
            if attribution_result is None:
                continue
            error_agents_list = attribution_result.get("error_agents", [])
            for agent_info in error_agents_list:
                agent_name = agent_info.get("智能体名称", agent_info.get("agent_name", "")).lower().strip()
                error_reason = agent_info.get("错误原因", agent_info.get("error_reason", ""))
                if agent_name in self.pool.agent_incorrect_samples:
                    for sample in self.pool.agent_incorrect_samples[agent_name]:
                        if sample.get("query") == error_query["query"]:
                            sample["error_reason"] = error_reason
                            break

        self._update_badcases_file(error_queries, attribution_results, agents_attributions)
        self.pool.load_historical_badcases()
        
        return {
            "evaluate_results": evaluate_results,
            "agents_attributions": agents_attributions,
            "need_optimized": need_optimized,
            "attribution_details": [
                {
                    "query": eq.get("query", ""),
                    "ground_truth": eq.get("ground_truth", ""),
                    "annotation": eq.get("annotation", ""),
                    "attribution": ar
                }
                for eq, ar in zip(error_queries, attribution_results)
            ]
        }
    
    def _update_badcases_file(self, error_queries: List[Dict], attribution_results: List[Dict], agents_attributions: Dict):
        badcases_file = f"cases/{self.task_type}/badcases_{OUTPUT_FILE_PREFIX}.jsonl"
        os.makedirs(os.path.dirname(badcases_file), exist_ok=True)
        
        existing_badcases = []
        if os.path.exists(badcases_file):
            with open(badcases_file, "r", encoding="utf-8") as f:
                for line in f:
                    existing_badcases.append(json.loads(line, strict=False))
        
        badcases_index_map = {(b.get("query"), b.get("agent_name")): i for i, b in enumerate(existing_badcases)}
        new_badcases = []
        
        for error_query, attribution_result in zip(error_queries, attribution_results):
            if attribution_result is None:
                continue
            error_agents_list = attribution_result.get("error_agents", [])
            for agent_info in error_agents_list:
                agent_name = agent_info.get("智能体名称", agent_info.get("agent_name", "")).lower().strip()
                key = (error_query["query"], agent_name)

                if key in badcases_index_map:
                    idx = badcases_index_map[key]
                    existing_badcases[idx]["error_output"] = agent_info.get("错误输出", agent_info.get("error_output", ""))
                    existing_badcases[idx]["error_reason"] = agent_info.get("错误原因", agent_info.get("error_reason", ""))
                    existing_badcases[idx]["num_wrong"] = existing_badcases[idx].get("num_wrong", 0) + 1
                else:
                    new_badcases.append({
                        "query": error_query["query"],
                        "error_output": agent_info.get("错误输出", agent_info.get("error_output", "")),
                        "ground_truth": error_query["ground_truth"],
                        "agent_name": agent_name,
                        "error_reason": agent_info.get("错误原因", agent_info.get("error_reason", "")),
                        "num_wrong": 1,
                        "optimize_priority": 1.0
                    })
        
        with open(badcases_file, "w", encoding="utf-8") as f:
            for bc in existing_badcases + new_badcases:
                f.write(json.dumps(bc, ensure_ascii=False) + "\n")
    
    def optimize_prompts(self, agents_attributions: Dict, evaluate_results: List[Dict], use_intent_framework: bool = True):
        global OPTIMIZE_BETTER, OPTIMIZE_WORSE

        new_generated_prompts = self.generated_prompts.copy()
        new_verification_prompt_template = self.verification_prompt_template

        optimization_tasks = []

        for agent_name in ["annotator1", "annotator2", "annotator3"]:
            prompt_num = -1
            if agent_name == "annotator1":
                prompt_num = 0
            elif agent_name == "annotator2":
                prompt_num = 1
            elif agent_name == "annotator3":
                prompt_num = 2

            if prompt_num != -1 and agent_name in agents_attributions:
                agent_info = agents_attributions[agent_name]
                failure_count = agent_info.get("failure_count", 0)
                if failure_count > 0:
                    contrastive_pairs = self._find_contrastive_pairs_for_agent(agent_name, evaluate_results, agents_attributions)
                    print(f"{agent_name} found {len(contrastive_pairs)} contrastive pairs (total {failure_count} error samples)")

                    optimization_tasks.append({
                        "agent_name": agent_name,
                        "prompt_num": prompt_num,
                        "failure_samples": agent_info.get("failure_samples", ""),
                        "failure_reasons": agent_info.get("failure_reasons", ""),
                        "contrastive_pairs": contrastive_pairs,
                        "error_dimension_names": agent_info.get("error_dimension_names", set()),
                    })

        if "verifier" in agents_attributions:
            verifier_info = agents_attributions["verifier"]
            verifier_failure_count = verifier_info.get("failure_count", 0)
            if verifier_failure_count > 0:
                contrastive_pairs = self._find_contrastive_pairs_for_agent("verifier", evaluate_results, agents_attributions)
                print(f"verifier found {len(contrastive_pairs)} contrastive pairs (total {verifier_failure_count} error samples)")

                optimization_tasks.append({
                    "agent_name": "verifier",
                    "prompt_num": -1,
                    "failure_samples": verifier_info.get("failure_samples", ""),
                    "failure_reasons": verifier_info.get("failure_reasons", ""),
                    "contrastive_pairs": contrastive_pairs,
                })

        def optimize_single_agent(task: Dict) -> Tuple[str, int, str]:
            agent_name = task["agent_name"]
            prompt_num = task["prompt_num"]
            failure_samples = task["failure_samples"]
            failure_reasons = task["failure_reasons"]
            contrastive_pairs = task.get("contrastive_pairs", [])

            if prompt_num != -1:
                original_prompt = self.generated_prompts[prompt_num]
                dimensions = None
                if use_intent_framework and self.dimension_manager.intent_hypotheses:
                    error_dim_names = task.get("error_dimension_names", set())
                    if error_dim_names:
                        dimensions = [
                            d for name, d in self.dimension_manager.intent_hypotheses.items()
                            if name in error_dim_names
                        ]
                        print(f"{agent_name}: Referencing {len(dimensions)} error-related dimensions during optimization (total {len(self.dimension_manager.intent_hypotheses)})")
                    else:
                        print(f"{agent_name}: No direct dimension labels")

                optimization_prompt = self._build_contrastive_optimization_prompt(
                    base_prompt=original_prompt,
                    contrastive_pairs=contrastive_pairs,
                    failure_reasons=failure_reasons,
                    doc_content=self.doc_content,
                    dimensions=dimensions,
                    failure_samples=failure_samples
                )

                optimization_messages = prompt_to_messages(optimization_prompt)
                optimization_result = call_llm_tracked(
                    optimization_messages,
                    self.model,
                    temperature=0.1,
                    stream=True,
                    step_name="optimize_prompt"
                )

                optimized_prompt = None
                if "optimized prompt:" in optimization_result.lower():
                    optimized_prompt = optimization_result.lower().split("optimized prompt:")[-1]
                else:
                    print("Failed to parse the optimized prompt")

                return agent_name, prompt_num, optimized_prompt
            else:
                base_rules = self.verification_prompt_template
                contrastive_pairs = task.get("contrastive_pairs", [])
                dimensions = task.get("dimensions", [])

                dimension_text = ""
                if dimensions:
                    for i, dim in enumerate(dimensions[:5], 1):
                        dim_name = dim.get('name', '')
                        if not dim_name:
                            continue
                        dimension_text += PROMPTS["dimension_item_optimization"].format(
                            dim_name=dim_name,
                            evidence_count=dim.get('evidence_count', 0),
                            rule_intent=dim.get('rule_intent', '') or 'None yet',
                            decision_boundary=dim.get('decision_boundary', '') or 'None yet',
                            applicable_boundary=dim.get('applicable_boundary', '') or 'None yet'
                        )

                contrastive_text = ""
                for i, (corr, err, sim) in enumerate(contrastive_pairs[:5], 1):
                    contrastive_text += f"\n[Contrastive Sample {i}]\n"
                    if corr is not None:
                        contrastive_text += f"Sample A (System judged correct):\n"
                        contrastive_text += f"  Query: {corr.get('query', '')}\n"
                        contrastive_text += f"  System output: {corr.get('annotation', '')}\n"
                        contrastive_text += f"  Correct answer: {self.pool._extract_representative_gt(corr.get('ground_truth', ''))}\n"
                    else:
                        contrastive_text += f"Sample A (System judged correct):\n"
                        contrastive_text += f"  Query: No correct sample similar to Sample B found\n"
                    contrastive_text += f"\nSample B (System judged incorrect):\n"
                    contrastive_text += f"  {err.get('full_entry', '')}\n"

                reference_section = ""
                if dimension_text or contrastive_text:
                    ref_parts = ["\n## Auxiliary Reference Information"]
                    ref_parts.append("The following dimensions and contrastive samples can help understand error patterns, but optimization should be primarily based on the error samples above:")

                    if dimension_text:
                        ref_parts.append("\n#### Learned Discriminative Dimensions Reference")
                        ref_parts.append(dimension_text)

                    if contrastive_text:
                        ref_parts.append("\n#### Contrastive Sample Reference")
                        ref_parts.append(contrastive_text)

                    reference_section = "\n".join(ref_parts)
                    reference_section = reference_section + PROMPTS["reference_optimization_principles"]

                verifier_optimization_prompt = PROMPTS["verifier_optimization_prompt"].format(
                    base_rules=base_rules,
                    failure_samples=failure_samples,
                    failure_reasons=failure_reasons,
                    reference_section=reference_section
                )

                verifier_optimization_messages = prompt_to_messages(verifier_optimization_prompt)
                verifier_optimization_result = call_llm_tracked(
                    verifier_optimization_messages,
                    self.model,
                    temperature=0.1,
                    stream=True,
                    step_name="optimize_verification_prompt"
                )

                optimized_template = None
                if "optimized verification prompt:" in verifier_optimization_result.lower():
                    optimized_template = verifier_optimization_result.lower().split("optimized verification prompt:")[-1]
                elif "优化后的验证提示:" in verifier_optimization_result:
                    optimized_template = verifier_optimization_result.split("优化后的验证提示:")[-1]
                elif "优化后的验证提示：" in verifier_optimization_result:
                    optimized_template = verifier_optimization_result.split("优化后的验证提示：")[-1]
                else:
                    print("Failed to parse the optimized verifier prompt")

                return agent_name, prompt_num, optimized_template

        if optimization_tasks:
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
                                print(f"{agent_name}: Prompt optimization based on contrastive samples completed")
                            else:
                                new_verification_prompt_template = optimized_result
                                print(f"{agent_name}: Verification prompt template optimization based on contrastive samples completed")
                        else:
                            print(f"{agent_name}: Optimization failed, could not extract optimization result")
                    except Exception as e:
                        agent_name = task["agent_name"]
                        print(f"{agent_name}: Optimization failed: {e}")
                        traceback.print_exc()
        else:
            print("No agents need optimization (all agents have no errors)")

        return new_generated_prompts, new_verification_prompt_template
    
    def _find_contrastive_pairs_for_agent(self, agent_name: str, evaluate_results: List[Dict], agents_attributions: Dict) -> List[Tuple]:
        agent_attribution = agents_attributions.get(agent_name, {})
        failure_count = agent_attribution.get("failure_count", 0)

        if failure_count == 0:
            return []

        if agent_name in ["annotator1", "annotator2", "annotator3"]:
            correct_samples = self.pool.get_agent_correct_samples(agent_name)
            print(f"[_find_contrastive_pairs] {agent_name}: Using agent-level correct sample pool ({len(correct_samples)} samples)")
        else:
            correct_samples = self.pool.correct_samples if self.pool.correct_samples else []
            print(f"[_find_contrastive_pairs] {agent_name}: Using system-level correct sample pool ({len(correct_samples)} samples)")

        if not correct_samples:
            correct_samples = [r for r in evaluate_results if r.get('is_correct', False)]
            print(f"[_find_contrastive_pairs] {agent_name}: Falling back to current batch correct samples ({len(correct_samples)} samples)")

        if not correct_samples:
            return []

        failure_samples_str = agent_attribution.get("failure_samples", "")
        error_entries = re.split(r'Error sample \d+:', failure_samples_str)
        error_entries = [e.strip() for e in error_entries if e.strip()]

        contrastive_pairs = []

        for error_entry in error_entries:
            lines = error_entry.split('\n')
            query_line = ''
            for line in lines:
                if line.startswith('Query to annotate:'):
                    query_line = line.replace('Query to annotate:', '').strip()
                    break

            if not query_line:
                continue

            err_embedding = get_or_create_embedding(query_line)
            if err_embedding == 0:
                continue

            candidates = []
            for corr in correct_samples:
                corr_query = corr.get('query', '')
                corr_embedding = get_or_create_embedding(corr_query)
                if corr_embedding == 0:
                    continue
                similarity = cosine_similarity(err_embedding, corr_embedding)
                if similarity > 0.2 and similarity < 0.99:
                    candidates.append((corr, similarity))

            error_sample_info = {
                'query': query_line,
                'full_entry': error_entry
            }

            if not candidates:
                contrastive_pairs.append((None, error_sample_info, 0.0))
                continue

            candidates.sort(key=lambda x: x[1], reverse=True)

            if len(candidates) >= 2:
                best1 = candidates[0]
                best2 = candidates[1]
                contrastive_pairs.append((best1[0], error_sample_info, best1[1]))
                contrastive_pairs.append((best2[0], error_sample_info, best2[1]))
            else:
                best = candidates[0]
                contrastive_pairs.append((best[0], error_sample_info, best[1]))

        contrastive_pairs.sort(key=lambda x: x[2], reverse=True)

        valid_pairs = [p for p in contrastive_pairs if p[0] is not None]
        print(f"[_find_contrastive_pairs] {agent_name}: Found {len(valid_pairs)}/{len(contrastive_pairs)} valid contrastive pairs")

        return contrastive_pairs

    def _build_contrastive_optimization_prompt(self, base_prompt: str, contrastive_pairs: List[Tuple],
                                                failure_reasons: str, doc_content: str,
                                                dimensions: List[Dict] = None, failure_samples: str = "") -> str:
        # Build error samples section (main focus)
        error_samples_section = ""
        if failure_samples:
            error_samples_section = PROMPTS["annotator_optimization_error_samples_section"].format(
                failure_samples=failure_samples
            )

        # Build reference information section (auxiliary)
        reference_section = ""

        # Contrastive samples as reference
        if contrastive_pairs:
            contrastive_text = ""
            for i, (corr, err, sim) in enumerate(contrastive_pairs[:5], 1):
                contrastive_text += f"\n[Contrastive Sample {i}]\n"
                if corr is not None:
                    contrastive_text += f"Sample A (Current Annotator judged correct):\n"
                    contrastive_text += f"  Query: {corr.get('query', '')}...\n"
                    contrastive_text += f"  Current Annotator output: {corr.get('annotation', '')}...\n"
                    contrastive_text += f"  Correct answer: {self.pool._extract_representative_gt(corr.get('ground_truth', ''))}...\n\n"
                else:
                    contrastive_text += f"Sample A (Current Annotator judged correct):\n"
                    contrastive_text += f"  Query: No correct sample similar to Sample B found\n"
                contrastive_text += f"Sample B (Current Annotator judged incorrect):\n"
                contrastive_text += f"  {err.get('full_entry', '')}...\n"
            reference_section += PROMPTS["annotator_optimization_contrastive_section"].format(
                contrastive_text=contrastive_text
            )

        if dimensions:
            dimension_text = ""
            for dim in dimensions:
                dim_name = dim.get('name', '')
                if not dim_name:
                    continue
                rule_intent = dim.get('rule_intent', '')
                decision_boundary = dim.get('decision_boundary', '')
                applicable_boundary = dim.get('applicable_boundary', '')
                evidence_count = dim.get('evidence_count', 0)

                dimension_text += PROMPTS["dimension_item_optimization"].format(
                    dim_name=dim_name,
                    evidence_count=evidence_count,
                    rule_intent=rule_intent or 'None yet',
                    decision_boundary=decision_boundary or 'None yet',
                    applicable_boundary=applicable_boundary or 'None yet'
                )
            if dimension_text:
                reference_section += PROMPTS["annotator_optimization_dimensions_section"].format(
                    dimension_text=dimension_text
                )

        # Wrap reference information
        if reference_section:
            reference_section = PROMPTS["annotator_optimization_reference_header"] + reference_section
            reference_section = reference_section + PROMPTS["reference_optimization_principles"]

        prompt = PROMPTS["annotator_optimization_prompt"].format(
            base_prompt=base_prompt,
            error_samples_section=error_samples_section,
            failure_reasons=failure_reasons,
            reference_section=reference_section
        )
        return prompt
    
    def _is_pair_related_to_dimension(self, pair: Tuple, dimension: Dict) -> bool:
        corr, err = pair[0], pair[1]
        dim_text = self.dimension_manager._get_dimension_embedding_text(dimension)
        
        if not dim_text:
            return False
        
        dim_emb = get_or_create_embedding(dim_text)
        if dim_emb == 0:
            return False
        
        max_sim = 0.0
        
        if corr:
            corr_query = corr.get('query', '')
            if corr_query:
                corr_emb = get_or_create_embedding(corr_query)
                if corr_emb != 0:
                    max_sim = max(max_sim, cosine_similarity(corr_emb, dim_emb))
        
        if err:
            err_query = err.get('query', '')
            if err_query:
                err_emb = get_or_create_embedding(err_query)
                if err_emb != 0:
                    max_sim = max(max_sim, cosine_similarity(err_emb, dim_emb))
        
        return max_sim > 0.5
    
    def _build_mining_pool(self, evaluate_results: List[Dict]) -> List[Dict]:
        batch_correct = []
        for r in evaluate_results:
            if r.get('is_correct', False):
                raw_outputs = r.get('annotation_results', [])
                agent_outputs_with_names = [
                    {'agent_name': f'Annotator{i+1}', 'output': output}
                    for i, output in enumerate(raw_outputs)
                ]
                batch_correct.append({
                    'query': r.get('query', ''),
                    'annotation': r.get('annotation', ''),
                    'ground_truth': self.pool._extract_representative_gt(r.get('ground_truth', '')),
                    'agent_outputs': agent_outputs_with_names
                })
        
        pool_correct = [s.copy() for s in self.pool.correct_samples]
        
        seen = set()
        merged = []
        for s in batch_correct + pool_correct:
            q = s.get('query', '')
            if q and q not in seen:
                seen.add(q)
                merged.append(s)
        
        return merged
    
    def evaluate_then_attributions(self, query_dicts: List[Dict], ground_truths: List[str],
                                    use_contrastive_analysis: bool = True):
        evaluate_results = self.evaluate_batch(
            query_dicts=query_dicts,
            ground_truths=ground_truths
        )
        return self.do_attributions(
            query_dicts=query_dicts,
            ground_truths=ground_truths,
            evaluate_results=evaluate_results,
            use_contrastive_analysis=use_contrastive_analysis
        )
    
    def evaluate_on_full_dataset(self, all_queries: List[Dict], all_ground_truths: List[str], batch_size: int = 50, sync_pool: bool = True) -> Tuple[int, int, float]:
        print(f"\n========== Starting evaluation of current prompts on full training set ({len(all_queries)} samples) ==========")
        prev_mode = self.pool.mode
        self.pool.set_mode("test")

        total_correct = 0
        total_count = len(all_queries)

        for i in range(0, total_count, batch_size):
            batch_queries = all_queries[i:i+batch_size]
            batch_ground_truths = all_ground_truths[i:i+batch_size]
            batch_results = self.evaluate_batch(batch_queries, batch_ground_truths)
            
            if sync_pool:
                self.pool.update_from_batch_results(batch_queries, batch_results)
            
            for result in batch_results:
                if result.get("is_correct", False):
                    total_correct += 1
            
            print(f"  Evaluated {min(i+batch_size, total_count)}/{total_count} samples, current accuracy: {total_correct/min(i+batch_size, total_count):.4f}")
        
        accuracy = total_correct / total_count if total_count > 0 else 0
        print(f"========== Training set evaluation complete: {total_correct}/{total_count} = {accuracy:.4f} ({accuracy*100:.2f}%) ==========\n")

        self.pool.set_mode(prev_mode)

        return total_correct, total_count, accuracy


# =============================================================================
# =============================================================================

def run_evaluation():
    global OPTIMIZE_REWARD, OPTIMIZE_PUNISH, OPTIMIZE_BETTER, OPTIMIZE_WORSE, RESUME_EPOCH, TASK_TYPE
    
    start_id = 100
    end_id = 300
    num_epochs = 6
    num_rounds = 3
    
    task_config = get_task_config(TASK_TYPE, end_id)
    TASK_TYPE = TASK_TYPE.replace("multi_","")
    evaluate_func = EVALUATION_FUNCTIONS.get(TASK_TYPE)
    
    all_queries = task_config["queries"][start_id:end_id]
    all_ground_truths = task_config["answers"][start_id:end_id]
    
    queries = all_queries.copy()
    ground_truths = all_ground_truths.copy()
    
    os.makedirs(os.path.dirname(f"output/{TASK_TYPE}"), exist_ok=True)
    
    model = "DeepSeek-V3.1-Terminus"
    
    evaluator = Annotator(
        model=model,
        api_key="",
        base_url="",
        task_type=TASK_TYPE,
        category_doc_path=task_config["requirement_doc_path"],
        eval_func=evaluate_func
    )
    
    evaluator.pool.initialize_from_dataset(all_queries, all_ground_truths)
    # evaluator.pool.load_historical_badcases()
    
    doc_content = evaluator.doc_content
    generated_prompts = [doc_content, doc_content, doc_content]
    evaluator.set_generated_prompts(generated_prompts)
    print(f"Using rule document as initial prompt for {len(generated_prompts)} annotators")

    print("generated prompts setup complete")
    init_generated_prompts = copy.copy(evaluator.get_generated_prompts())
    init_verification_prompt_template = evaluator.get_verification_prompt_template()
    
    epoch_accuracies = []
    
    if RESUME_EPOCH is not None and RESUME_EPOCH > 0:
        resume_file = f"output/{TASK_TYPE}/{TASK_TYPE}_multi_agents_prompt_optimization_with_{OUTPUT_FILE_PREFIX}_epoch{RESUME_EPOCH-1}.json"
        if os.path.exists(resume_file):
            print(f"\n>>> Resuming training state from {resume_file} ...")
            with open(resume_file, 'r', encoding='utf-8') as f:
                resume_data = json.load(f)
            
            if "current_generated_prompts" in resume_data:
                evaluator.set_generated_prompts(resume_data["current_generated_prompts"])
                print(f"  Restored generated_prompts")
            
            if "current_verification_prompt_template" in resume_data:
                evaluator.set_verification_prompt_template(resume_data["current_verification_prompt_template"])
                print("  Restored verification_prompt_template")
            
            if "intent_hypotheses" in resume_data:
                evaluator.dimension_manager.intent_hypotheses = resume_data["intent_hypotheses"]
                evaluator.dimension_manager._save_intent_hypotheses()
                print(f"  Restored intent_hypotheses ({len(resume_data['intent_hypotheses'])} dimensions)")
            
            if "pool_state" in resume_data and resume_data["pool_state"]:
                evaluator.pool.correct_samples = resume_data["pool_state"].get("correct_samples", [])
                evaluator.pool.incorrect_samples = resume_data["pool_state"].get("incorrect_samples", [])
                
                if "agent_correct_samples" in resume_data["pool_state"]:
                    evaluator.pool.agent_correct_samples = resume_data["pool_state"]["agent_correct_samples"]
                if "agent_incorrect_samples" in resume_data["pool_state"]:
                    evaluator.pool.agent_incorrect_samples = resume_data["pool_state"]["agent_incorrect_samples"]
                if "agent_evaluated_flags" in resume_data["pool_state"]:
                    evaluator.pool._agent_evaluated_flags = resume_data["pool_state"]["agent_evaluated_flags"]
                if "sample_scorer_state" in resume_data["pool_state"]:
                    evaluator.pool.sample_scorer.import_state(resume_data["pool_state"]["sample_scorer_state"])
                    scorer_stats = evaluator.pool.sample_scorer.get_sample_stats()
                    print(f"[Pool] Restored sample quality scorer from checkpoint: {scorer_stats.get('total_samples', 0)} samples with scores")

                if "dimension_sample_index" in resume_data["pool_state"]:
                    evaluator.pool.dimension_sample_index = resume_data["pool_state"]["dimension_sample_index"]
                    print(f"[Pool] Restored dimension-sample index from checkpoint: {len(evaluator.pool.dimension_sample_index)} dimensions")

                if "sample_dimension_tags" in resume_data["pool_state"]:
                    evaluator.pool.sample_dimension_tags = resume_data["pool_state"]["sample_dimension_tags"]
                    print(f"[Pool] Restored sample dimension tags from checkpoint: {len(evaluator.pool.sample_dimension_tags)} samples")
                
                agent_stats = ", ".join([f"{agent}: {len(evaluator.pool.agent_correct_samples.get(agent, []))}正" 
                                          for agent in evaluator.pool.agent_names])
                print(f"  Restored pool (correct: {len(evaluator.pool.correct_samples)}, incorrect: {len(evaluator.pool.incorrect_samples)})")
                print(f"  Restored agent-level sample pool: {agent_stats}")
            
            if "epoch_accuracies_so_far" in resume_data:
                epoch_accuracies = resume_data["epoch_accuracies_so_far"]
                print(f"  Restored epoch_accuracies")
            
            if "optimize_better" in resume_data:
                OPTIMIZE_BETTER = resume_data["optimize_better"]
            if "optimize_worse" in resume_data:
                OPTIMIZE_WORSE = resume_data["optimize_worse"]
            
            if "dimension_manager_state" in resume_data:
                dm_state = resume_data["dimension_manager_state"]
                if "similarity_history" in dm_state:
                    evaluator.dimension_manager._similarity_history = dm_state["similarity_history"]
                    print(f"  Restored similarity_history ({len(dm_state['similarity_history'])} records)")
            
            if epoch_accuracies:
                last_epoch_info = epoch_accuracies[-1]
                init_correct_count = last_epoch_info["after"]["correct"]
                init_total_count = last_epoch_info["after"]["total"]
                init_accuracy = last_epoch_info["after"]["accuracy"]
            else:
                init_correct_count = 0
                init_total_count = len(all_queries)
                init_accuracy = 0
            
            print(f">>> Successfully resumed to epoch {RESUME_EPOCH} state\n")
            pool_restored = True
        else:
            print(f"\nWarning: Resume file {resume_file} not found, starting training from scratch\n")
            RESUME_EPOCH = None
            pool_restored = False
    else:
        pool_restored = False
    
    if not pool_restored:
        print("""
>>> Initializing dynamic correct sample pool...""")
        evaluator.pool.refresh(evaluator, batch_size=POOL_REFRESH_BATCH_SIZE)
        init_correct_count = len(evaluator.pool.correct_samples)
        init_total_count = len(evaluator.pool.all_samples)
        init_accuracy = init_correct_count / init_total_count if init_total_count > 0 else 0
        print(f"Dynamic correct sample pool initialization complete: {init_correct_count}/{init_total_count} = {init_accuracy:.4f} ({init_accuracy*100:.2f}%)\n")
    else:
        print(f"\n>>> Dynamic correct sample pool restored")
        print(f"Using previous round optimized evaluation results as reference: {init_correct_count}/{init_total_count} = {init_accuracy:.4f} ({init_accuracy*100:.2f}%)\n")
    
    prev_after_correct, prev_after_total, prev_after_accuracy = init_correct_count, init_total_count, init_accuracy
    start_epoch = RESUME_EPOCH if RESUME_EPOCH is not None else 0
    
    for epoch in range(start_epoch, num_epochs):
        output_file = f"output/{TASK_TYPE}/{TASK_TYPE}_multi_agents_prompt_optimization_with_{OUTPUT_FILE_PREFIX}_epoch{epoch}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs} started")
        print(f"{'='*60}")
        
        if epoch == 0:
            before_correct, before_total, before_accuracy = init_correct_count, init_total_count, init_accuracy
        else:
            before_correct, before_total, before_accuracy = prev_after_correct, prev_after_total, prev_after_accuracy
        
        print(f">>> Epoch {epoch+1} accuracy before optimization: {before_correct}/{before_total} = {before_accuracy:.4f} ({before_accuracy*100:.2f}%)")
        
        combined = list(zip(queries, ground_truths))
        random.shuffle(combined)
        queries[:], ground_truths[:] = zip(*combined)
        
        results = [None] * len(queries)
        missing_or_error_indices = list(range(len(queries)))
        batch_size = 20
        
        for i in range(0, len(missing_or_error_indices), batch_size):
            batch_idxs = missing_or_error_indices[i:i+batch_size]
            batch_queries = [queries[idx] for idx in batch_idxs]
            batch_ground_truths = [ground_truths[idx] for idx in batch_idxs]
            print(f"----------------------Starting optimization round {i//batch_size}-----------------------")
            
            try:
                result = evaluator.evaluate_then_attributions(
                    batch_queries,
                    batch_ground_truths,
                    use_contrastive_analysis=True,
                )

                ori_num_correct = 0
                for evaluate_result in result["evaluate_results"]:
                    if evaluate_result["is_correct"]:
                        ori_num_correct += 1

                ori_generated_prompts = copy.copy(evaluator.get_generated_prompts())
                ori_verification_prompt_template = evaluator.get_verification_prompt_template()

                if ori_num_correct < len(batch_queries):
                    for round_id in range(num_rounds):
                        print(f"Starting optimization round {round_id} this time")
                        use_framework = len(evaluator.dimension_manager.intent_hypotheses) > 0
                        if use_framework:
                            print(f"Using intent framework optimization ({len(evaluator.dimension_manager.intent_hypotheses)} dimensions)")

                        new_generated_prompts, new_verification_prompt_template = evaluator.optimize_prompts(
                            result["agents_attributions"],
                            result["evaluate_results"],
                            use_intent_framework=use_framework
                        )
                        evaluator.set_generated_prompts(new_generated_prompts)
                        evaluator.set_verification_prompt_template(new_verification_prompt_template)

                        eval_results_for_new_prompts = evaluator.evaluate_batch(
                            batch_queries,
                            batch_ground_truths,
                        )

                        new_num_correct = 0
                        for eval_result in eval_results_for_new_prompts:
                            if eval_result["is_correct"]:
                                new_num_correct += 1

                        if ori_num_correct > new_num_correct:
                            OPTIMIZE_WORSE += 1
                            print("New prompt performance poor, fallback to original prompt (dimensions remain unchanged)")
                            evaluator.set_generated_prompts(ori_generated_prompts)
                            evaluator.set_verification_prompt_template(ori_verification_prompt_template)
                        else:
                            result["evaluate_results"] = eval_results_for_new_prompts

                            if ori_num_correct < new_num_correct:
                                OPTIMIZE_BETTER += 1
                                print("Newly generated prompt better than original prompt")

                                print(">>> Prompt optimization successful, accepting new prompt")

                                ori_generated_prompts = copy.copy(new_generated_prompts)
                                ori_verification_prompt_template = new_verification_prompt_template
                                ori_num_correct = new_num_correct
                            elif ori_num_correct == new_num_correct:
                                print("New prompt performance unchanged, accepting new prompt")
                                ori_generated_prompts = copy.copy(new_generated_prompts)
                                ori_verification_prompt_template = new_verification_prompt_template
                                ori_num_correct = new_num_correct

                            if new_num_correct >= len(batch_queries):
                                print("This batch all answered correctly, can skip")
                                break

                            if round_id >= num_rounds - 1:
                                continue
                            else:
                                result = evaluator.do_attributions(
                                    batch_queries,
                                    batch_ground_truths,
                                    eval_results_for_new_prompts,
                                    use_contrastive_analysis=True
                                )
                else:
                    print("This batch samples all correct, no need for prompt optimization")

                evaluator.pool.update_from_batch_results(batch_queries, result["evaluate_results"])

                for eval_res in result["evaluate_results"]:
                    if eval_res.get("context_injected", False):
                        evaluator.dimension_manager.update_dimension_outcomes(
                            eval_res.get("relevant_dimensions", []),
                            eval_res.get("is_correct", False)
                        )
                
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
                evaluate_results = result["evaluate_results"]
                results[idx] = {}
                results[idx]["annotation"] = evaluate_results[batch_id].get("annotation", "")
                results[idx]["is_correct"] = evaluate_results[batch_id]["is_correct"]
                results[idx]["query"] = json.dumps(queries[idx], ensure_ascii=False)
                results[idx]["ground_truth"] = ground_truths[idx]
                results[idx]["llm_interactions"] = evaluate_results[batch_id]["llm_interactions"]
                results[idx]["generated_prompts"] = ori_generated_prompts
                results[idx]["verification_prompt_template"] = ori_verification_prompt_template
                results[idx]["context_injected"] = evaluate_results[batch_id].get("context_injected", False)
                results[idx]["per_agent_context_injected"] = evaluate_results[batch_id].get("per_agent_context_injected", {})
                results[idx]["relevant_dimensions"] = evaluate_results[batch_id].get("relevant_dimensions", [])
                results[idx]["agent_sample_referenced"] = evaluate_results[batch_id].get("agent_sample_referenced", {})
                results[idx]["agent_similarity_scores"] = evaluate_results[batch_id].get("agent_similarity_scores", {})
        
        if OPTIMIZE_BETTER + OPTIMIZE_WORSE > 0:
            OPTIMIZE_REWARD = OPTIMIZE_WORSE / (OPTIMIZE_WORSE + OPTIMIZE_BETTER)
            OPTIMIZE_PUNISH = OPTIMIZE_BETTER / (OPTIMIZE_WORSE + OPTIMIZE_BETTER)
        
        batch_correct = 0
        for result in results:
            if result and result.get("is_correct", False):
                batch_correct += 1
        batch_accuracy = batch_correct / len(results) if results else 0
        print(f"\n>>> Epoch {epoch+1} batch optimization complete, batch accuracy: {batch_correct}/{len(results)} = {batch_accuracy:.4f} ({batch_accuracy*100:.2f}%)")
        
        print(f">>> Epoch {epoch+1} after optimization, testing current prompts accuracy on training set (1st time):")
        after_correct1, after_total1, after_accuracy1 = evaluator.evaluate_on_full_dataset(
            all_queries, all_ground_truths, batch_size=50
        )
        print(f">>> Epoch {epoch+1} after optimization, testing current prompts accuracy on training set (2nd time):")
        after_correct2, after_total2, after_accuracy2 = evaluator.evaluate_on_full_dataset(
            all_queries, all_ground_truths, batch_size=50
        )
        print(f">>> Epoch {epoch+1} after optimization, testing current prompts accuracy on training set (3rd time):")
        after_correct3, after_total3, after_accuracy3 = evaluator.evaluate_on_full_dataset(
            all_queries, all_ground_truths, batch_size=50
        )
        after_correct = (after_correct1 + after_correct2 + after_correct3) / 3
        after_total = (after_total1 + after_total2 + after_total3) / 3
        after_accuracy = (after_accuracy1 + after_accuracy2 + after_accuracy3) / 3
        prev_after_correct, prev_after_total, prev_after_accuracy = after_correct, after_total, after_accuracy
        
        accuracy_change = after_accuracy - before_accuracy
        change_symbol = "↑" if accuracy_change > 0 else ("↓" if accuracy_change < 0 else "-")
        
        print(f">>> Epoch {epoch+1} accuracy after optimization: {after_correct}/{after_total} = {after_accuracy:.4f} ({after_accuracy*100:.2f}%)")
        print(f">>> Accuracy change: {change_symbol} {abs(accuracy_change):.4f} ({abs(accuracy_change)*100:.2f}%)")
        print(f">>> Current optimization successes: {OPTIMIZE_BETTER}, failures: {OPTIMIZE_WORSE}")
        total_injections = sum(1 for r in results if r and r.get("context_injected", False))
        if total_injections > 0:
            total_references = sum(
                sum(1 for ref in r.get("agent_sample_referenced", {}).values() if ref)
                for r in results if r and r.get("context_injected", False)
            )
            total_agents = sum(
                len(r.get("agent_sample_referenced", {}))
                for r in results if r and r.get("context_injected", False)
            )
            if total_agents > 0:
                print(f">>> Sample reference rate stats: {total_references}/{total_agents} ({total_references/total_agents*100:.1f}%) agents chose to reference samples")
        
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
            print(f">>> Per-agent injection stats: {agent_stats}")
        
        print(f">>> Currently learned intent dimensions ({len(evaluator.dimension_manager.intent_hypotheses)}): {list(evaluator.dimension_manager.intent_hypotheses.keys())}")
        if evaluator.dimension_manager.intent_hypotheses:
            print(">>> Dimension details:")
            for dim_name, dim_info in evaluator.dimension_manager.intent_hypotheses.items():
                evidence = dim_info.get("evidence_count", 0)
                rule_intent = dim_info.get("rule_intent", "") + "..." if len(dim_info.get("rule_intent", "")) > 50 else dim_info.get("rule_intent", "")
                print(f"    - {dim_name}: evidence_count={evidence}, intent={rule_intent}")
        
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
            "intent_hypotheses": evaluator.dimension_manager.intent_hypotheses,
            "pool_state": {
                "correct_samples": evaluator.pool.correct_samples,
                "incorrect_samples": evaluator.pool.incorrect_samples,
                "agent_correct_samples": evaluator.pool.agent_correct_samples,
                "agent_incorrect_samples": evaluator.pool.agent_incorrect_samples,
                "agent_evaluated_flags": evaluator.pool._agent_evaluated_flags,
                "sample_scorer_state": evaluator.pool.sample_scorer.export_state(),
                "dimension_sample_index": evaluator.pool.dimension_sample_index,
                "sample_dimension_tags": evaluator.pool.sample_dimension_tags
            },
            "dimension_manager_state": {
                "similarity_history": evaluator.dimension_manager._similarity_history,
                "history_max_size": evaluator.dimension_manager._history_max_size
            },
            "attribution_details": [],
            "optimize_better": OPTIMIZE_BETTER,
            "optimize_worse": OPTIMIZE_WORSE,
            "resumed_from_epoch": start_epoch if epoch == start_epoch and RESUME_EPOCH is not None else None,
            "token_usage": token_counter.get_summary()
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\nEpoch {epoch+1} complete! Results saved to: {output_file}")
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
    
    token_counter.print_summary()


if __name__ == "__main__":
    run_evaluation()
