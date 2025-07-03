import sys
import os
import heapq
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from framework.register import register_processor
from utils.util import load_json, save_json, timestamped_print
from infer_module.infer_vllm import LLM_Service
from infer_module.infer_critic import Critic_Service
from vllm import SamplingParams
from math_verify import parse, verify
from eval_utils.beam_search import BeamSearchEnv
from typing import Any, Dict, List, Tuple


_cached_llm_service = None  # Global variable to cache the LLM_Service instance
_cached_critic_service = None  # Global variable to cache the Critic_Service instance

def get_llm_service(model_path: str = None, tensor_parallel_size: int = 1) -> LLM_Service:
    """
    Get a singleton instance of the LLM_Service.
    If the instance is already created, return it.
    """
    global _cached_llm_service
    if _cached_llm_service is None:
        timestamped_print("MODEL: First call to get_llm_service. Attempting to load model...")
        try:
            _cached_llm_service = LLM_Service(model_path=model_path, tensor_parallel_size=tensor_parallel_size)
            timestamped_print("MODEL: Model loaded successfully via get_llm_service.")
        except Exception as e:
            timestamped_print(f"MODEL: Failed to load model via get_llm_service: {e}", "ERROR")
            raise ValueError(f"Failed to load model: {e}")
    else:
        timestamped_print("MODEL: Model loaded successfully via get_llm_service.")
    return _cached_llm_service

def get_critic_service(model_path: str = None, tensor_parallel_size: int = 1) -> Critic_Service:
    global _cached_critic_service
    if _cached_critic_service is None:
        timestamped_print("CRITIC: Initializing Critic service...")
        try:
            _cached_critic_service = Critic_Service(
                model_path=model_path,
                tensor_parallel_size=tensor_parallel_size,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2"
            )
        except Exception as e:
            timestamped_print(f"CRITIC: Failed to initialize: {e}", "ERROR")
            raise
    return _cached_critic_service


@register_processor
def process_file(args) -> None:
    """
    beam_size = args.beam_size
    max_expansions = args.max_expansions
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    max_tokens_per_step = args.max_tokens_per_step
    stop_sequences = args.stop_sequences_beam
    system_prompt = args.system_prompt
    user_prompt_template = args.user_prompt_template
    """
    timestamped_print(f"BEAM_PROCESSOR: Starting file {args.input_filepath} with Beam Search.")
    data = load_json(args.input_filepath)
    problem = data.get("problem")
    answer = data.get("answer")

    if not problem:
        timestamped_print(f"BEAM_PROCESSOR: 'problem' not found in {args.input_filepath}. Skipping.", "ERROR")
        save_json({"error": "Problem not found in input file."}, args.output_filepath)
        return

    llm = get_llm_service(model_path=args.model_path, tensor_parallel_size=args.tensor_parallel_size)
    critic = get_critic_service(model_path=args.reward_path, tensor_parallel_size=args.reward_tensor_parallel_size)

    beam_size = args.beam_size
    max_expansions = args.max_expansions
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    max_tokens_per_step = args.max_tokens_per_step
    stop_sequences = args.stop_sequences_beam
    system_prompt = args.system_prompt
    user_prompt_template = args.user_prompt_template

    initial_env = BeamSearchEnv(
        problem=problem,
        llm_service=llm,
        max_expansions=max_expansions,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template
    )

    current_beams: List[Tuple[float, BeamSearchEnv]] = [(0.0, initial_env.copy())]
    completed_beams: List[Tuple[float, BeamSearchEnv]] = []

    for step in range(max_expansions):
        timestamped_print(f"PRM_BEAM_PROCESSOR: Expansion step {step + 1}/{max_expansions}, Current active beams: {len(current_beams)}")
        if not current_beams:
            timestamped_print("PRM_BEAM_PROCESSOR: No active beams left to expand.", "INFO")
            break

        next_candidate_beams_heap = []

        for neg_parent_prm_score, parent_env in current_beams:
            if parent_env.is_terminated:
                is_already_completed = any(parent_env is c_env for _, c_env in completed_beams)
                if not is_already_completed:
                    heapq.heappush(completed_beams, (neg_parent_prm_score, parent_env))
                continue

            step_sampling_params = SamplingParams(
                n=beam_size,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens_per_step,
                stop=stop_sequences
            )

            try:
                expansions = parent_env.generate_expansions(step_sampling_params)
                if not expansions:
                    timestamped_print(f"PRM_BEAM_PROCESSOR: No expansions generated for a beam. Path: '...{parent_env.get_full_response()[-50:]}'", "WARNING")

                    current_prm_score = critic.prm_function(parent_env.get_full_response())
                    heapq.heappush(completed_beams, (-current_prm_score, parent_env))
                    continue
            except Exception as e:
                timestamped_print(f"PRM_BEAM_PROCESSOR: Error during LLM expansion: {e}", "ERROR")
                
                heapq.heappush(completed_beams, (float('inf'), parent_env))
                continue

            for expansion_data in expansions:
                new_env = parent_env.copy()
                new_env.step_update(expansion_data['text'], expansion_data['finish_reason'])

                current_full_text = new_env.get_full_response()
                prm_score_for_this_beam = critic.prm_function(current_full_text)

                beam_sort_key = -prm_score_for_this_beam

                if new_env.is_terminated:
                    heapq.heappush(completed_beams, (beam_sort_key, new_env))
                else:
                    heapq.heappush(next_candidate_beams_heap, (beam_sort_key, new_env))

        current_beams = heapq.nsmallest(beam_size, next_candidate_beams_heap)
    
    # in case the last beam is not terminated
    for neg_score_approx, env_instance in current_beams:
        if not env_instance.is_terminated:
            env_instance.is_terminated = True
        
        final_prm_score = critic.prm_function(env_instance.get_full_response())
        is_already_completed = any(env_instance is c_env for _, c_env in completed_beams)
        if not is_already_completed:
            heapq.heappush(completed_beams, (-final_prm_score, env_instance))


    sorted_completed_beams = sorted(completed_beams, key=lambda item: item[0])

    output_results = []
    timestamped_print(f"PRM_BEAM_PROCESSOR: Beam search finished. Found {len(sorted_completed_beams)} completed paths.")
    
    for i, (neg_prm_score, env) in enumerate(sorted_completed_beams[:beam_size]):
        response_text = env.get_full_response()
        is_correct = False
        if answer:
            try:
                is_correct = verify(parse(response_text), parse(f"\\boxed{{{answer}}}"))
            except Exception as e:
                timestamped_print(f"PRM_BEAM_PROCESSOR: Error during verification for '{response_text[:50]}...': {e}", "WARNING")
        
        output_results.append({
            "rank": i + 1,
            "response": response_text,
            "prm_score": -neg_prm_score,
            "is_correct": is_correct,
            "num_expansions": env.num_depth,
            "is_terminated_by_eos_or_stop": env.is_terminated and env.num_depth < max_expansions
        })

    data['prm_beam_search_results'] = output_results
    if output_results:
        data['policy_responses'] = [output_results[0]['response']]
        data['correctness'] = [output_results[0]['is_correct']]
        data['finish_reason'] = "prm_beam_search_completed"
    else:
        data['policy_responses'] = ["No complete path found by PRM beam search."]
        data['correctness'] = [False]
        data['finish_reason'] = "prm_beam_search_failed"


    save_json(data, args.output_filepath)
    timestamped_print(f"PRM_BEAM_PROCESSOR: Results saved to {args.output_filepath}")