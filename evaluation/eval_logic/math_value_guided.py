import sys
import os
import argparse
import importlib

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from framework.register import register_processor
from utils.util import load_json, save_json, timestamped_print, random_initialize
from infer_module.infer_vllm import LLM_Service
from infer_module.infer_reward import Reward_Service
from vllm import SamplingParams
from math_verify import parse, verify


_cached_llm_service = None  # Global variable to cache the LLM_Service instance
_cached_reward_service = None  # Global variable to cache the Reward_Service instance

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

def get_reward_service(model_path: str = None) -> Reward_Service:
    global _cached_reward_service
    if _cached_reward_service is None:
        timestamped_print("REWARD: Initializing reward service...")
        try:
            _cached_reward_service = Reward_Service(
                model_path=model_path
            )
        except Exception as e:
            timestamped_print(f"REWARD: Failed to initialize: {e}", "ERROR")
            raise
    return _cached_reward_service

@register_processor('check_finish')
def check_finish(args: argparse.Namespace, output_filepath: str) -> bool:
    try:
        record = load_json(output_filepath)
    except Exception as e:
        timestamped_print(f"Error loading JSON file {output_filepath}: {e}", "ERROR")
        return False
    if len(record['policy_responses']) < args.num:
        timestamped_print(f"Not enough responses in {output_filepath}. Expected {args.num}, got {len(record['policy_responses'])}.", "WARNING")
        return False
    return True

@register_processor('process')
def process_file(args) -> None:
    """
    input_filepath: str, 
    output_filepath: str, 
    model_path: str, 
    tensor_parallel_size: int
    num: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    system_prompt: str
    args.reward_mode,
    args.decode_mode,
    args.max_candidates,
    args.entropy_threshold,
    """
    data = load_json(args.input_filepath)

    llm_service = get_llm_service(model_path=args.model_path, tensor_parallel_size=args.tensor_parallel_size)
    reward_service = get_reward_service(model_path=args.reward_path)

    messages = [ 
        { "role": "system", "content": args.system_prompt }, 
        { "role": "user", "content": f"{data['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}." }, 
    ]

    if 'policy_responses' not in data:
        data['policy_responses'] = []
        data['correctness'] = []
        data['tokens'] = []
        data['token_rewards'] = []
        data['entropies'] = []
        data['entropy_indices'] = []
    
    for i in range(args.num-len(data['policy_responses'])):
        sampling_params = SamplingParams(
            seed=random_initialize(),
            n=1,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,  # Maximum number of tokens to generate
            logprobs=20,
            prompt_logprobs=20,
        )
        combine_prob = importlib.import_module("eval_logic.eval_utils.combine_probs")
        combine_prob = getattr(combine_prob, args.combine_prob)

        prompt = llm_service.build_prompt(messages)
        result = llm_service.token_level_inference(
            prompt, 
            sampling_params,
            reward_mode = args.reward_mode,
            decode_mode = args.decode_mode,
            max_candidates = args.max_candidates,
            entropy_threshold=args.entropy_threshold,
            combine_prob=combine_prob, 
            reward_service=reward_service
        )

        data['policy_responses'].append(result['content'])
        data['tokens'].append(result['tokens'])
        data['token_rewards'].append(result['token_rewards'])
        data['entropies'].append(result['entropies'])
        data['entropy_indices'].append(result['entropy_indices'])
        data['correctness'].append(
            verify(
                parse(result['content']), 
                parse(f"\\boxed{{{data['answer']}}}"),
            )
        )

        save_json(data, args.output_filepath)
