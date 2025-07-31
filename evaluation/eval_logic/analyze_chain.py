import sys
import os
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from framework.register import register_processor
from utils.util import load_json, save_json, timestamped_print, random_initialize
from infer_module.infer_vllm import LLM_Service
from vllm import SamplingParams
from math_verify import parse, verify
from typing import List, Any, Dict


_cached_llm_service = None  # Global variable to cache the LLM_Service instance

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

def analyze(prompt: str, llm_service: LLM_Service) -> str:
    local_prompt = prompt + "We can directly obtain that the final answer should be \\boxed{"
    sampling_params = SamplingParams(
        seed=random_initialize(),
        n=1,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        max_tokens=10,  # Maximum number of tokens to generate
        logprobs=20,
        stop=["}"],
        include_stop_str_in_output=False
    )
    results = llm_service.inference(local_prompt, sampling_params)
    answer = llm_service.get_text(results)[0][0].strip()
    return answer

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
    """
    data = load_json(args.input_filepath)
    llm_service = get_llm_service(model_path=args.model_path, tensor_parallel_size=args.tensor_parallel_size)
    messages = [ 
        { "role": "system", "content": args.system_prompt }, 
        { "role": "user", "content": f"{data['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}." }, 
    ]
    prompt = llm_service.build_prompt(messages)
    responses: List[str] = data["policy_responses"]
    data["analyze_res"] = {}
    for idd, res in enumerate(responses):
        data["analyze_res"][idd] = []
        responses_list = res.split("\n\n")
        for ii in range(1, len(responses_list) + 1):
            data["analyze_res"][idd] += [
                analyze( prompt + "\n\n".join(responses_list[:ii]), llm_service )
            ]
    
    save_json(data, args.output_filepath)
