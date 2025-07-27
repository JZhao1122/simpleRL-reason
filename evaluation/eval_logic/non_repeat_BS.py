import sys
import os
import argparse
import importlib
import random
import copy

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from framework.register import register_processor
from utils.util import load_json, save_json, timestamped_print, random_initialize
from infer_module.infer_vllm import LLM_Service
from vllm import SamplingParams
from math_verify import parse, verify
from collections import defaultdict
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

def single2multi(n: int, llm_service: LLM_Service, prompt: str, sampling_params: SamplingParams) -> List[str]:
    """
    Convert a single prompt to multiple new_prompts by sampling n times.
    """
    local_sampling_params = copy.deepcopy(sampling_params)
    local_sampling_params.n = n
    results = llm_service.inference(prompt, local_sampling_params)
    policy_responses = llm_service.get_text(results)[0]
    finish_reasons = llm_service.get_finish_reason(results)[0]
    stop_reasons = llm_service.get_stop_reason(results)[0]
    return [prompt + response for response in policy_responses], finish_reasons, stop_reasons

def multi2multi(k: int, n: int, llm_service: LLM_Service, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
    """
    Convert k prompts to n new_prompts.
    """
    k = len(prompts)

    # Handle edge cases where generation is not possible or needed.
    if k == 0 or n == 0:
        return []

    # --- Distribute n tasks among k prompts as evenly as possible ---
    # Each prompt gets a base number of samples to generate.
    base_samples_per_prompt = n // k
    
    # The first 'remainder' prompts will generate one extra sample to reach the total of n.
    remainder = n % k
    
    # Create a list that defines how many samples each prompt will generate.
    # For example, for n=10 prompts from k=3 sources, the distribution will be [4, 3, 3].
    samples_distribution = [base_samples_per_prompt + 1] * remainder + \
                           [base_samples_per_prompt] * (k - remainder)

    all_new_prompts = []
    all_finish_reasons = []
    all_stop_reasons = []
    # Use zip to iterate through each prompt and its assigned sample count.
    for prompt, num_samples in zip(prompts, samples_distribution):
        # Only call the LLM if we need to generate samples for this prompt.
        if num_samples > 0:
            # Use the provided single2multi function to generate continuations.
            newly_generated_prompts, finish_reasons, stop_reasons = single2multi(
                n=num_samples,
                llm_service=llm_service,
                prompt=prompt,
                sampling_params=sampling_params
            )
            # Add the results to our final list.
            all_new_prompts.extend(newly_generated_prompts)
            all_finish_reasons.extend(finish_reasons)
            all_stop_reasons.extend(stop_reasons)

    return all_new_prompts, all_finish_reasons, all_stop_reasons

def cluster_and_filter_prompts(
    prompts: List[str], 
    labels: List[Any],
    selection_method: str = 'random'
) -> List[str]:
    """
    Clusters prompts based on their corresponding labels and selects one
    representative prompt from each cluster.
    """
    if len(prompts) != len(labels):
        raise ValueError("The number of prompts must be equal to the number of labels.")

    # Step 1: Cluster the prompts by their label.
    # A defaultdict(list) automatically creates an empty list for a new key.
    clusters: Dict[Any, List[str]] = defaultdict(list)
    for label, prompt in zip(labels, prompts):
        clusters[label].append(prompt)

    # Step 2: Filter one prompt from each cluster.
    filtered_prompts: List[str] = []
    
    # Iterate through the lists of prompts in each cluster.
    for prompt_list in clusters.values():
        if not prompt_list:
            continue # Skip empty clusters, though this is unlikely.
            
        # Select one prompt based on the chosen method.
        if selection_method == 'first':
            # Simple and deterministic: always take the first one.
            filtered_prompts.append(prompt_list[0])
        elif selection_method == 'random':
            # Useful for getting a different sample each time.
            filtered_prompts.append(random.choice(prompt_list))
        else:
            raise ValueError(f"Unknown selection_method: '{selection_method}'. Use 'first' or 'random'.")

    return filtered_prompts

def filter(prompts: List[str], llm_service: LLM_Service, strategy: str) -> List[str]:
    """
    Filter the similar prompts.
    """
    if strategy == "prompt":
        local_prompts = copy.deepcopy(prompts)
        local_prompts = [local_prompt + "We can directly obtain that the final answer should be \\boxed{" 
                         for local_prompt in local_prompts]
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
        results = llm_service.inference(local_prompts, sampling_params)
        answers = [result[0].strip() for result in llm_service.get_text(results)]
        filtered_prompts = cluster_and_filter_prompts(
            prompts=prompts, 
            labels=answers, 
            selection_method='random'
        )
        return filtered_prompts
    else:
        raise ValueError(f"Unknown filter strategy: {strategy}")


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
    filter_strategy
    """
    data = load_json(args.input_filepath)

    llm_service = get_llm_service(model_path=args.model_path, tensor_parallel_size=args.tensor_parallel_size)

    messages = [ 
        { "role": "system", "content": args.system_prompt }, 
        { "role": "user", "content": f"{data['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}." }, 
    ]
    sampling_params = SamplingParams(
        n=args.num,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,  # Maximum number of tokens to generate
        stop=["\n\n"],
        include_stop_str_in_output=True,
    )
    prompts = [llm_service.build_prompt(messages)]
    initial_length = len(prompts[0])
    current_num = args.num
    final_prompts = []
    trajectory = {}
    idd = 0
    while len(prompts) > 0:
        idd += 1
        timestamped_print(f"Iteration {idd}: Current number of prompts: {len(prompts)}")
        sampling_params.max_tokens = args.max_tokens - min([len(prompt) for prompt in prompts]) + initial_length
        trajectory[f"turn {idd}"] = {
            "num_prompts": len(prompts),
            "max_tokens": sampling_params.max_tokens,
            "prompts": prompts,
        }
        new_prompts, finish_reasons, stop_reasons = multi2multi(
            k=len(prompts),
            n=current_num,
            llm_service=llm_service,
            prompts=prompts,
            sampling_params=sampling_params
        )
        prompts = []
        for prompt, finish_reason, stop_reason in zip(new_prompts, finish_reasons, stop_reasons):
            if finish_reason == 'length' or stop_reason is None:
                final_prompts.append(prompt)
                current_num -= 1
                continue
            prompts.append(prompt)

        if idd < args.step_threshold:
            # If we are still in the initial steps, we should not filter.
            continue
        
        filtered_prompts = filter(
            prompts=prompts, 
            llm_service=llm_service, 
            strategy=args.filter_strategy
        )
        if len(filtered_prompts) < args.filter_proportation_threshold * len(prompts):
            # If the number of filtered prompts is less than the threshold
            prompts = prompts
        else:
            prompts = filtered_prompts
    
    data['policy_responses'] = [prompt[initial_length:] for prompt in final_prompts]
    data['correctness'] = [
        verify(
            parse(response), 
            parse(f"\\boxed{{{data['answer']}}}"),
        )
        for response in data['policy_responses']
    ]
    data['trajectory'] = trajectory
    save_json(data, args.output_filepath)
