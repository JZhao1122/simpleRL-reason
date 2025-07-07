import sys
import os
import argparse
import queue

current_dir = os.path.dirname(os.path.abspath('/cpfs02/user/liurunze/_/simpleRL-reason/evaluation/eval_logic/test_eval.ipynb'))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from framework.register import register_processor
from utils.util import load_json, save_json, timestamped_print
from infer_module.infer_vllm import LLM_Service
from vllm import SamplingParams
from math_verify import parse, verify

llm_service1 = LLM_Service(model_path="/cpfs02/user/liurunze/hf_models/models--Qwen--Qwen2.5-7B", tensor_parallel_size=1, device='cuda:0')

problem = "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."

messages = [
    { "role": "system", "content": "You are a helpful assistant." }, 
    { "role": "user", "content": f"{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}." }, 
]

sampling_params = SamplingParams(
    n=1,
    temperature=1.0,
    top_k=-1,
    top_p=1.0,
    max_tokens=16384,
    # stop=["\n"],  # Stop sequences for the model
    include_stop_str_in_output=True,  # Include the stop string in the output
    logprobs=20,
    prompt_logprobs=20,
)

llm_service = llm_service1
results = llm_service.inference(llm_service.build_prompt(messages), sampling_params)
# print(llm_service.get_prompt_tokenIDs(results))
# print(llm_service.get_response_tokenIDs(results))
# print(llm_service.get_entropys(results))
# print(llm_service.get_response_tokens(results))

import math
from typing import Dict
from vllm.sequence import Logprob

def normalize_logprobs(logprobs: Dict[int, Logprob]) -> Dict[int, Logprob]:
    """
    Normalizes a dictionary of log-probabilities so that the sum of their
    exponentiated values (the probabilities) equals 1.

    This is done in log-space to prevent numerical underflow/overflow.

    Args:
        logprobs: The dictionary of {token_id: Logprob} to normalize.

    Returns:
        A new dictionary with normalized log-probability values.
    """
    if not logprobs:
        return {}

    # --- Use the Log-Sum-Exp trick for numerical stability ---
    # 1. Get all logprob values
    lp_values = [lp.logprob for lp in logprobs.values()]
    
    # 2. Find the maximum logprob
    max_lp = max(lp_values)
    
    # 3. Calculate sum(exp(lp - max_lp))
    sum_of_exps = sum(math.exp(lp - max_lp) for lp in lp_values)
    
    # 4. The log of the total sum is log(sum(exp(lp))) = max_lp + log(sum(exp(lp - max_lp)))
    log_total_prob = max_lp + math.log(sum_of_exps)

    # --- Create the new normalized dictionary ---
    normalized_dict = {}
    for token_id, original_logprob_obj in logprobs.items():
        # The new log-probability is the original minus the log of the sum
        # This is equivalent to: log( P(i) / sum(P) ) = log(P(i)) - log(sum(P))
        new_logprob = original_logprob_obj.logprob - log_total_prob
        
        normalized_dict[token_id] = Logprob(
            logprob=new_logprob,
            rank=original_logprob_obj.rank,
            decoded_token=original_logprob_obj.decoded_token
        )
        
    return normalized_dict

def norm(normalized_a: Dict[int, Logprob]):
    return math.sqrt(sum([math.exp(2*normalized_a[key].logprob) for key in normalized_a.keys()]))

def calculate_normalized_shared_prob_sum(
    logprobs_a: Dict[int, Logprob],
    logprobs_b: Dict[int, Logprob]
) -> float:
    """
    First, normalizes the log-probabilities in each dictionary. Then, it finds
    tokens present in both and calculates the sum of their joint probabilities.

    The calculation for each shared token is: exp(norm_logprob_a + norm_logprob_b)

    Args:
        logprobs_a: The first dictionary of {token_id: Logprob}.
        logprobs_b: The second dictionary of {token_id: Logprob}.

    Returns:
        A float representing the sum of joint probabilities for shared tokens,
        calculated after normalizing each input distribution.
    """
    # Step 1: Normalize both input dictionaries
    normalized_a = normalize_logprobs(logprobs_a)
    normalized_b = normalize_logprobs(logprobs_b)
    
    # Step 2: Find common keys in the normalized dictionaries
    common_keys = normalized_a.keys() & normalized_b.keys()

    if not common_keys:
        return 0.0

    # Step 3: Calculate the sum of joint probabilities using the normalized values
    total_prob_sum = sum(
        math.exp(normalized_a[key].logprob + normalized_b[key].logprob)
        for key in common_keys
    )/norm(normalized_a)/norm(normalized_b)

    return total_prob_sum

import gc
import ray
import torch
try:
    del llm_service
    # del tokenizer
except Exception as e:
    print(f"Failed to unload model: {e}")
finally:
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
import gc
import ray
import torch
try:
    del llm_service
    # del tokenizer
except Exception as e:
    print(f"Failed to unload model: {e}")
finally:
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
llm_service2 = LLM_Service(model_path="/cpfs02/user/liurunze/_/simpleRL-reason/_outputs/checkpoints/verl-ppo_models--Qwen--Qwen2.5-7B_models--Skywork--Skywork-o1-Open-PRM-Qwen-2.5-7B_simplelr_qwen_level3to5_max_response8192_batch1024_rollout8_klcoef0.0001_entcoef0.001/global_step_90/actor/huggingface", tensor_parallel_size=1, device='cuda:0')

llm_service = llm_service2
prompt = llm_service.build_prompt(messages) + llm_service.get_text(results)[0][0]

sampling_params = SamplingParams(
    n=1,
    temperature=1.0,
    top_k=-1,
    top_p=1.0,
    max_tokens=1,
    # stop=["\n"],  # Stop sequences for the model
    include_stop_str_in_output=True,  # Include the stop string in the output
    logprobs=20,
    prompt_logprobs=20,
)

results_1 = llm_service.inference(prompt, sampling_params)

log1 = (results[0].prompt_logprobs + results[0].outputs[0].logprobs)[1:-1]
log2 = results_1[0].prompt_logprobs[1:]
tokenizer = llm_service.tokenizer
tokens = tokenizer.convert_ids_to_tokens(list(results_1[0].prompt_token_ids))
print(tokens[1:])
print([calculate_normalized_shared_prob_sum(log1[ii], log2[ii]) for ii in range(len(log2))])

