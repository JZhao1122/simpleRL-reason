import sys
import os
import argparse
import queue

current_dir = os.path.dirname(os.path.abspath('/cpfs02/user/liurunze/_/simpleRL-reason/evaluation/eval_logic/test_eval.ipynb'))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(current_dir)
sys.path.append(root_dir)
from eval_utils.prob import *
from eval_utils.vllm_util import *
from infer_module.infer_vllm import LLM_Service
from infer_module.infer_reward import Reward_Service
from utils.util import load_json, save_json, timestamped_print, cprint
from vllm import SamplingParams

# global variable to cache the LLM_Service instance
_cached_reward_service = None  # Global variable to cache the Reward_Service instance
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

reward_service = get_reward_service(model_path="/cpfs02/user/liurunze/_/simpleRL-reason/_outputs/checkpoints/verl-ppo_models--Qwen--Qwen2.5-7B_models--Skywork--Skywork-o1-Open-PRM-Qwen-2.5-7B_simplelr_qwen_level3to5_max_response8192_batch1024_rollout8_klcoef0.0001_entcoef0.001/global_step_90/critic/huggingface")

# load the prompt
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

# load the base model
# llm_service1 = LLM_Service(model_path="/cpfs02/user/liurunze/hf_models/models--Qwen--Qwen2.5-7B", tensor_parallel_size=1, device='cuda:0')
llm_service1 = LLM_Service(model_path="/cpfs02/user/liurunze/_/simpleRL-reason/_outputs/checkpoints/verl-ppo_models--Qwen--Qwen2.5-7B_models--Skywork--Skywork-o1-Open-PRM-Qwen-2.5-7B_simplelr_qwen_level3to5_max_response8192_batch1024_rollout8_klcoef0.0001_entcoef0.001/global_step_90/actor/huggingface", tensor_parallel_size=1, device='cuda:0')
llm_service = llm_service1
results = llm_service.inference(llm_service.build_prompt(messages), sampling_params)
# print(llm_service.get_prompt_tokenIDs(results))
# print(llm_service.get_response_tokenIDs(results))
# print(llm_service.get_entropys(results))
# print(llm_service.get_response_tokens(results))

# unload the model to free up memory
import gc
import ray
import torch

"""
Unload the model and clear resources.
"""
try:
    del llm_service
    # if tokenizer:
    #     del tokenizer
except Exception as e:
    print(f"Failed to unload model: {e}")
finally:
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
try:
    del llm_service1
    # if tokenizer:
    #     del tokenizer
except Exception as e:
    print(f"Failed to unload model: {e}")
finally:
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()


# load the RL model
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

# print the tokens
print(tokens[1:])

# print the log probabilities
print([calculate_normalized_shared_prob_sum(log1[ii], log2[ii]) for ii in range(len(log2))])

# print the entropys
print(llm_service.get_prompt_entropys(results_1)[0][1:])

# calculate the values
prompt_ids = llm_service.get_prompt_tokenIDs(results)[0]
response_ids_list = llm_service.get_response_tokenIDs(results)[0]
token_rewards = reward_service.BS_predict_rewards(
    prompt_ids=[prompt_ids[1]],
    response_ids=prompt_ids[1:] + response_ids_list[0]
)
print(token_rewards)
