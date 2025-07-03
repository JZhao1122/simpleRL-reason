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

llm_service = LLM_Service(model_path="/cpfs02/user/liurunze/hf_models/DeepSeek-R1-Distill-Qwen-1.5B", tensor_parallel_size=1)

problem = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$"

messages = [
    { "role": "system", "content": "You are a helpful assistant." }, 
    { "role": "user", "content": f"{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}." }, 
]

sampling_params = SamplingParams(
    n=1,
    temperature=0.7,
    top_k=-1,
    top_p=1.0,
    max_tokens=16384,
    stop=["\n"],  # Stop sequences for the model
    include_stop_str_in_output=True,  # Include the stop string in the output
    logprobs=5,
)
results = llm_service.inference(llm_service.build_prompt(messages), sampling_params)
print(llm_service.get_prompt_tokenIDs(results))
print(llm_service.get_response_tokenIDs(results))
print(llm_service.get_entropys(results))
# sampling_params = SamplingParams(
#     n=1,
#     temperature=0.7,
#     top_k=-1,
#     top_p=1.0,
#     max_tokens=16384,
#     # stop=["\n"],  # Stop sequences for the model
#     include_stop_str_in_output=True,  # Include the stop string in the output
# )
# results = llm_service.inference(llm_service.build_prompt(messages), sampling_params)
# print(results[0].outputs[0].finished)
# print(results[0].outputs[0].finish_reason)
# print(results[0].outputs[0].stop_reason)
