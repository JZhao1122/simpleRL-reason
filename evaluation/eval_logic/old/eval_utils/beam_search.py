import sys
import os
import copy

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from typing import Dict, List, Optional
from utils.util import timestamped_print
from infer_module.infer_vllm import LLM_Service
from vllm import SamplingParams
from math_verify import parse, verify
from typing import Any


class BeamSearchEnv:
    def __init__(self,
                 problem: str,
                 llm_service: LLM_Service,
                 max_expansions: int,
                 system_prompt: str,
                 user_prompt_template: str = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
                 ):
        self.problem: str = problem
        self.llm_service: LLM_Service = llm_service
        self.max_expansions: int = max_expansions

        self.system_prompt: str = system_prompt
        self.user_prompt_template: str = user_prompt_template

        self.current_assistant_response: str = ""
        self.num_depth: int = 0
        self.is_terminated: bool = False

        self._initial_messages: List[Dict[str, str]] = self._build_initial_messages()

    def _build_initial_messages(self) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_template.format(problem=self.problem)}
        ]

    def reset(self) -> None:
        self.current_assistant_response = ""
        self.num_depth = 0
        self.is_terminated = False

        
    def get_prompt_for_llm(self) -> str:
        messages_for_prompt = copy.deepcopy(self._initial_messages)
        prompt = self.llm_service.build_prompt(messages_for_prompt)
        if self.current_assistant_response != "":
            prompt += self.current_assistant_response
        return prompt

    def generate_expansions(self, sampling_params: SamplingParams) -> List[Dict[str, Any]]:
        if self.is_terminated:
            timestamped_print("Warning: Attempting to expand a terminated environment. Returning empty list.", "WARNING")
            return []

        current_prompt = self.get_prompt_for_llm()
        
        timestamped_print(f"ENV (PID:{os.getpid()}): Generating expansions from prompt (last 50 chars): '...{current_prompt[-50:]}'", "DEBUG")

        llm_results_list = self.llm_service.inference(current_prompt, sampling_params)
        
        expansions = []

        if not llm_results_list:
            timestamped_print("ENV: LLM service returned no results.", "WARNING")
            return []

        request_output = llm_results_list[0]
        
        for completion_output in request_output.outputs:
            expansion_data = {
                "text": completion_output.text,
                "finish_reason": completion_output.finish_reason,
                "logprobs": completion_output.logprobs,
                "cumulative_logprob_of_chunk": completion_output.cumulative_logprob
            }
            expansions.append(expansion_data)
            timestamped_print(f"ENV (PID:{os.getpid()}): Generated expansion chunk: '{completion_output.text[:30]}...', finish: {completion_output.finish_reason}", "DEBUG")
            
        return expansions

    def check_if_terminated(self, llm_output_chunk_for_check: Optional[str] = None, finish_reason_for_check: Optional[str] = None) -> bool:
        if self.is_terminated:
            return True

        if self.num_depth >= self.max_expansions:
            return True

        if finish_reason_for_check == "stop":
            return True

        if llm_output_chunk_for_check:
            chunk_lower = llm_output_chunk_for_check.lower()
            if "[eos]" in chunk_lower or "<|im_end|>" in chunk_lower:
                return True
        
        return False

    def step_update(self, llm_generated_chunk: str, finish_reason: Optional[str]) -> None:
        if self.is_terminated:
            timestamped_print("Warning: Stepping a terminated environment.", "WARNING")
            return

        self.current_assistant_response += llm_generated_chunk
        self.num_depth += 1

        if self.check_if_terminated(llm_generated_chunk, finish_reason):
            self.is_terminated = True
        elif self.num_depth >= self.max_expansions:
            self.is_terminated = True
            timestamped_print(f"Environment terminated due to max_expansions ({self.max_expansions}) reached.", "DEBUG")

    def copy(self) -> "BeamSearchEnv":
        new_env = BeamSearchEnv(
            problem=self.problem,
            llm_service=self.llm_service,
            max_expansions=self.max_expansions,
            system_prompt=self.system_prompt,
            user_prompt_template=self.user_prompt_template
        )
        new_env._initial_messages = copy.deepcopy(self._initial_messages)
        new_env.current_assistant_response = copy.deepcopy(self.current_assistant_response)
        new_env.num_depth = self.num_depth
        new_env.is_terminated = self.is_terminated
        return new_env

    def get_full_response(self) -> str:
        return self.current_assistant_response