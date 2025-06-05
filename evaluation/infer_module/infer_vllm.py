import re
import math
import io
import signal
from copy import *
from typing import List
from utils.util import timestamped_print, cprint
from contextlib import redirect_stdout
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class LLM_Service:
    def __init__(self, model_path: str, tensor_parallel_size: int):
        # Load the model and tokenizer
        timestamped_print(f"Loading model from {model_path}", level="INFO")
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            enable_chunked_prefill=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        timestamped_print(f"VLLM model loaded successfully", level="INFO")
    
    def build_prompt(self, messages: List) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt
    
    def inference(self, prompt: str, sampling_params: SamplingParams):
        cprint(prompt, "Prompt")
        
        # Perform inference
        request_results = self.model.generate(prompt, sampling_params)
        
        return request_results
    
    def get_text(self, request_results: List) -> List[List]:
        # Extract text from the request results
        text_results = [
            [
                result.text
                for result in request_result.outputs
            ]
            for request_result in request_results
        ]
        
        return text_results

    def get_finish_reason(self, request_results: List) -> List[List]:
        # Extract finish reason from the request results
        finish_reasons = [
            [
                result.finish_reason
                for result in request_result.outputs
            ]
            for request_result in request_results
        ]
        
        return finish_reasons
    
    def get_logprobs(self, request_results: List) -> List[List]:
        # Extract logit probabilities from the request results
        logprobs = [
            [
                result.logprobs
                for result in request_result.outputs
            ]
            for request_result in request_results
        ]
        
        return logprobs