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
from typing import Any, Dict, List, Tuple


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
    
    def get_stop_reason(self, request_results: List) -> List[List]:
        # Extract stop reason from the request results
        stop_reasons = [
            [
                result.stop_reason
                for result in request_result.outputs
            ]
            for request_result in request_results
        ]
        
        return stop_reasons
    
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
    
    def get_entropys(self, request_results: List) -> List[List]:
        '''get the responses' entropys from the request results'''
        def calculate_entropy(logprob_distribution: Dict[Any, Any]) -> float:
            if not logprob_distribution:
                return 0.0

            entropy = 0.0
            for item in logprob_distribution.values():
                log_prob = item.logprob
                
                # 1. Convert log-probability to actual probability
                # p(x) = e^(log_prob)
                prob = math.exp(log_prob)
                
                # 2. Add to the entropy sum: p(x) * log2(p(x))
                # We check for prob > 0 to avoid math.log2(0) which is undefined.
                if prob > 0:
                    entropy += prob * math.log2(prob)
                    
            return -entropy
        
        entropys = [
            [
                [calculate_entropy(logprob_distribution) for logprob_distribution in result.logprobs]
                for result in request_result.outputs
            ]
            for request_result in request_results
        ]
        
        return entropys

    def get_prompt_tokenIDs(self, request_results: List) -> List[List]:
        prompt_token_ids = [
            request_result.prompt_token_ids
            for request_result in request_results
        ]
        
        return prompt_token_ids

    def get_response_tokenIDs(self, request_results: List) -> List[List]:
        token_ids = [
            [
                result.token_ids
                for result in request_result.outputs
            ]
            for request_result in request_results
        ]
        
        return token_ids
