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
    def __init__(self, model_path: str, tensor_parallel_size: int, device: str):
        # Load the model and tokenizer
        timestamped_print(f"Loading model from {model_path}", level="INFO")
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            enable_chunked_prefill=True,
            device=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        timestamped_print(f"VLLM model loaded successfully", level="INFO")
    
    def build_prompt(self, messages: List) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt
    
    def inference(self, prompt: str, sampling_params: SamplingParams, use_tqdm: bool = True, verbose: bool = True):
        if verbose:
            cprint(prompt, "Prompt")
        
        # Perform inference
        request_results = self.model.generate(prompt, sampling_params, use_tqdm=use_tqdm)
        
        return request_results
    
    def token_level_inference(self, prompt: str, sampling_params: SamplingParams):
        '''
        custom token-level inference function for vLLM
        This function generates tokens one by one, allowing for more control over the generation process.
        The sampling_params.n should be set to 1.
        Return is Dict{
            "content": String,  # The generated text content
            "tokens": List of generated tokens,
            "token_rewards": List of rewards for each token,
            "entropies": List of entropy for each token,
        }
        '''
        assert sampling_params.n == 1, "For token-level inference, sampling_params.n should be set to 1."
        cprint(prompt, "Prompt")
        max_tokens = sampling_params.max_tokens

        content = ""
        tokens = []
        token_rewards = []
        entropies = []
        # Perform token-level inference
        for _ in range(max_tokens):
            print('*')
            # Generate the next token
            sampling_params.max_tokens = 1
            request_results = self.inference(prompt, sampling_params, use_tqdm=False, verbose=False)
            
            # get the text, token and entropy from the request results
            text = self.get_text(request_results)[0][0]
            token = self.get_response_tokens(request_results)[0][0]
            entropy = self.get_entropys(request_results)[0][0]

            # Append the generated token to the content
            content += text
            tokens += token
            entropies += entropy
            prompt += text
        
        return {
            "content": content,
            "tokens": tokens,
            "token_rewards": token_rewards,
            "entropies": entropies
        }

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
    
    def get_prompt_entropys(self, request_results: List) -> List[List]:
        '''get the prompts' entropys from the request results'''
        def calculate_entropy(logprob_distribution) -> float:
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
            [calculate_entropy(logprob_distribution) for logprob_distribution in request_result.prompt_logprobs]
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
                list(result.token_ids)
                for result in request_result.outputs
            ]
            for request_result in request_results
        ]
        
        return token_ids
    
    def get_response_tokens(self, request_results: List) -> List[List]:
        tokens = [
            [
                self.tokenizer.convert_ids_to_tokens(list(result.token_ids))
                for result in request_result.outputs
            ]
            for request_result in request_results
        ]
        
        return tokens
