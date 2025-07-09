import re
import math
import io
import signal
import numpy as np
from collections import Counter
from copy import *
from typing import List
from utils.util import timestamped_print, cprint
from contextlib import redirect_stdout
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from typing import Any, Dict, List, Tuple
from .infer_reward import Reward_Service


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
    
    def token_level_inference(
            self, 
            prompt: str, 
            sampling_params: SamplingParams, 
            reward_mode: str = "none", # 'token'
            decode_mode: str = "none",  # 'token' or 'entropy'
            entropy_threshold: float = 0.02,
            reward_service: Reward_Service = None,
        ):
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
        prompt_token_ids = []
        response_token_ids = []
        # Perform token-level inference
        for i in range(max_tokens):
            print('*', end='')
            # Generate the next token
            sampling_params.max_tokens = 1
            request_results = self.inference(prompt, sampling_params, use_tqdm=False, verbose=False)

            if i == 0:
                prompt_token_ids = self.get_prompt_tokenIDs(request_results)[0]
            
            # get the text, token and entropy from the request results
            text = self.get_text(request_results)[0][0]
            token = self.get_response_tokens(request_results)[0][0]
            entropy = self.get_entropys(request_results)[0][0]
            token_id = self.get_response_tokenIDs(request_results)[0][0]

            if not text or not token:
                print(" <end> No text or token generated, breaking the loop.")
                # If no text or token is generated, break the loop
                break

            if text in sampling_params.stop:
                print(" <stop> Stop token reached, breaking the loop.")
                # If the stop token is reached, break the loop
                break
            
            if decode_mode != "none":
                result = self.customized_decode(
                    prompt_ids = prompt_token_ids + response_token_ids,
                    request_results=request_results,
                    decode_mode=decode_mode,
                    entropy=entropy, 
                    sampling_params=sampling_params,
                    entropy_threshold=entropy_threshold,
                    reward_service=reward_service,
                )
                if result:
                    text, token, token_id = result

            # Append the generated token to the content
            content += text
            tokens += token
            entropies += entropy
            response_token_ids += token_id
            prompt += text
        
        if reward_mode == "token":
            # If reward mode is 'token', calculate the token reward
            origin_token_rewards = reward_service.BS_predict_rewards(
                prompt_ids=prompt_token_ids,
                response_ids=response_token_ids
            )
            assert len(token_rewards) == len(tokens), "The length of token rewards should match the length of tokens."
            token_rewards = origin_token_rewards
        
        entropy_indices = [i for i, entropy in enumerate(entropies) if entropy >= entropy_threshold]

        return {
            "content": content,
            "tokens": tokens,
            "token_rewards": token_rewards,
            "entropies": entropies,
            "entropy_indices": entropy_indices,
            "prompt_token_ids": prompt_token_ids,
            "response_token_ids": response_token_ids,
        }
    
    def customized_decode(
            self, 
            prompt_ids: List[int], 
            request_results: Any, 
            decode_mode: str,
            entropy: float,
            entropy_threshold: float, 
            sampling_params: SamplingParams, 
            reward_service: Reward_Service = None,
        ): # -> text, token, token_id
        if decode_mode == 'entropy':
            if entropy < entropy_threshold:
                return None
        
        '''
        {
            15: Logprob(logprob=-0.04350040480494499, rank=1, decoded_token='0'), 
            16: Logprob(logprob=-5.1685004234313965, rank=2, decoded_token='1'), 
            8948: Logprob(logprob=-17.641645431518555, rank=17952, decoded_token='system'), 
        }
        '''
        log_distribution = self.get_logprobs(request_results)[0][0][0]

        print("Log distribution:", log_distribution)

        new_log_distribution = {}
        id2token = {}
        for key, value in log_distribution.items():
            reward = reward_service.BS_predict_rewards(
                prompt_ids=prompt_ids,
                response_ids=[int(key)]
            )[0]

            print(f"Key: {key}, Logprob: {value.logprob}, Decoded Token: {value.decoded_token}, Reward: {reward}")

            new_log_distribution[key] = math.exp(value.logprob) * reward
            id2token[key] = value.decoded_token
        
        print("New log distribution:", new_log_distribution)

        def norm_sample(scores_dict: dict):
            if not scores_dict:
                raise ValueError("Input dictionary cannot be empty.")
            
            items = list(scores_dict.keys())
            scores = np.array(list(scores_dict.values()), dtype=np.float64)

            if len(items) == 1:
                return items[0]

            # --- 1. Normalization (Numerically stable) ---
            probabilities = scores / np.sum(scores)

            # --- 2. Weighted Random Sampling ---
            sampled_item = np.random.choice(items, p=probabilities)

            return sampled_item
        
        token_id = norm_sample(new_log_distribution)
        token = id2token[token_id]
        text = token

        print(f"Sampled token: {token}, Token ID: {token_id}")

        return text, [token], [token_id]

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
