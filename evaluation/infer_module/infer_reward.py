import torch
from typing import List, Dict, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.util import timestamped_print, cprint
from .prm_utils.rm_call import (
    RMRemoteCaller,
    RemoteRewardModelConfig,
    get_prm_special_tokens,
)

class Reward_Service:
    def __init__(self, model_path: str):
        timestamped_print(f"Loading Reward model from {model_path}", level="INFO")
        
        # load model and tokenizer
        rm_model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(rm_model_path, trust_remote_code=True)
        prm_step_tag, step_tag_id, returned_token_ids = get_prm_special_tokens(rm_model_path, self.tokenizer)
        if 'pqm' in rm_model_path:
            prm_format_str = "{question}\n{answer}"
        else:
            prm_format_str = "{question} {answer}"
        rm_config = RemoteRewardModelConfig(
            prm_step_tag=prm_step_tag, 
            format_str=prm_format_str, 
            model_name=rm_model_path, 
            controller_addr="http://localhost:10014",
            step_tag_id=step_tag_id, 
            returned_token_ids=returned_token_ids, 
            rm_serve_type="fastchat", 
            multi_gpu=False,
        )
        self.rm_call = RMRemoteCaller(rm_config, tokenizer=self.tokenizer)

        timestamped_print("Reward model loaded successfully", level="INFO")

    def predict_rewards(self, 
                      problem: str,
                      steps: List[str]) -> List:
        '''from Runze's implementation'''
        qa_pairs = [(problem, ' ки\n'.join(steps))]
        step_scores, token_scores = self.rm_call(qa_pairs, verbose=True)
        return step_scores, token_scores
    
    def BS_predict_rewards(self, 
                      prompt_ids: List,
                      response_ids: List,
                      past_key_values: Any) -> List:
        '''Specialized method for Beam Search'''
        step_scores, token_scores, current_past_key_values = self.rm_call(
            qa_pairs=None, 
            verbose=True,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
        )
        assert len(token_scores) == len(prompt_ids) + len(response_ids), \
            f"Expected {len(prompt_ids) + len(response_ids)} token scores, got {len(token_scores)}"
        
        return token_scores[-len(response_ids):], current_past_key_values
    
    # def build_prompt(self, messages: List[List[Dict]]) -> Dict[str, torch.Tensor]:
    #     try:
    #         user_prompt = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    #     except:
    #         user_prompt = '\n'.join([m['content'] for m in messages[:-1]])
    #     responses = messages[-1]['content'].split("\n\n")
    #     results = [
    #         (
    #             user_prompt + responses[0], 
    #             len(
    #                 self.tokenizer.encode(
    #                     responses[0],
    #                     add_special_tokens=False
    #                 )
    #             )
    #         )
    #     ]
    #     for response in responses[1:]:
    #         results.append(
    #             (
    #                 results[-1][0] + self.step_tag + response, 
    #                 len(
    #                     self.tokenizer.encode(
    #                         (results[-1][0] + self.step_tag + response)[len(user_prompt):],
    #                         add_special_tokens=False
    #                     )
    #                 )
    #             )
                
    #         )

    #     return results

    # def simple_tokenize(self, prompt: str) -> torch.Tensor:
    #     inputs = self.tokenizer(
    #         prompt,
    #         return_tensors="pt"
    #     ).to("cuda")
    #     return inputs["input_ids"][0]
