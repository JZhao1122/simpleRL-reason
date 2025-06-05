import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.util import timestamped_print, cprint

class Reward_Service:
    def __init__(self, 
                 model_path: str,
                 tensor_parallel_size: int,
                 good_token: str,
                 bad_token: str,
                 step_tag: str):
        timestamped_print(f"Loading Reward model from {model_path}", level="INFO")
        
        # load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
        ).eval().cuda()
        self.good_token = good_token
        self.bad_token = bad_token
        self.step_tag = step_tag
        self.candidate_tokens = self.tokenizer.encode(f"{good_token} {bad_token}")[1:] # [648, 387]
        self.step_tag_id = self.tokenizer.encode(f"{step_tag}")[-1] # 12902
        
        timestamped_print("Reward model loaded successfully", level="INFO")

    def build_prompt(self, messages: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        try:
            user_prompt = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        except:
            user_prompt = '\n'.join([m['content'] for m in messages[:-1]])
        responses = messages[-1]['content'].split("\n\n")
        results = [
            (
                user_prompt + responses[0], 
                len(
                    self.tokenizer.encode(
                        responses[0],
                        add_special_tokens=False
                    )
                )
            )
        ]
        for response in responses[1:]:
            results.append(
                (
                    results[-1][0] + self.step_tag + response, 
                    len(
                        self.tokenizer.encode(
                            (results[-1][0] + self.step_tag + response)[len(user_prompt):],
                            add_special_tokens=False
                        )
                    )
                )
                
            )

        return results

    def simple_tokenize(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to("cuda")
        return inputs["input_ids"][0]

    def predict_values(self, 
                      prompt: str,
                      response_length: int = 512) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            ).to("cuda")
            with torch.no_grad():
                logits = self.model(inputs['input_ids']).logits[:,:,self.candidate_tokens]
                scores = logits.softmax(dim=-1)[:,:,0][0]
                step_scores = scores[inputs['input_ids'][0] == self.step_tag_id]
                # print(step_scores)
            
        return scores.tolist(), step_scores.tolist()

    # def prm_function(self, prompt: str) -> float:
    #     try:
    #         inputs = self.tokenizer(
    #             prompt,
    #             padding=True,
    #             truncation=True,
    #             max_length=2048,
    #             return_tensors="pt"
    #         ).to("cuda")
    #         values = self.predict_values(inputs, response_length=1)
    #         Reward_score = float(values[0, -1].sigmoid())
    #     except Exception as e:
    #         timestamped_print(f"PRM: Reward evaluation failed: {e}", "WARNING")
    #         Reward_score = 0.5
        
    #     return Reward_score