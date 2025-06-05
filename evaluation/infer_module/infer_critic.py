import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification
from utils.util import timestamped_print, cprint

class Critic_Service:
    def __init__(self, 
                 model_path: str, 
                 tensor_parallel_size: int = 1,
                 torch_dtype: torch.dtype = torch.float16,
                 attn_implementation: str = "flash_attention_2"):
        timestamped_print(f"Loading Critic model from {model_path}", level="INFO")
        
        # load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            # tensor_parallel_size=tensor_parallel_size,
            num_labels=1  # value prediction
        ).eval().cuda()
        
        timestamped_print("Critic model loaded successfully", level="INFO")

    def build_prompt(self, messages: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        user_prompt = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
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
                    results[-1][0] + "\n\n" + response, 
                    len(
                        self.tokenizer.encode(
                            (results[-1][0] + "\n\n" + response)[len(user_prompt):],
                            add_special_tokens=False
                        )
                    )
                )
                
            )

        return results

    def simple_tokenize(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        ).to("cuda")
        return inputs["input_ids"][0]

    def predict_values(self, 
                      prompt: str,
                      response_length: int = 512) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            ).to("cuda")
            outputs = self.model(
                input_ids=inputs["input_ids"],
                # attention_mask=inputs["attention_mask"],
                # position_ids=inputs.get("position_ids", None)
            )
            
            values = outputs.logits.squeeze(-1)  # (batch_size, seq_len)
            
            if response_length > 0:
                values = values[:, -response_length:]
            
        return values[0].tolist()

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
    #         critic_score = float(values[0, -1].sigmoid())
    #     except Exception as e:
    #         timestamped_print(f"PRM: Critic evaluation failed: {e}", "WARNING")
    #         critic_score = 0.5
        
    #     return critic_score