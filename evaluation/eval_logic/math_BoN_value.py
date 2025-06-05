import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from framework.register import register_processor
from utils.util import load_json, save_json, timestamped_print
from infer_module.infer_critic import Critic_Service


_cached_critic_service = None  # Global variable to cache the Critic_Service instance

def get_critic_service(model_path: str = None, tensor_parallel_size: int = 1) -> Critic_Service:
    global _cached_critic_service
    if _cached_critic_service is None:
        timestamped_print("CRITIC: Initializing Critic service...")
        try:
            _cached_critic_service = Critic_Service(
                model_path=model_path,
                tensor_parallel_size=tensor_parallel_size,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2"
            )
        except Exception as e:
            timestamped_print(f"CRITIC: Failed to initialize: {e}", "ERROR")
            raise
    return _cached_critic_service

@register_processor('process')
def process_file(args) -> None:
    """
    input_filepath: str, 
    output_filepath: str, 
    reward_path: str, 
    reward_tensor_parallel_size: int
    system_prompt: str
    user_prompt_template: str
    """
    data = load_json(args.input_filepath)
    critic_service = get_critic_service(model_path=args.reward_path, tensor_parallel_size=args.reward_tensor_parallel_size)
    data['rewards'] = []
    data['steps'] = []
    data['conversations'] = []
    data['tag_indices'] = []
    for idd in range(len(data['policy_responses'])):
        messages = [ 
            { "role": "system", "content": args.system_prompt }, 
            { "role": "user", "content": args.user_prompt_template.format(problem=data['problem']) },
            { "role": "assistant", "content": data['policy_responses'][idd] }, 
        ]
        results = critic_service.build_prompt(messages)
        values = []
        tag_indices = []
        for prompt, response_length in results:
            tokens = critic_service.simple_tokenize(prompt)
            tag_indices.append(len(tokens))
        
        values = critic_service.predict_values(prompt, response_length)

        steps = data['policy_responses'][idd].split('\n\n')
        steps[0] = data['problem'] + '\n' + steps[0]
        data['steps'].append(steps)
        data['conversations'].append(messages)
        data['rewards'].append(values)
        data['tag_indices'].append(tag_indices)

    save_json(data, args.output_filepath)
