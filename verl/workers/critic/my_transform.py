from skywork_o1_prm_inference.prm_model import PRM_MODEL
from transformers import AutoModel

# Path to your ORIGINAL sharded model (e.g., from the Hugging Face Hub)
original_model_path = "/cpfs02/user/liurunze/_/simpleRL-reason/_outputs/checkpoints/verl-ppoFREEZE__models--Qwen--Qwen2.5-7B_models--Skywork--Skywork-o1-Open-PRM-Qwen-2.5-7B_simplelr_qwen_level3to5_max_response8192_batch1024_rollout8_klcoef0.0001_entcoef0.001/global_step_140/critic/huggingface"

my_new_model = PRM_MODEL.from_pretrained_value(original_model_path)

for name, param in my_new_model.named_parameters():
    # if "v_head" in name:
        print(f"{name:<60} {param[0,:10]}")
        break

import torch
my_new_model.eval()
print(my_new_model(torch.tensor([[0, 1111, 2222, 3333]])))
print(my_new_model(torch.tensor([[0, 1111, 2222, 3333]]), return_probs=True))
