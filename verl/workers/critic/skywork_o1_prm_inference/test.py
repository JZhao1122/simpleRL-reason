# test_fsdp.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools
import time
from prm_model import PRM_MODEL

def setup():
    """Initializes the distributed environment."""
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

def cleanup():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()

def print_rank_0(message):
    """Prints a message only on the rank 0 process."""
    if dist.get_rank() == 0:
        print(message)

def run_test(rank, world_size):
    """The main testing function."""
    setup()
    
    # --- 1. Model and Optimizer Setup ---
    print_rank_0(f"--> Setting up model on Rank {rank}")
    # Use bfloat16 for modern GPUs, float16 for older ones
    model_dtype = torch.bfloat16
    
    # Create the model on the meta device to save memory before FSDP wrapping
    with torch.device("meta"):
        model = PRM_MODEL.from_pretrained('/cpfs02/user/liurunze/hf_models/models--Skywork--Skywork-o1-Open-PRM-Qwen-2.5-1.5B')
    
    # FSDP wrapping policy
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1_000_000
    )

    # Wrap the model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        param_init_fn=lambda module: module.to_empty(device=torch.cuda.current_device(), recurse=False),
        use_orig_params=True, # Important for state_dict and optimizers
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=model_dtype,
            reduce_dtype=model_dtype,
            buffer_dtype=model_dtype,
        )
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print_rank_0(f"Model wrapped with FSDP. Total Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # --- 2. Dummy Data and Training Loop ---
    batch_size = 8
    seq_len = 1024
    vocab_size = 10000
    num_steps = 10

    print_rank_0("\n--> Starting dummy training loop...")
    start_time = time.time()
    for step in range(num_steps):
        # Generate dummy data directly on GPU to avoid CPU-GPU transfer overhead
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())

        optimizer.zero_grad()
        output = model(input_ids)
        
        # Loss calculation
        loss = nn.CrossEntropyLoss()(output.view(-1, vocab_size), labels.view(-1))
        
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"Step {step+1}/{num_steps} | Loss: {loss.item():.4f}")

    dist.barrier() # Wait for all processes to finish training
    total_time = time.time() - start_time
    print_rank_0(f"--> Training loop finished in {total_time:.2f} seconds.")
    
    # --- 3. Checkpoint Test (state_dict) ---
    print_rank_0("\n--> Testing FSDP state_dict checkpointing...")
    
    # FSDP requires you to get the state_dict on rank 0
    # after gathering all the shards.
    from torch.distributed.fsdp import FSDP, StateDictType, FullStateDictConfig
    
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        cpu_state_dict = model.state_dict()

    if rank == 0:
        print("Successfully got model state_dict on rank 0.")
        # You could save this dict to a file here
        # torch.save(cpu_state_dict, "my_checkpoint.pt")
        print(f"Number of keys in state_dict: {len(cpu_state_dict)}")
        
    dist.barrier()
    print_rank_0("--> Checkpoint test PASSED.")

    cleanup()

if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    run_test(rank, world_size)