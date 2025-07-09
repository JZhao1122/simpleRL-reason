# main.py
import argparse
import os
import sys
import importlib
import traceback

from framework import run_infer
from utils.util import timestamped_print, print_args


def parse_args():
    parser = argparse.ArgumentParser(description="Process data framework.")
    # init config
    parser.add_argument("--input_path", type=str, help="Path to the input data.")
    parser.add_argument("--output_path", type=str, help="Path to the output data.")
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    parser.add_argument("--reward_path", type=str, help="Path to the reward model.")
    parser.add_argument("--tensor_parallel_size", type=int, help="Number of tensor parallelism.")
    parser.add_argument("--reward_tensor_parallel_size", type=int, help="Number of tensor parallelism for reward model.")
    parser.add_argument("--store_type", type=str, choices=['file', 'folder'], default='file', help="Type of data storage.")
    # policy inference
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="System prompt for the model.")
    parser.add_argument("--user_prompt_template", type=str, default="{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.", help="User prompt template.")
    # tree
    parser.add_argument("--expand_size", type=int, default=1, help="Number of expansions for tree_build.")
    # step tree
    parser.add_argument("--step_tag", type=str, default="\n\n", help="Tag for the step tree.")
    # entropy tree
    parser.add_argument("--ep_threshold", type=float, default=0.1, help="Entropy threshold for the entropy tree.")
    # value guiding
    parser.add_argument("--reward_mode", type=str, choices=['none', 'token', 'entropy'], default='none', help="Mode for reward calculation.")
    parser.add_argument("--decode_mode", type=str, choices=['none', 'token', 'entropy'], default='none', help="Mode for decoding.")
    parser.add_argument("--max_candidates", type=int, default=5, help="Maximum number of candidates for sampling.")
    parser.add_argument("--entropy_threshold", type=float, default=0.5, help="Entropy threshold for filtering candidates.")
    parser.add_argument("--combine_prob", type=str, default="multiple", help="Function to combine probabilities.")
    # # Beam Search
    # parser.add_argument("--beam_size", type=int, default=1, help="Number of beams for beam search.")
    # parser.add_argument("--max_depth", type=int, default=1, help="Maximum depth for beam search.")
    # parser.add_argument("--max_tokens_per_step", type=int, default=512, help="Maximum tokens per step for beam search.")
    # parser.add_argument("--stop_sequences_beam", type=str, default="\n\n", help="Stop sequences for beam search.")


    parser.add_argument(
        "--process_module",
        type=str,
        required=True,
        help="Path to the user's processor module (e.g., 'user_logic.my_test_processor')."
    )

    return parser.parse_args()

def main():
    args = parse_args()
    print_args(args, program_name="Main Data Processor", version="1.0")
    try:
        timestamped_print(f"Attempting to import processor module: {args.process_module}")
        importlib.import_module(args.process_module)
        timestamped_print(f"Successfully imported '{args.process_module}'.")
    except ImportError as e:
        timestamped_print(f"ERROR: Could not import processor module '{args.process_module}'. Make sure it's in PYTHONPATH or a valid path. Details: {e}", 'ERROR')
        sys.exit(1)
    except Exception as e:
        timestamped_print(f"ERROR: An error occurred during import of '{args.process_module}'. Details: {e}", 'ERROR')
        traceback.print_exc()
        sys.exit(1)
    
    timestamped_print("Application starting...")
    run_infer(args)
    timestamped_print("Application finished.")

if __name__ == "__main__":
    main()