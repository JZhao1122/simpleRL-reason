import argparse
import os
import sys
import importlib
import traceback
import multiprocessing as mp
import queue
import math # For ceiling division

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

from tqdm import tqdm
from functools import partial
from utils.util import timestamped_print, save_json, print_args
from aggregation_util import load_and_merge_json_files_to_hf_dataset
from transformers import AutoTokenizer
from math_verify import parse # Assuming this is your custom parsing function


def parse_args():
    parser = argparse.ArgumentParser(description="Process data framework.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Directory containing input JSON files to process."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output JSON file."
    )
    return parser.parse_args()

def load_tree(step_tree: dict) -> queue.Queue:
    """
    Load the tree structure from the step_tree dictionary.
    """
    not_end = queue.Queue()
    bfs_queue = queue.Queue()
    
    if not step_tree:
        timestamped_print("Step tree is empty.", "WARNING")
        raise ValueError("Step tree is empty. Please provide a valid step tree.")
    
    bfs_queue.put(step_tree)
    while not bfs_queue.empty():
        node = bfs_queue.get()
        if node['is_final'] == False and node['child_nodes'] == []:
            not_end.put(node)
            timestamped_print(f"Found a non-final node: {node}", "INFO")
            continue
        
        best_node = {}
        for child in node.get('child_nodes', []):
            if best_node == {} or child['token_rewards'][-1] > best_node['token_rewards'][-1]:
                best_node = child
        
        # if the node is not final, add it to the queue for further expansion
        if not best_node['is_final']:
            bfs_queue.put(best_node)
        else:
            if best_node['correctness']:
                return True
            else:
                return False
    
    return None


def main():
    args = parse_args()
    print_args(args, program_name="Chunked Budget Aggregation Logic with Fallbacks", version="1.2")
    dataset = load_and_merge_json_files_to_hf_dataset(
        directory_path=args.input_path,
        features=None,
        encoding='utf-8',
        max_workers=16 
    )
    timestamped_print(f"Loaded data: {dataset}")
    
    correct = 0
    total = 0
    error = 0
    for data in dataset:
        result = load_tree(data['step_tree'])
        if result is True:
            correct += 1
        elif result is False:
            pass
        else:
            error += 1
        
        total += 1
    
    print(f"Total: {total}, Correct: {correct}, Errors: {error}")
    timestamped_print("Processing complete.", "INFO")