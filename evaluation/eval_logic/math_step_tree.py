import sys
import os
import argparse
import queue

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from framework.register import register_processor
from utils.util import load_json, save_json, timestamped_print, cprint
from infer_module.infer_vllm import LLM_Service
from vllm import SamplingParams
from math_verify import parse, verify


_cached_llm_service = None  # Global variable to cache the LLM_Service instance

def get_llm_service(model_path: str = None, tensor_parallel_size: int = 1) -> LLM_Service:
    """
    Get a singleton instance of the LLM_Service.
    If the instance is already created, return it.
    """
    global _cached_llm_service
    if _cached_llm_service is None:
        timestamped_print("MODEL: First call to get_llm_service. Attempting to load model...")
        try:
            _cached_llm_service = LLM_Service(model_path=model_path, tensor_parallel_size=tensor_parallel_size)
            timestamped_print("MODEL: Model loaded successfully via get_llm_service.")
        except Exception as e:
            timestamped_print(f"MODEL: Failed to load model via get_llm_service: {e}", "ERROR")
            raise ValueError(f"Failed to load model: {e}")
    else:
        timestamped_print("MODEL: Model loaded successfully via get_llm_service.")
    return _cached_llm_service

def load_tree(step_tree: dict) -> queue.Queue:
    """
    Load the tree structure from the step_tree dictionary.
    """
    not_end = queue.Queue()
    bfs_queue = queue.Queue()
    
    if not step_tree:
        timestamped_print("Step tree is empty.", "WARNING")
        return None
    
    bfs_queue.put(step_tree)
    while not bfs_queue.empty():
        node = bfs_queue.get()
        if node['is_final'] == False and node['child_nodes'] == []:
            not_end.put(node)
            timestamped_print(f"Found a non-final node: {node['node_content']}", "INFO")
        
        for child in node.get('child_nodes', []):
            bfs_queue.put(child)
    
    # Further processing can be added here
    return not_end

@register_processor('check_finish')
def check_finish(args: argparse.Namespace, output_filepath: str) -> bool:
    try:
        record = load_json(output_filepath)
    except Exception as e:
        timestamped_print(f"Error loading JSON file {output_filepath}: {e}", "ERROR")
        return False

    try:
        not_end = load_tree(record.get('step_tree', {}))
        if not_end.qsize() > 0:
            timestamped_print(f"Found {not_end.qsize()} non-final nodes in the step tree.", "INFO")
            return False
    except Exception as e:
        timestamped_print(f"Error processing step tree: {e}", "ERROR")
        return False
    
    return True

@register_processor('process')
def process_file(args) -> None:
    """
    input_filepath: str, 
    output_filepath: str, 
    model_path: str, 
    tensor_parallel_size: int
    num: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    system_prompt: str

    expand_size: int,
    step_tag: str
    """
    # get data from input file & load the llm_service
    data = load_json(args.input_filepath)
    llm_service = get_llm_service(model_path=args.model_path, tensor_parallel_size=args.tensor_parallel_size)
    
    data['expand_size'] = args.expand_size
    data['step_tag'] = args.step_tag

    # construct the root
    messages = [
        { "role": "system", "content": args.system_prompt }, 
        { "role": "user", "content": f"{data['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}." }, 
    ]
    root_prompt = llm_service.build_prompt(messages)

    # init the tree
    Root = data.get('step_tree', 
        {
            "index_list": [0],
            "history_content": [],
            "node_content": root_prompt,
            "is_final": False,
            "finish_reason": None,
            "correctness": None,
            "child_nodes": []
        }
    )

    # configure the sampling parameters
    sampling_params = SamplingParams(
        n=args.expand_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,  # Maximum number of tokens to generate
        stop=[args.step_tag],
        include_stop_str_in_output=True,  # Include the stop string in the output
    )

    # BFS to build the tree
    cprint(Root, "initial root node")
    q = load_tree(Root)
    cprint(q, "initial queue")
    cprint(q.qsize(), "initial queue size")
    cprint(q.empty(), "initial queue empty")
    while not q.empty():
        node = q.get()
        cur_prompt = ''.join(node['history_content']) + node['node_content']
        cprint(cur_prompt, "Current prompt for node")

        # generate the responses for the current node
        results = llm_service.inference(cur_prompt, sampling_params)
        new_contents = llm_service.get_text(results)[0]
        finish_reasons = llm_service.get_finish_reason(results)[0]
        stop_reasons = llm_service.get_stop_reason(results)[0]
        
        cprint(new_contents, "New contents generated for the node")
        for i, new_content, finish_reason, stop_reason in enumerate(zip(new_contents, finish_reasons, stop_reasons)):
            # create a new node
            new_node = {
                "index_list": node['index_list'] + [i],
                "history_content": node['history_content'] + [node['node_content']],
                "node_content": new_content,
                "is_final": finish_reason == 'length' or stop_reason is None,  # Check if the node is final based on finish reason or stop reason
                "finish_reason": finish_reason,
                "correctness": None,  # To be filled later
                "child_nodes": []
            }
            if new_node['is_final']:
                # if the node is final, we need to verify the correctness
                new_node['correctness'] = verify(
                    parse(new_content), 
                    parse(f"\\boxed{{{data['answer']}}}"),
                )
            
            node['child_nodes'].append(new_node)

            # if the node is not final, add it to the queue for further expansion
            if not new_node['is_final']:
                q.put(new_node)

        data['step_tree'] = Root

        save_json(data, args.output_filepath)
