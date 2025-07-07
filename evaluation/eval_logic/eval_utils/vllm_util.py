import gc
import ray
import torch

def unload_model(llm_service, tokenizer=None):
    """
    Unload the model and clear resources.
    """
    try:
        del llm_service
        if tokenizer:
            del tokenizer
    except Exception as e:
        print(f"Failed to unload model: {e}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        ray.shutdown()
    try:
        del llm_service
        if tokenizer:
            del tokenizer
    except Exception as e:
        print(f"Failed to unload model: {e}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        ray.shutdown()
