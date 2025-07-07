import math
from typing import Dict
from vllm.sequence import Logprob

def normalize_logprobs(logprobs: Dict[int, Logprob]) -> Dict[int, Logprob]:
    """
    Normalizes a dictionary of log-probabilities so that the sum of their
    exponentiated values (the probabilities) equals 1.

    This is done in log-space to prevent numerical underflow/overflow.

    Args:
        logprobs: The dictionary of {token_id: Logprob} to normalize.

    Returns:
        A new dictionary with normalized log-probability values.
    """
    if not logprobs:
        return {}

    # --- Use the Log-Sum-Exp trick for numerical stability ---
    # 1. Get all logprob values
    lp_values = [lp.logprob for lp in logprobs.values()]
    
    # 2. Find the maximum logprob
    max_lp = max(lp_values)
    
    # 3. Calculate sum(exp(lp - max_lp))
    sum_of_exps = sum(math.exp(lp - max_lp) for lp in lp_values)
    
    # 4. The log of the total sum is log(sum(exp(lp))) = max_lp + log(sum(exp(lp - max_lp)))
    log_total_prob = max_lp + math.log(sum_of_exps)

    # --- Create the new normalized dictionary ---
    normalized_dict = {}
    for token_id, original_logprob_obj in logprobs.items():
        # The new log-probability is the original minus the log of the sum
        # This is equivalent to: log( P(i) / sum(P) ) = log(P(i)) - log(sum(P))
        new_logprob = original_logprob_obj.logprob - log_total_prob
        
        normalized_dict[token_id] = Logprob(
            logprob=new_logprob,
            rank=original_logprob_obj.rank,
            decoded_token=original_logprob_obj.decoded_token
        )
        
    return normalized_dict

def norm(normalized_a: Dict[int, Logprob]):
    return math.sqrt(sum([math.exp(2*normalized_a[key].logprob) for key in normalized_a.keys()]))

def calculate_normalized_shared_prob_sum(
    logprobs_a: Dict[int, Logprob],
    logprobs_b: Dict[int, Logprob]
) -> float:
    """
    First, normalizes the log-probabilities in each dictionary. Then, it finds
    tokens present in both and calculates the sum of their joint probabilities.

    The calculation for each shared token is: exp(norm_logprob_a + norm_logprob_b)

    Args:
        logprobs_a: The first dictionary of {token_id: Logprob}.
        logprobs_b: The second dictionary of {token_id: Logprob}.

    Returns:
        A float representing the sum of joint probabilities for shared tokens,
        calculated after normalizing each input distribution.
    """
    # Step 1: Normalize both input dictionaries
    normalized_a = normalize_logprobs(logprobs_a)
    normalized_b = normalize_logprobs(logprobs_b)
    
    # Step 2: Find common keys in the normalized dictionaries
    common_keys = normalized_a.keys() & normalized_b.keys()

    if not common_keys:
        return 0.0

    # Step 3: Calculate the sum of joint probabilities using the normalized values
    total_prob_sum = sum(
        math.exp(normalized_a[key].logprob + normalized_b[key].logprob)
        for key in common_keys
    )/norm(normalized_a)/norm(normalized_b)

    return total_prob_sum