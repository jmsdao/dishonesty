import numpy as np
import pandas as pd

from typing import Optional, Union, List
from torch import Tensor


def ntensor_to_long(
    tensor: Union[Tensor, np.array],
    value_name: str = "values",
    dim_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Converts an n-dimensional tensor to a long format dataframe.
    """
    df = pd.DataFrame()
    df[value_name] = tensor.cpu().numpy().flatten()

    for i, _ in enumerate(tensor.shape):
        pattern = np.repeat(np.arange(tensor.shape[i]), np.prod(tensor.shape[i+1:]))
        n_repeats = np.prod(tensor.shape[:i])
        df[f"dim{i}"] = np.tile(pattern, n_repeats)

    if dim_names is not None:
        df.columns = [value_name] + dim_names
    
    return df


def calc_soft_kl_div(target_logits, input_logits, temperature=1.0, eps=1e-6):
    # When temperature is 0.0, calculate using argmax for numerical stability.
    if temperature == 0.0:
        argmax_indices = target_logits.argmax(dim=-1, keepdim=True)
        target_logprobs = target_logits.log_softmax(dim=-1)
        input_logprobs = input_logits.log_softmax(dim=-1)
        target_logprobs_at_argmax = target_logprobs.gather(dim=-1, index=argmax_indices)
        input_logprobs_at_argmax = input_logprobs.gather(dim=-1, index=argmax_indices)
        return (target_logprobs_at_argmax - input_logprobs_at_argmax).squeeze(-1)

    if temperature <= 0.01:
        print("Warning: be aware of numerical instability as temperature gets very small.")

    probs = (target_logits / (temperature + eps)).softmax(dim=-1)
    logprob_diffs = target_logits.log_softmax(dim=-1) - input_logits.log_softmax(dim=-1)
    return (probs * logprob_diffs).sum(dim=-1)
