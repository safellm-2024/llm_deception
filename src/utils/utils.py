from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from torch import Tensor
from jaxtyping import Float, Int, Bool
import torch


def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch d_vocab"],
    answer_tokens_correct: Float[Tensor, "batch "],
    answer_tokens_wrong: Float[Tensor, "batch "],
    per_prompt: bool = False,
) -> Union[Float[Tensor, ""], Float[Tensor, "batch"]]:
    """
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    """
    if len(logits.shape) == 3:
        logits = logits[:, -1, :]
    elif len(logits.shape) == 2:
        logits = logits
    answer_tokens = torch.stack((answer_tokens_correct, answer_tokens_wrong), dim=-1)
    answer_logits = torch.gather(logits, dim=-1, index=answer_tokens)
    correct_logit, wrong_logit = answer_logits.unbind(dim=-1)
    logit_diff = correct_logit - wrong_logit
    return logit_diff if per_prompt else logit_diff.mean()
