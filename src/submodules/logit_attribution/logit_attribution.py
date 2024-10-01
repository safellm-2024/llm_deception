from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from torch import Tensor
from jaxtyping import Float, Int, Bool
import torch


def get_logi_diff_direction(
    model,
    answer_tokens_correct: Float[Tensor, "batch 1"],
    answer_tokens_wrong: Float[Tensor, "batch 1"],
):
    """
    Given the tokens of correct and wrong answer, get the logit diff direction in the residual stream
    logit_diff_direction = residual_directions_correct - residual_directions_wrong
    """
    # stack the correct and wrong answer token together
    answer_tokens = torch.stack(
        (
            torch.unsqueeze(answer_tokens_correct, 1),
            torch.unsqueeze(answer_tokens_wrong, 1),
        ),
        dim=1,
    )  # [batch 2 d_model]

    # from token indices to residual directions
    answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)

    # separate out the correct and wrong direction
    residual_directions_correct, residual_directions_wrong = (
        answer_residual_directions.unbind(dim=1)
    )

    # get logit diff directions
    logit_diff_directions = (
        residual_directions_correct - residual_directions_wrong
    )  # [batch d_model]

    return logit_diff_directions
