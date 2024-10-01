import random
import json
import pprint
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
import numpy as np
from matplotlib.widgets import EllipseSelector
from torch import Tensor
from jaxtyping import Float, Int, Bool
import einops
import os
import argparse
from pipeline.configs.config_generation import Config
from datasets import load_dataset
import sae_lens
from sae_lens import HookedSAETransformer
from tqdm import tqdm
from src.submodules.evaluations.run_evaluate_generation_deception import (
    plot_lying_honest_performance,
)
from src.submodules.evaluations.run_evaluate_generation_deception import (
    evaluate_generation_deception,
)
from src.submodules.logit_attribution.logit_attribution import (
    get_logi_diff_direction,
)
from src.utils.utils import logits_to_logit_diff
import torch
from src.submodules.dataset.task_and_dataset_deception import (
    ContrastiveDatasetDeception,
    tokenize_prompts_and_answers,
)
from src.submodules.model.load_model import load_model
from transformer_lens import utils, HookedTransformer, ActivationCache
from rich.table import Table, Column
from rich import print as rprint
from src.utils.plotly_utils import imshow, line, scatter, bar
import plotly.io as pio
import circuitsvis as cv  # for visualize attetnion pattern

from IPython.display import display, HTML  # for visualize attetnion pattern

torch.set_grad_enabled(False)


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="google/gemma-2-2b-it"
    )
    parser.add_argument(
        "--save_dir", type=str, required=False, default="D:\Data\deception"
    )
    parser.add_argument("--task_name", type=str, required=False, default="deception")

    return parser.parse_args()


def test_prompt(cfg, model, dataset):

    prompts = dataset.prompts_positive
    labels = dataset.labels_positive
    example_id = 0
    example_prompt = prompts[example_id]
    if labels[example_id] == 0:
        example_answer = "false"
    else:
        example_answer = "true"

    utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)


def residual_stack_to_logit_diff(
    cfg,
    residual_stack: Float[Tensor, "... batch d_model"],
    cache: ActivationCache,
    logit_diff_directions: Float[Tensor, "batch d_model"],
) -> Float[Tensor, "..."]:
    """
    Gets the avg logit difference between the correct and incorrect answer for a given
    stack of components in the residual stream.
    """
    batch_size = residual_stack.size(-2)
    scaled_final_token_residual_stream = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )
    if len(logit_diff_directions.shape) == 3:
        logit_diff_directions = logit_diff_directions.squeeze()

    if cfg.dtype == "bfloat16":
        average_logit_diff = (
            einops.einsum(
                scaled_final_token_residual_stream.to(torch.bfloat16),
                logit_diff_directions,
                "... batch d_model, batch d_model -> ...",
            )
            / batch_size
        )
    else:
        average_logit_diff = (
            einops.einsum(
                scaled_final_token_residual_stream,
                logit_diff_directions,
                "... batch d_model, batch d_model -> ...",
            )
            / batch_size
        )
    return average_logit_diff


def test_logit_diff(cfg, cache, logit_diff_directions, original_average_logit_diff):
    final_residual_stream: Float[Tensor, "batch seq d_model"] = cache["resid_post", -1]
    print(f"Final residual stream shape: {final_residual_stream.shape}")
    final_token_residual_stream: Float[Tensor, "batch d_model"] = final_residual_stream[
        :, -1, :
    ]
    average_logit_diff = residual_stack_to_logit_diff(
        cfg, final_token_residual_stream, cache, logit_diff_directions
    )
    print(f"average_logit_diff: {average_logit_diff}")
    print(f"original_average_logit_diff: {original_average_logit_diff}")
    if cfg.dtype != "bfloat16":
        torch.testing.assert_close(
            average_logit_diff, original_average_logit_diff, rtol=0.001, atol=0.001
        )


def residual_logit_diff_acc(cfg, cache, logit_diff_directions):
    """
    get residual logit difference by accumulating residual layers
    :param cfg:
    :param cache:
    :param logit_diff_directions:
    :return:
    """
    accumulated_residual, labels = cache.accumulated_resid(
        layer=-1, incl_mid=True, pos_slice=-1, return_labels=True
    )
    # accumulated_residual has shape (component, batch, d_model)

    logit_lens_logit_diffs: Float[Tensor, "component"] = residual_stack_to_logit_diff(
        cfg, accumulated_residual, cache, logit_diff_directions
    )

    fig = line(
        logit_lens_logit_diffs,
        hovermode="x unified",
        title="Logit Difference From Accumulated Residual Stream",
        labels={"x": "Layer", "y": "Logit Diff"},
        xaxis_tickvals=labels,
        width=800,
        return_fig=True,
    )
    # save figure
    artifact_dir = cfg.artifact_path()
    save_path = os.path.join(artifact_dir, "logit_diff")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.write_html(
        save_path + os.sep + cfg.model_alias + "_logit_diff_accumulated.html"
    )
    pio.write_image(
        fig,
        save_path + os.sep + cfg.model_alias + "_logit_diff_accumulated.png",
        scale=6,
    )


def residual_logit_diff_dec(cfg, cache, logit_diff_directions):
    """
    get residual logit difference by decomposing layers by layers

    :param cfg:
    :param cache:
    :param logit_diff_directions:
    :return:
    """
    per_layer_residual, labels = cache.decompose_resid(
        layer=-1, pos_slice=-1, return_labels=True
    )
    # accumulated_residual has shape (component, batch, d_model)

    logit_lens_logit_diffs: Float[Tensor, "component"] = residual_stack_to_logit_diff(
        cfg, per_layer_residual, cache, logit_diff_directions
    )

    fig = line(
        logit_lens_logit_diffs,
        hovermode="x unified",
        title="Logit Difference From Accumulated Residual Stream",
        labels={"x": "Layer", "y": "Logit Diff"},
        xaxis_tickvals=labels,
        width=800,
        return_fig=True,
    )
    # save figure
    artifact_dir = cfg.artifact_path()
    save_path = os.path.join(artifact_dir, "logit_diff")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.write_html(save_path + os.sep + cfg.model_alias + "_logit_diff_decomposed.html")
    pio.write_image(
        fig,
        save_path + os.sep + cfg.model_alias + "_logit_diff_decomposed.png",
        scale=6,
    )


def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    """
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].

    Example: if tensor is 2D array of values for each head in each layer, this will
    return a list of heads.
    """
    # first flatten to get the top k index
    i = torch.topk(tensor.flatten(), k).indices
    # and then from flattern --> to corrdinate
    top_k_heads = np.array(
        np.unravel_index(utils.to_numpy(i), tensor.shape)
    )  # you get tuple here

    return top_k_heads.T.tolist()  # from tuple to list


class ContrastiveLogitsAttribution:
    def __init__(self, cfg, model, dataset):
        self.cfg = cfg
        self.model = model
        self.dataset = dataset

        self.prompts = None
        self.answers_correct = None
        self.answers_wrong = None
        self.answer_tokens_correct = None
        self.answer_tokens_wrong = None
        self.logits = None
        self.cache = None
        self.prompt_tokens = None
        self.answer_tokens_correct = None
        self.answer_tokens_wrong = None

    def run_with_cache_all(self):
        """ "
        run with cache by concatenate contrastive data (positive + negative)
        """
        # (1) Concatenate positive and negative prompts together
        prompts = self.dataset.prompts_positive + self.dataset.prompts_negative
        answers_correct = (
            self.dataset.answers_positive_correct
            + self.dataset.answers_negative_correct
        )
        answers_wrong = (
            self.dataset.answers_positive_wrong + self.dataset.answers_negative_wrong
        )

        # (2) Tokenize answers and prompts
        prompt_tokens, answer_tokens_correct, answer_tokens_wrong = (
            tokenize_prompts_and_answers(
                self.model, prompts, answers_correct, answers_wrong
            )
        )

        # (3) run with cache
        original_logits, cache = self.model.run_with_cache(prompt_tokens)

        # save as instance attribute
        self.prompts = prompts
        self.answers_correct = answers_correct
        self.answers_wrong = answers_wrong
        self.answer_tokens_correct = answer_tokens_correct
        self.answer_tokens_wrong = answer_tokens_wrong
        self.prompt_tokens = prompt_tokens
        self.answer_tokens_correct = answer_tokens_correct
        self.answer_tokens_wrong = answer_tokens_wrong
        self.logits = original_logits
        self.cache = cache
        return original_logits, cache

    def get_logits_diff_and_directions(self):

        # (1) Get logit difference
        # per prompt
        original_per_prompt_diff = logits_to_logit_diff(
            self.logits,
            self.answer_tokens_correct,
            self.answer_tokens_wrong,
            per_prompt=True,
        )
        print(f"original_per_prompt_diff: {original_per_prompt_diff}")
        # mean of all prompts
        original_average_logit_diff = logits_to_logit_diff(
            self.logits,
            self.answer_tokens_correct,
            self.answer_tokens_wrong,
            per_prompt=False,
        )
        print(f"original_average_logit_diff: {original_average_logit_diff}")

        # (2) Get logit difference direction in the residual stream
        logit_diff_directions = get_logi_diff_direction(
            self.model, self.answer_tokens_correct, self.answer_tokens_wrong
        )

        # (3) Test if logit difference from last residual layer is same as the original logit difference
        test_logit_diff(
            self.cfg, self.cache, logit_diff_directions, original_average_logit_diff
        )

        return logit_diff_directions

    def get_residual_logit_diff_per_head(self, logit_diff_directions):

        # (1) Get the stack of attention residuals per head
        per_head_residual, labels = self.cache.stack_head_results(
            layer=-1, pos_slice=-1, return_labels=True
        )
        # rearrange into shape [layer, head, source, destination]
        per_head_residual = einops.rearrange(
            per_head_residual,
            "(layer head) ... -> layer head ...",
            layer=self.model.cfg.n_layers,
        )

        # (2) From a stack of residuals to their logit differences
        per_head_logit_diffs = residual_stack_to_logit_diff(
            self.cfg, per_head_residual, self.cache, logit_diff_directions
        )

        # (3) Plot per head logit difference
        fig = imshow(
            per_head_logit_diffs,
            labels={"x": "Head", "y": "Layer"},
            title="Logit Difference From Each Head",
            width=600,
            return_fig=True,
        )

        # (4) Save
        artifact_dir = self.cfg.artifact_path()
        save_path = os.path.join(artifact_dir, "logit_diff")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.write_html(
            save_path + os.sep + self.cfg.model_alias + "_logit_diff_per_head.html"
        )
        pio.write_image(
            fig,
            save_path + os.sep + self.cfg.model_alias + "_logit_diff_per_head.png",
            scale=6,
        )
        return per_head_logit_diffs

    def visualize_top_logit_diff_head(self, per_head_logit_diffs, tokens, topK=10):
        # save figure
        artifact_dir = self.cfg.artifact_path()
        save_path = os.path.join(artifact_dir, "logit_diff")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Visualize attention pattern of top k heads (based on their logit difference)
        for head_type in ["Positive", "Negative"]:
            # Get the heads with largest (or smallest) contribution to the logit difference
            top_heads = topk_of_Nd_tensor(
                per_head_logit_diffs * (1 if head_type == "Positive" else -1), topK
            )

            # Get all their attention patterns
            attn_patterns_for_important_heads: Float[Tensor, "head q k"] = torch.stack(
                [
                    self.cache["pattern", layer][:, head].mean(0)  # mean over batch
                    for layer, head in top_heads
                ]
            )

            # Display results
            display(HTML(f"<h2>Top {topK} {head_type} Logit Attribution Heads</h2>"))

            vis = cv.attention.attention_patterns(
                attention=attn_patterns_for_important_heads,  # [heads, source, destination]
                tokens=self.model.to_str_tokens(tokens[:300]),
                attention_head_names=[f"{layer}.{head}" for layer, head in top_heads],
            )

            # save the visualization as .html file
            vis_path = save_path + os.sep + f"vis_layer_top3_heads.html"
            with open(vis_path, "w") as f:
                f.write(vis._repr_html_())


def main(
    model_path: str,
    save_dir: str,
    task_name: str = "deception",
    contrastive_label: Tuple[str] = ("honest", "lying"),
):

    # 0. Set configuration
    model_alias = os.path.basename(model_path)
    cfg = Config(
        model_path=model_path,
        model_alias=model_alias,
        task_name=task_name,
        contrastive_label=contrastive_label,
        save_dir=save_dir,
    )
    pprint.pp(cfg)

    # 1. Load model
    model = load_model(cfg)

    # 2. Load data
    print("loading data")
    dataset = ContrastiveDatasetDeception(cfg)
    dataset.load_and_sample_datasets(train_test="train")
    dataset.process_dataset()
    dataset.construct_contrastive_prompts()

    # 3.0 Initialize
    contrastive_logits_attribution = ContrastiveLogitsAttribution(cfg, model, dataset)
    # test_prompt(cfg, model, dataset_train)

    # 3.1 Run with cache
    contrastive_logits_attribution.run_with_cache_all()

    # 4. Get logit difference direction
    logit_diff_directions = (
        contrastive_logits_attribution.get_logits_diff_and_directions()
    )

    # 5. Get per head logit difference in the residual
    per_head_logit_diffs = (
        contrastive_logits_attribution.get_residual_logit_diff_per_head(
            logit_diff_directions
        )
    )

    # 6. Visualize attention pattern
    contrastive_logits_attribution.visualize_top_logit_diff_head(
        per_head_logit_diffs,
        contrastive_logits_attribution.prompt_tokens[0],
        topK=10,
    )


if __name__ == "__main__":

    args = parse_arguments()
    print(sae_lens.__version__)
    print(sae_lens.__version__)
    print("run_contrastive_generation\n\n")
    print("model_path")
    print(args.model_path)
    print("save_dir")
    print(args.save_dir)
    print("task_name")
    print(args.task_name)

    if args.task_name == "deception":
        contrastive_label = ("honest", "lying")
    elif args.task_name == "jailbreak":
        contrastive_label = ("HHH", args.jailbreak)

    main(
        model_path=args.model_path,
        save_dir=args.save_dir,
        task_name=args.task_name,
        contrastive_label=contrastive_label,
    )
