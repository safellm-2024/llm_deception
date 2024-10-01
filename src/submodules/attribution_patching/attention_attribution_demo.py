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
import pysvelte
from torchtyping import TensorType as TT
from IPython import get_ipython
from pathlib import Path
import pickle

# ipython = get_ipython()
# Code to automatically update the HookedTransformer code as its edited without restarting the kernel
# ipython.magic("load_ext autoreload")
# ipython.magic("autoreload 2")
from IPython.display import HTML, Markdown
from sae_lens import HookedSAETransformer
from tqdm import tqdm
from src.submodules.evaluations.run_evaluate_generation_deception import (
    plot_lying_honest_performance,
)
from src.submodules.evaluations.run_evaluate_generation_deception import (
    evaluate_generation_deception,
)
from functools import partial
import torch
from src.submodules.dataset.task_and_dataset_deception import (
    load_and_sample_datasets,
    construct_prompt,
)
from src.submodules.model.load_model import load_model
from transformer_lens import utils, HookedTransformer, ActivationCache
from rich.table import Table, Column
from rich import print as rprint
from src.utils.plotly_utils import imshow, line, scatter, bar
import plotly.io as pio
import circuitsvis as cv  # for visualize attetnion pattern

from IPython.display import display, HTML  # for visualize attetnion pattern


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


def ioi_task_setup(
    model,
) -> tuple[List[str], List[tuple[str, str]]]:

    prompts = [
        "When John and Mary went to the shops, John gave the bag to",
        "When John and Mary went to the shops, Mary gave the bag to",
        "When Tom and James went to the park, James gave the ball to",
        "When Tom and James went to the park, Tom gave the ball to",
        "When Dan and Sid went to the shops, Sid gave an apple to",
        "When Dan and Sid went to the shops, Dan gave an apple to",
        "After Martin and Amy went to the park, Amy gave a drink to",
        "After Martin and Amy went to the park, Martin gave a drink to",
    ]
    answers = [
        (" Mary", " John"),
        (" John", " Mary"),
        (" Tom", " James"),
        (" James", " Tom"),
        (" Dan", " Sid"),
        (" Sid", " Dan"),
        (" Martin", " Amy"),
        (" Amy", " Martin"),
    ]

    return prompts, answers


def run_with_cache(model, clean_tokens, corrupted_tokens):

    # Run the model and cache all activations
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

    return clean_logits, corrupted_logits, clean_cache, corrupted_cache


def get_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_token_indices: Float[Tensor, "batch 2"],
) -> Tensor:
    """
    Get logit difference between the correct and incorrect answers
    :param logits:
    :param answer_token_indices: answer token indices for correct and incorrect answers
    :return: a scaler for mean logit difference across samples
    """
    # (1) Get last token logits only
    logits = logits[:, -1, :]  # [batch, d_vocab]

    # (2) Get the index into correct dimension
    # squeeze because the index tensor must have the same number of dimensions as input tensor
    correct_index = answer_token_indices[:, 0].unsqueeze(1)  # [batch, 1]
    incorrect_index = answer_token_indices[:, 1].unsqueeze(1)  # [batch, 1]

    # (3) gather the logits corresponding to the indices
    correct_logits = logits.gather(dim=1, index=correct_index)
    incorrect_logits = logits.gather(dim=1, index=incorrect_index)

    return (correct_logits - incorrect_logits).mean()


def get_cache_fwd_and_bwd(model, metric, tokens):
    filter_not_qkv_input = lambda name: "_input" not in name

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    model.reset_hooks()
    cache = {}
    grad_cache = {}

    model.add_hook(filter_not_qkv_input, forward_cache_hook, "fwd")
    model.add_hook(filter_not_qkv_input, backward_cache_hook, "bwd")

    value = metric(model(tokens))
    value.backward()  # the loss for computing the backward pass
    model.reset_hooks()
    return (
        value.item(),
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )


def create_attention_attr(
    model, clean_cache, clean_grad_cache
) -> TT["batch", "layer", "head_index", "dest", "src"]:
    attention_stack = torch.stack(
        [clean_cache["pattern", l] for l in range(model.cfg.n_layers)], dim=0
    )
    attention_grad_stack = torch.stack(
        [clean_grad_cache["pattern", l] for l in range(model.cfg.n_layers)], dim=0
    )
    attention_attr = attention_grad_stack * attention_stack
    attention_attr = einops.rearrange(
        attention_attr,
        "layer batch head_index dest src -> batch layer head_index dest src",
    )

    return attention_attr


class AttributionPatching:
    def __init__(self, model, prompts, answers):
        self.model = model
        self.prompts = prompts
        self.answers = answers
        self.clean_tokens, self.corrupted_tokens, self.answer_token_indices = (
            self.get_tokens_and_indices()
        )
        self.clean_baseline = None
        self.corrupted_baseline = None

        self.head_names = [
            f"L{l}H{h}"
            for l in range(self.model.cfg.n_layers)
            for h in range(self.model.cfg.n_heads)
        ]
        self.head_names_signed = [
            f"{name}{sign}" for name in self.head_names for sign in ["+", "-"]
        ]
        self.head_nams_qkv = [
            f"{name}{act_name}"
            for name in self.head_names
            for act_name in ["Q", "K", "V"]
        ]
        print(self.head_names[:5])
        print(self.head_names_signed[:5])
        print(self.head_nams_qkv[:5])

    def get_tokens_and_indices(
        self,
    ) -> tuple[
        [Tensor, "batch seq d_vocab"],
        [Tensor, "batch seq d_vocab"],
        [Tensor, "batch 2"],
    ]:
        clean_tokens = self.model.to_tokens(self.prompts)
        # Swap each adjacent pair, with a hacky list comprehension
        corrupted_tokens = clean_tokens[
            [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
        ]
        print("Clean string 0", self.model.to_string(clean_tokens[0]))
        print("Corrupted string 0", self.model.to_string(corrupted_tokens[0]))

        answer_token_indices = torch.tensor(
            [
                [self.model.to_single_token(self.answers[i][j]) for j in range(2)]
                for i in range(len(self.answers))
            ],
            device=self.model.cfg.device,
        )
        print("Answer token indices", answer_token_indices)

        return clean_tokens, corrupted_tokens, answer_token_indices

    def get_baseline_logit_diff(self) -> None:
        """
        get baseline logit differnece for lean and corrupted inputs
        :return:
        """
        clean_logits, corrupted_logits, clean_cache, corrupted_cache = run_with_cache(
            self.model, self.clean_tokens, self.corrupted_tokens
        )
        clean_logit_diff = get_logit_diff(
            clean_logits, self.answer_token_indices
        ).item()
        print(f"Clean logit diff: {clean_logit_diff:.4f}")

        corrupted_logit_diff = get_logit_diff(
            corrupted_logits, self.answer_token_indices
        ).item()
        print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")

        self.clean_baseline = clean_logit_diff
        self.corrupted_baseline = corrupted_logit_diff

        print(f"Clean Baseline is 1: {self.ioi_metric(clean_logits).item():.4f}")
        print(
            f"Corrupted Baseline is 0: {self.ioi_metric(corrupted_logits).item():.4f}"
        )

    def ioi_metric(
        self,
        logits: Float[Tensor, "batch seq d_vocab"],
    ) -> Tensor:
        """
        Scale the logit difference between 0 and 1 (clean baseline will have logit diff =1, and corrupted baseline will have logit diff =0)
        :param logits:
        :param answer_token_indices:
        :param clean_logit_diff:
        :param corrupted_logit_diff:
        :return:
        """

        return (
            get_logit_diff(logits, self.answer_token_indices) - self.corrupted_baseline
        ) / (self.clean_baseline - self.corrupted_baseline)

    def get_attention_attribution(self):
        clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(
            self.model, self.ioi_metric, self.clean_tokens
        )
        corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(
            self.model, self.ioi_metric, self.corrupted_tokens
        )
        attention_attr = create_attention_attr(
            self.model, clean_cache, clean_grad_cache
        )
        return attention_attr

    def plot_attention_attr(
        self, attention_attr, tokens, top_k=20, index=0, title="", save_path=None
    ):
        # save the visualization as .html file
        vis_path = save_path + os.sep + "attention_attribution"
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)

        if len(tokens.shape) == 2:
            tokens = tokens[index]
        if len(attention_attr.shape) == 5:
            attention_attr = attention_attr[index]
        attention_attr_pos = attention_attr.clamp(min=-1e-5)
        attention_attr_neg = -attention_attr.clamp(max=1e-5)
        attention_attr_signed = torch.stack(
            [attention_attr_pos, attention_attr_neg], dim=0
        )
        attention_attr_signed = einops.rearrange(
            attention_attr_signed,
            "sign layer head_index dest src -> (layer head_index sign) dest src",
        )
        attention_attr_signed = attention_attr_signed / attention_attr_signed.max()
        attention_attr_indices = (
            attention_attr_signed.max(-1).values.max(-1).values.argsort(descending=True)
        )
        # print(attention_attr_indices.shape)
        # print(attention_attr_indices)
        attention_attr_signed = attention_attr_signed[attention_attr_indices, :, :]
        head_labels = [self.head_names_signed[i.item()] for i in attention_attr_indices]

        if title:
            display(Markdown("### " + title))
        str_tokens = self.model.to_str_tokens(tokens)
        attention = attention_attr_signed.permute(1, 2, 0)[:, :, :top_k]
        head_labels = head_labels[:top_k]

        results = {
            "str_tokens": str_tokens,
            "attention": attention,
            "head_labels": head_labels,
        }
        with open(vis_path + os.sep + "attention_attribution.pkl", "wb") as f:
            pickle.dump(results, f)
        # vis = pysvelte.AttentionMulti(
        #     tokens=self.model.to_str_tokens(tokens),
        #     attention=attention_attr_signed.permute(1, 2, 0)[:, :, :top_k],
        #     head_labels=head_labels[:top_k],
        # ).publish(dev=True)
        # vis.publish(vis_path + os.sep + f"top20.html", dev_host="try")


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
    artifact_dir = cfg.artifact_path()
    save_path = os.path.join(artifact_dir, "attribution_patching")

    # 1. Load model
    model = load_model(cfg)
    model.set_use_attn_result(True)
    model.cfg.use_attn_in = True
    model.cfg.use_hook_mlp_in = True

    # 2. ioi_task_setup
    print("loading data")
    prompts, answers = ioi_task_setup(model)

    # initiate
    attribution_patching = AttributionPatching(model, prompts, answers)
    # get baseline logit diff
    attribution_patching.get_baseline_logit_diff()
    # get attention attribution
    attention_attr = attribution_patching.get_attention_attribution()
    # visualize attention pattern
    attribution_patching.plot_attention_attr(
        attention_attr,
        attribution_patching.clean_tokens,
        top_k=20,
        index=0,
        title="",
        save_path=save_path,
    )


if __name__ == "__main__":

    args = parse_arguments()
    print(sae_lens.__version__)
    print(sae_lens.__version__)
    print("attribution_patching_demo\n\n")
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
