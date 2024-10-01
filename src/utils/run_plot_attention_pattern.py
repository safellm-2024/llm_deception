from pipeline.configs.config_plot_attention import Config
import os
import sys
import random
import json
import pprint
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from torch import Tensor
from jaxtyping import Float, Int, Bool
import einops
import argparse
from datasets import load_dataset
import sae_lens
import pysvelte
from torchtyping import TensorType as TT
from IPython import get_ipython
from pathlib import Path
import pickle
import numpy as np
from matplotlib.widgets import EllipseSelector
from IPython.display import HTML, Markdown
import torch
from src.submodules.dataset.task_and_dataset_deception import (
    ContrastiveDatasetDeception,
    tokenize_prompts_and_answers,
)
from src.submodules.basic.contrastive_model_base import (
    ContrastiveBase,
)
from src.submodules.activation_patching.activation_patching import ActivationPatching
from src.submodules.model.load_model import load_model
from src.utils.utils import logits_to_ave_logit_diff
from transformer_lens import utils, HookedTransformer, ActivationCache
from rich.table import Table, Column
from rich import print as rprint
from src.utils.plotly_utils import imshow, line, scatter, bar
import plotly.io as pio
import circuitsvis as cv  # for visualize attetnion pattern

from IPython.display import display, HTML  # for visualize attetnion pattern

torch.set_grad_enabled(False)  # save memory


def parse_arguments():
    """Parse model_base path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model_base path argument.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="google/gemma-2-2b-it"
    )
    parser.add_argument(
        "--save_dir", type=str, required=False, default="D:\Data\deception"
    )
    parser.add_argument("--layers", nargs="+", type=int, required=False, default=0)
    parser.add_argument("--heads", nargs="+", type=int, required=False, default=0)
    return parser.parse_args()


def main(
    model_path: str,
    save_dir: str,
    heads_plot: Tuple[list],
):

    # 0. Set configuration
    model_alias = os.path.basename(model_path)
    cfg = Config(
        model_path=model_path,
        model_alias=model_alias,
        save_dir=save_dir,
    )
    pprint.pp(cfg)

    # load data
    load_path = os.path.join(
        cfg.artifact_path(), "activation_patching", "block", "transformer_lens"
    )
    os.path.exists(load_path)

    with open(
        load_path
        + os.sep
        + f"{cfg.model_alias}_('z', 'q', 'pattern', 'v', 'k')_activation_patching.pkl",
        "rb",
    ) as f:
        data = pickle.load(f)

    plot_attention_pattern(cfg, data, "clean")
    plot_attention_pattern(cfg, data, "corrupted")

    print("done")


def plot_attention_pattern(cfg, data, contrastive_label):
    pattern = data[f"pattern_{contrastive_label}"]
    attn_patterns_for_important_heads: Float[Tensor, "head q k"] = torch.stack(
        [pattern[layer][:, head][0] for layer, head in heads_plot]
    )
    tokens_clean = data[f"tokens_{contrastive_label}"]

    # plot
    # display(HTML(f"<h2>Top {k} Logit Attribution Heads (from value-activation_patching)</h2>"))
    vis = cv.attention.attention_patterns(
        attention=attn_patterns_for_important_heads,
        tokens=tokens_clean,
        attention_head_names=[f"{layer}.{head}" for layer, head in heads_plot],
    )

    # save the visualization as .html file
    save_path = os.path.join(cfg.artifact_path(), "attention_patterns")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    vis_path = (
        save_path
        + os.sep
        + f"{cfg.model_alias}_attention_{heads_plot}_{contrastive_label}.html"
    )
    with open(vis_path, "w") as f:
        f.write(vis._repr_html_())


if __name__ == "__main__":

    args = parse_arguments()
    print(sae_lens.__version__)
    print(sae_lens.__version__)
    print("run attention pattern\n\n")

    # print("heads")
    # print(args.heads)
    # print("layers")
    # print(args.layers)

    heads_plot = []
    for ll, hh in zip(args.layers, args.heads):
        heads_plot.append([ll, hh])
    main(
        model_path=args.model_path,
        save_dir=args.save_dir,
        heads_plot=heads_plot,
    )
