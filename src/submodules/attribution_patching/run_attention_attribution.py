import random
import json
import pprint
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
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
import numpy as np
from matplotlib.widgets import EllipseSelector
from IPython.display import HTML, Markdown
import torch
from src.submodules.dataset.task_and_dataset_deception import (
    ContrastiveDatasetDeception,
    tokenize_prompts_and_answers,
)
from src.submodules.attribution_patching.attribution_patching import (
    AttributionPatching,
)
from src.submodules.model.load_model import load_model
from src.utils.utils import logits_to_logit_diff
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

    # 2. Load data
    print("loading data")
    dataset = ContrastiveDatasetDeception(cfg)
    dataset.load_and_sample_datasets(train_test="train")
    dataset.process_dataset()
    dataset.construct_contrastive_prompts()

    # Initiate attribution activation_patching
    attribution_patching = AttributionPatching(cfg, model, dataset)
    # Get baseline logit diff
    attribution_patching.get_contrastive_baseline_logit_diff()
    # Get attention attribution
    attention_attr = attribution_patching.get_attention_attribution()
    # Visualize attention pattern
    attribution_patching.plot_attention_attr(
        attention_attr,
        attribution_patching.prompt_tokens_positive[0],
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
