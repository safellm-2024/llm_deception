from pipeline.configs.config_generation_jailbreak import Config
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
from src.submodules.dataset.task_and_dataset_jailbreaks import (
    ContrastiveDatasetJailbreak,
)
from src.submodules.basic.contrastive_model_base import (
    ContrastiveBase,
)
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
        "--save_dir", type=str, required=False, default="D:\Data\jailbreak"
    )
    parser.add_argument("--task_name", type=str, required=False, default="jailbreak")
    parser.add_argument("--do_sample", type=bool, required=False, default=False)
    parser.add_argument("--framework", type=str, required=True, default="transformers")
    parser.add_argument(
        "--jailbreak", type=str, required=True, default="evil_confidant"
    )

    return parser.parse_args()


def main(
    model_path: str,
    save_dir: str,
    task_name: str = "deception",
    contrastive_type: Tuple[str] = ("honest", "lying"),
    answer_type: Tuple[str] = ("true", "false"),
    do_sample: bool = False,
    framework: str = "transformer_lens",
    jailbreak: str = "evil_confidant",
):

    # 0. Set configuration
    model_alias = os.path.basename(model_path)
    cfg = Config(
        model_path=model_path,
        model_alias=model_alias,
        task_name=task_name,
        contrastive_type=contrastive_type,
        answer_type=answer_type,
        save_dir=save_dir,
        do_sample=do_sample,
        framework=framework,
        jailbreak=jailbreak,
    )
    pprint.pp(cfg)

    # 1. Load model_base
    model_base = load_model(cfg)
    # model_base.set_use_attn_result(True)
    # model_base.cfg.use_attn_in = True
    # model_base.cfg.use_hook_mlp_in = True

    # 2. Load data
    print("loading data")
    dataset = ContrastiveDatasetJailbreak(cfg)
    dataset.load_and_sample_datasets(train_test="train")
    dataset.process_dataset()
    dataset.construct_contrastive_prompts()

    # Initiate contrastive model base
    print("Initializing model_base base:")
    contrastive_basic = ContrastiveBase(cfg, model_base, dataset)

    # Generate and cache
    print("Generate and cache:")
    contrastive_basic.generate_and_contrastive_activation_cache()

    # Evaluate generation results
    print("Evaluate generation results:")
    contrastive_basic.evaluate_jailbreak()

    print("Get stage statistics:")
    contrastive_basic.get_stage_statistics()

    # Plot pca on the residual stream
    contrastive_basic.get_contrastive_activation_pca()

    print("done")


if __name__ == "__main__":

    args = parse_arguments()
    print(sae_lens.__version__)
    print("run_contrastive_basic\n\n")
    print("model_base_path")
    print(args.model_path)
    print("save_dir")
    print(args.save_dir)
    print("task_name")
    print(args.task_name)
    print("do_sample")
    do_sample = True
    print(do_sample)
    print(args.do_sample)

    print("transformer_lens")
    # transformer_lens = False
    # print(transformer_lens)
    print(args.framework)

    if args.task_name == "deception":
        contrastive_type = ("honest", "lying")
        answer_type = ("true", "false")
    elif args.task_name == "jailbreak":
        contrastive_type = ("HHH", args.jailbreak)
        answer_type = ("harmless", "harmful")

    print("answer_type")
    print(answer_type)

    print("contrastive_type")
    print(contrastive_type)
    main(
        model_path=args.model_path,
        save_dir=args.save_dir,
        task_name=args.task_name,
        contrastive_type=contrastive_type,
        answer_type=answer_type,
        do_sample=do_sample,
        framework=args.framework,
        jailbreak=args.jailbreak,
    )
