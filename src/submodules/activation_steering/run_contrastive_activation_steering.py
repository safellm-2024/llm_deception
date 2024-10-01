from pipeline.configs.config_activation_steering import Config
import os
import sys
import random
import json
import pprint
from tqdm import tqdm

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
)

from src.submodules.dataset.task_and_dataset_jailbreaks import (
    ContrastiveDatasetJailbreak,
)
from src.submodules.basic.contrastive_model_base import (
    ContrastiveBase,
)
from src.submodules.activation_patching.activation_patching import ActivationPatching
from src.submodules.model.load_model import load_model
from src.submodules.activation_steering.contrastive_act_steering import (
    ContrastiveActSteering,
    ContrastiveActSteeringAll,
)
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
    parser.add_argument("--source_layer", type=int, required=False, default=-1)
    parser.add_argument("--target_layer", type=int, required=False, default=-1)
    parser.add_argument("--task_name", type=str, required=False, default="deception")
    parser.add_argument("--do_sample", type=bool, required=False, default=True)
    parser.add_argument("--framework", type=str, required=False, default="transformers")
    parser.add_argument("--cache_name", type=str, required=False, default="resid_post")
    parser.add_argument(
        "--cache_pos", type=str, required=False, default="prompt_last_token"
    )
    parser.add_argument(
        "--jailbreak", type=str, required=False, default="evil_confidant"
    )
    parser.add_argument(
        "--jailbreak_source", type=str, required=False, default="evil_confidant"
    )
    parser.add_argument(
        "--intervention",
        type=str,
        required=False,
        default="input_positive_addition_negative",
    )
    parser.add_argument("--analysis_only", type=bool, required=False, default=False)
    return parser.parse_args()


def get_dataset(cfg):
    if cfg.task_name == "deception":
        dataset = ContrastiveDatasetDeception(cfg)
    elif cfg.task_name == "jailbreak":
        dataset = ContrastiveDatasetJailbreak(cfg)
    return dataset


def activation_steering_layer(cfg, model_base, layer):

    print("Generate steering vector")
    dataset = get_dataset(cfg)
    dataset.load_and_sample_datasets(train_test="train")
    dataset.process_dataset()
    dataset.construct_contrastive_prompts(jailbreak_source=cfg.jailbreak_source)

    contrastive_basic = ContrastiveBase(cfg, model_base, dataset)
    contrastive_basic.run_and_cache_contrastive_activation()
    act_steer = ContrastiveActSteering(
        cfg, dataset, contrastive_basic, source_layer=layer, target_layer=layer
    )
    steering_vector = act_steer.get_steering_vector()

    print("Apply steering vector")
    dataset = get_dataset(cfg)
    dataset.load_and_sample_datasets(train_test="test")
    dataset.process_dataset()
    dataset.construct_contrastive_prompts()
    contrastive_basic = ContrastiveBase(cfg, model_base, dataset)
    act_steer = ContrastiveActSteering(
        cfg, dataset, contrastive_basic, source_layer=layer, target_layer=layer
    )
    act_steer.generate_and_contrastive_activation_steering(steering_vector)
    act_steer.get_contrastive_activation_pca()
    act_steer.evaluate_intervention()
    act_steer.get_stage_statistics()


def main(
    model_path: str,
    save_dir: str,
    task_name: str = "deception",
    contrastive_type: Tuple[str] = ("honest", "lying"),
    answer_type: Tuple[str] = ("true", "false"),
    do_sample: bool = False,
    framework: str = "transformers",
    cache_name: str = "resid_pre",
    target_layer: int = 0,
    source_layer: int = 0,
    cache_pos: str = "prompt_last_token",
    jailbreak: str = "evil_confidant",
    jailbreak_source: str = "evi_confidant",
    intervention: str = "positive_addition",
    analysis_only: bool = False,
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
        cache_name=cache_name,
        source_layer=source_layer,
        target_layer=target_layer,
        cache_pos=cache_pos,
        intervention=intervention,
        jailbreak=jailbreak,
        jailbreak_source=jailbreak_source,
    )
    pprint.pp(cfg)

    # 1. Load model_base
    model_base = load_model(cfg)

    if cfg.source_layer == -1:
        if not analysis_only:
            print("Steering layer by layer:")
            for layer in tqdm(range(model_base.cfg.n_layers)):
                print(f"layer: {layer}")

                activation_steering_layer(cfg, model_base, layer)

        # summary statistics across layers
        print("Get summary statistics across layers: ")
        act_steer = ContrastiveActSteeringAll(cfg, model_base)
        act_steer.get_honest_score_all_layers()
        act_steer.get_cos_sim_all_layers()

    else:
        activation_steering_layer(cfg, model_base, layer=None)

    print("done")


if __name__ == "__main__":
    import time

    start = time.time()
    print("hello")

    args = parse_arguments()
    print(sae_lens.__version__)
    print(sae_lens.__version__)
    print("run contrastive activation steering\n\n")
    print("model_base_path")
    print(args.model_path)
    print("save_dir")
    print(args.save_dir)
    print("task_name")
    print(args.task_name)
    print("do_sample")
    print(args.do_sample)
    print("source_layer")
    print(args.source_layer)
    print("target_layer")
    print(args.target_layer)

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
        do_sample=args.do_sample,
        framework=args.framework,
        cache_name=args.cache_name,
        source_layer=args.source_layer,
        target_layer=args.target_layer,
        cache_pos=args.cache_pos,
        jailbreak=args.jailbreak,
        jailbreak_source=args.jailbreak_source,
        intervention=args.intervention,
        analysis_only=args.analysis_only,
    )
    end = time.time()
    print(end - start)
