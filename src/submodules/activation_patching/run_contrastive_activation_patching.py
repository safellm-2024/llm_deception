from pipeline.configs.config_activation_patching import Config
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

import torch
from src.submodules.dataset.task_and_dataset_deception import (
    ContrastiveDatasetDeception,
)
from src.submodules.dataset.task_and_dataset_jailbreaks_multiple_choice import (
    ContrastiveDatasetJailbreakChoice,
)
from src.submodules.basic.contrastive_model_base import (
    ContrastiveBase,
)
from src.submodules.activation_patching.activation_patching import ActivationPatching
from src.submodules.model.load_model import load_model


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
    parser.add_argument("--task_name", type=str, required=False, default="deception")
    parser.add_argument("--do_sample", type=bool, required=False, default=False)
    parser.add_argument(
        "--framework", type=str, required=False, default="transformer_lens"
    )
    parser.add_argument("--cache_name", type=str, required=False, default="resid_post")
    parser.add_argument(
        "--jailbreak", type=str, required=False, default="evil_confidant"
    )
    return parser.parse_args()


def main(
    model_path: str,
    save_dir: str,
    task_name: str = "deception",
    contrastive_type: Tuple[str] = ("honest", "lying"),
    answer_type: Tuple[str] = ("true", "false"),
    do_sample: bool = False,
    framework: str = "transformers",
    cache_name: str = "resid_pre",
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
        cache_name=cache_name,
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
    if cfg.task_name == "deception":
        dataset = ContrastiveDatasetDeception(cfg)
        dataset.load_and_sample_datasets(train_test="train")
        dataset.process_dataset()
        dataset.construct_contrastive_prompts()
    elif cfg.task_name == "jailbreak_choice":
        dataset = ContrastiveDatasetJailbreakChoice(cfg)
        dataset.load_and_sample_datasets(train_test="train")
        dataset.process_dataset()
        dataset.construct_contrastive_prompts()

    # Initiate contrastive model base
    print("Initializing model_base base:")
    contrastive_basic = ContrastiveBase(cfg, model_base, dataset)

    # Generate and cache
    print("Run and cache:")
    contrastive_basic.run_and_cache_contrastive_activation()

    # contrastive activation patching
    print("Run contrastive activation patching:")
    attr_patch = ActivationPatching(dataset, contrastive_basic)
    attr_patch.get_activation_patching_block()

    print("done")


if __name__ == "__main__":

    args = parse_arguments()
    print(sae_lens.__version__)
    print(sae_lens.__version__)
    print("run contrastive activation_patching\n\n")
    print("model_base_path")
    print(args.model_path)
    print("save_dir")
    print(args.save_dir)
    print("task_name")
    print(args.task_name)
    print("do_sample")
    print(args.do_sample)

    if args.task_name == "deception":
        contrastive_type = ("honest", "lying")
        answer_type = ("true", "false")
    elif args.task_name == "jailbreak" or args.task_name == "jailbreak_choice":
        contrastive_type = ("HHH", args.jailbreak)
        answer_type = ("harmless", "harmful")

    print("answer_type")
    print(answer_type)
    print("contrastive_type")
    print(contrastive_type)

    if args.cache_name == "kqv":
        cache_name = ["z", "q", "pattern", "v", "k"]
    else:
        cache_name = args.cache_name

    main(
        model_path=args.model_path,
        save_dir=args.save_dir,
        task_name=args.task_name,
        contrastive_type=contrastive_type,
        answer_type=answer_type,
        do_sample=args.do_sample,
        framework=args.framework,
        cache_name=cache_name,
        jailbreak=args.jailbreak,
    )
