from pipeline.configs.config_generation import Config
import pickle
import json
import os
import argparse
import numpy as np
import sae_lens
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pprint
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from sklearn.metrics import confusion_matrix
from src.submodules.stage_statistics.stage_statistics import get_state_quantification


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model_base path argument.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="google/gemma-2-2b-it"
    )
    parser.add_argument(
        "--save_dir", type=str, required=False, default="D:\Data\deception"
    )
    parser.add_argument("--task_name", type=str, required=False, default="deception")
    parser.add_argument("--do_sample", type=bool, required=False, default=False)
    parser.add_argument("--framework", type=str, required=False, default="transformers")

    return parser.parse_args()


def main(
    model_path: str,
    save_dir: str,
    task_name: str = "deception",
    contrastive_type: Tuple[str] = ("honest", "lying"),
    answer_type: Tuple[str] = ("true", "false"),
    do_sample: bool = False,
    framework: str = "transformer_lens",
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
    )
    pprint.pp(cfg)

    # load
    load_path = os.path.join(cfg.artifact_path(), "activations")
    if not os.path.exists(load_path):
        os.makedirs(load_path)
    with open(load_path + os.sep + f"{cfg.model_alias}_activations.pkl", "rb") as f:
        activations = pickle.load(f)
    activations_positive = activations["activations_positive"]
    activations_negative = activations["activations_negative"]
    labels = activations["labels"]

    # do stage statistics
    get_state_quantification(
        cfg,
        activations_positive,
        activations_negative,
        labels,
        save_plot=True,
    )


if __name__ == "__main__":
    args = parse_arguments()
    print(sae_lens.__version__)
    print(sae_lens.__version__)
    print("run_pipeline_contrastive_generation\n\n")
    print("model_path")
    print(args.model_path)
    print("save_dir")
    print(args.save_dir)
    print("task_name")
    print(args.task_name)

    if args.task_name == "deception":
        contrastive_type = ("honest", "lying")
        answer_type = ("true", "false")
    elif args.task_name == "jailbreak":
        contrastive_type = ("HHH", args.jailbreak)
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
    )
