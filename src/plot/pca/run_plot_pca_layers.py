import os
import argparse

# from pipeline.configs.honesty_config_generation_intervention import Config
from pipeline.configs.config_generation import Config
from src.plot.pca.plot_some_layer_pca import plot_contrastive_activation_pca_layer
import pickle
import plotly.io as pio
import pprint
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--save_dir", type=str, required=False, default="D:\Data\deception"
    )
    parser.add_argument("--task_name", type=str, required=False, default="deception")
    parser.add_argument("--layer_plot", nargs="+", type=int)
    parser.add_argument("--do_sample", type=bool, required=False, default=True)
    parser.add_argument("--framework", type=str, required=False, default="transformers")

    return parser.parse_args()


def main(
    model_path: str,
    save_dir: str,
    task_name: str = "deception",
    contrastive_type: Tuple[str] = ("honest", "lying"),
    input_type: Tuple[str] = ("true", "false"),
    do_sample: bool = False,
    framework: str = "transformer_lens",
    layer_plot: tuple = (10, 11, 12),
):
    # 0. Set configuration
    model_alias = os.path.basename(model_path)
    cfg = Config(
        model_path=model_path,
        model_alias=model_alias,
        task_name=task_name,
        contrastive_type=contrastive_type,
        answer_type=input_type,
        save_dir=save_dir,
        do_sample=do_sample,
        framework=framework,
    )
    pprint.pp(cfg)

    artifact_path = cfg.artifact_path()

    # 2. Load pca
    filename = (
        artifact_path
        + os.sep
        + "activations"
        + os.sep
        + model_alias
        + "_activations.pkl"
    )
    with open(filename, "rb") as file:
        data = pickle.load(file)

    activations_positive = data["activations_positive"]
    activations_negative = data["activations_negative"]
    input_label = data["labels"]

    # 5. plot
    save_dir = os.path.join(cfg.artifact_path(), "pca")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig = plot_contrastive_activation_pca_layer(
        activations_positive,
        activations_negative,
        contrastive_type,
        input_label=input_label,
        input_type=input_type,
        layers=layer_plot,
    )

    pio.write_image(
        fig,
        save_dir + os.sep + str(layer_plot) + "_pca.pdf",
    )


if __name__ == "__main__":
    args = parse_arguments()

    if args.task_name == "deception":
        contrastive_type = ("honest", "lying")
        input_type = ("true", "false")
    elif args.task_name == "jailbreak":
        contrastive_type = ("HHH", args.jailbreak)
    print("input_type")
    print(input_type)
    print("contrastive_type")
    print(contrastive_type)
    main(
        model_path=args.model_path,
        save_dir=args.save_dir,
        task_name=args.task_name,
        contrastive_type=contrastive_type,
        input_type=input_type,
        do_sample=args.do_sample,
        framework=args.framework,
        layer_plot=tuple(args.layer_plot),
    )
