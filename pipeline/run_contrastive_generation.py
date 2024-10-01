import random
import json
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
import torch
from typing import Tuple

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


def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    category = "facts"

    dataset_all = load_dataset("notrichardren/azaria-mitchell-diff-filtered-2")
    dataset = [row for row in dataset_all[f"{category}"]]
    dataset_train = random.sample(dataset, cfg.n_train)
    dataset_test = random.sample(dataset, cfg.n_test)

    return dataset_train, dataset_test


def generate_without_steering(
    cfg, model, statements, labels: list[int], contrastive_label: str
):
    """
    Generate model output without steering
    :param cfg:
    :param model:
    :param statements:
    :param labels:
    :param contrastive_label:
    :return:
    """
    max_new_tokens = cfg.max_new_tokens
    batch_size = cfg.batch_size
    task_name = cfg.task_name

    artifact_dir = cfg.artifact_path()
    save_path = os.path.join(artifact_dir, "completions")

    completions = []
    for ii in tqdm(range(0, len(statements), batch_size)):

        # 1. From prompt to input
        prompt = construct_prompt(
            cfg, statements[ii : ii + batch_size], contrastive_label=contrastive_label
        )
        input_ids = model.to_tokens(prompt, prepend_bos=model.cfg.default_prepend_bos)

        # 4. Generation
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.1,
            freq_penalty=1.0,
            # stop_at_eos = False if device == "mps" else True,
            stop_at_eos=False,
            prepend_bos=model.cfg.default_prepend_bos,
        )

        # 5. Get generation output (one batch)
        generation_toks = output[:, input_ids.shape[-1] :]
        for generation_idx, generation in enumerate(generation_toks):
            completions.append(
                {
                    "prompt": statements[ii + generation_idx],
                    "response": model.tokenizer.decode(
                        generation, skip_special_tokens=True
                    ).strip(),
                    "label": labels[ii + generation_idx],
                    "ID": ii + generation_idx,
                }
            )

    # 6. Store all generation results (all batches)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + os.sep + f"completions_{contrastive_label}.json", "w") as f:
        json.dump(completions, f, indent=4)
    return completions


def contrastive_generation(cfg, model, dataset):

    statements = [row["claim"] for row in dataset]
    labels = [row["label"] for row in dataset]

    contrastive_label = cfg.contrastive_label

    # positive
    print("generation positive:")
    completions_positive = generate_without_steering(
        cfg, model, statements, labels, contrastive_label[0]
    )
    # negative
    print("generation negative:")
    completions_negative = generate_without_steering(
        cfg, model, statements, labels, contrastive_label[1]
    )
    return completions_positive, completions_negative


def main(
    model_path: str,
    save_dir: str,
    task_name: str = "deception",
    contrastive_label: Tuple[str] = ("honest", "lying"),
):

    model_alias = os.path.basename(model_path)

    cfg = Config(
        model_path=model_path,
        model_alias=model_alias,
        task_name=task_name,
        contrastive_type=contrastive_label,
        save_dir=save_dir,
    )

    # 1. Load Model
    print("loading model")
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_path, device=device, dtype=torch.bfloat16
    )
    model.tokenizer.padding_side = "left"
    print(
        "free(Gb):",
        torch.cuda.mem_get_info()[0] / 1000000000,
        "total(Gb):",
        torch.cuda.mem_get_info()[1] / 1000000000,
    )

    # 2. Load data
    print("loading data")
    dataset_train, dataset_test = load_and_sample_datasets(cfg)

    # 3. Run generation
    print("generation start")
    contrastive_generation(cfg, model, dataset_train)

    # 4. evaluate performance
    completions_honest = evaluate_generation_deception(
        cfg, contrastive_label=contrastive_label[0]
    )
    completions_lying = evaluate_generation_deception(
        cfg, contrastive_label=contrastive_label[1]
    )

    # 5. plot performance
    print("plot performance")
    plot_lying_honest_performance(cfg, completions_honest, completions_lying)
    print("done")


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
        contrastive_label = ["honest", "lying"]
    elif args.task_name == "jailbreak":
        contrastive_label = ["HHH", args.jailbreak]

    main(
        model_path=args.model_path,
        save_dir=args.save_dir,
        task_name=args.task_name,
        contrastive_label=contrastive_label,
    )
