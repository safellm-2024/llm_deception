import random
import json
import pprint
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from torch import Tensor
from jaxtyping import Float, Int, Bool
import einops
from fancy_einsum import einsum
import gc
import os
from typing_extensions import Literal
from tqdm import tqdm
from functools import partial

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
from src.submodules.model.load_model import load_model
from src.utils.utils import logits_to_logit_diff
from transformer_lens import utils, HookedTransformer, ActivationCache
from rich.table import Table, Column
from rich import print as rprint
from src.utils.plotly_utils import line, scatter, bar
from src.utils.plotly_utils_neel import imshow
from src.submodules.activation_pca.activation_pca import get_pca_and_plot
import plotly.io as pio
import circuitsvis as cv  # for visualize attetnion pattern

from IPython.display import display, HTML  # for visualize attetnion pattern


class AttributionPatching:
    def __init__(self, cfg, model, dataset):
        self.cfg = cfg
        self.model = model
        self.dataset = dataset
        self.clean_label = cfg.clean_label

        if self.clean_label == "positive":
            self.prompts_clean = self.dataset.prompts_positive
            self.prompts_corrupted = self.dataset.prompts_negative

            self.answers_clean_correct = self.dataset.answers_positive_correct
            self.answers_corrupted_correct = self.dataset.answers_negative_correct

            self.answers_clean_wrong = self.dataset.answers_positive_wrong
            self.answers_corrupted_wrong = self.dataset.answers_negative_wrong

            self.labels_clean_correct = self.dataset.labels_positive_correct
            self.labels_corrupted_correct = self.dataset.labels_negative_correct

            self.labels_clean_wrong = self.dataset.labels_positive_wrong
            self.labels_corrupted_wrong = self.dataset.labels_negative_wrong

        else:
            self.prompts_clean = self.dataset.prompts_negative
            self.prompts_corrupted = self.dataset.prompts_positive

            self.answers_clean_correct = self.dataset.answers_negative_correct
            self.answers_corrupted_correct = self.dataset.answers_positive_correct

            self.answers_clean_wrong = self.dataset.answers_negative_wrong
            self.answers_corrupted_wrong = self.dataset.answers_positive_wrong

            self.labels_clean_correct = self.dataset.labels_negative_correct
            self.labels_corrupted_correct = self.dataset.labels_positive_correct

            self.labels_clean_wrong = self.dataset.labels_negative_wrong
            self.labels_corrupted_wrong = self.dataset.labels_positive_wrong

        # tokenize prompts and answer
        (
            self.prompt_tokens_clean,
            self.answer_tokens_clean_correct,
            self.answer_tokens_clean_wrong,
        ) = tokenize_prompts_and_answers(
            model,
            self.prompts_clean,
            self.answers_clean_correct,
            self.answers_clean_wrong,
        )
        (
            self.prompt_tokens_corrupted,
            self.answer_tokens_corrupted_correct,
            self.answer_tokens_corrupted_wrong,
        ) = tokenize_prompts_and_answers(
            model,
            self.prompts_corrupted,
            self.answers_corrupted_correct,
            self.answers_corrupted_wrong,
        )

        self.clean_baseline = None
        self.corrupted_baseline = None
        self.clean_cache = None
        self.corrupted_cache = None
        self.clean_grad_cache = None
        self.corrupted_grad_cache = None

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

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")
        print("test clean prompt:")
        utils.test_prompt(
            self.prompts_clean[0],
            self.answers_clean_correct[0],
            self.model,
            prepend_bos=True,
        )

        print("test corrupted prompt:")
        utils.test_prompt(
            self.prompts_corrupted[0],
            self.answers_corrupted_correct[0],
            self.model,
            prepend_bos=True,
        )
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")

        # save the visualization
        save_path = self.cfg.artifact_path()
        self.vis_path = save_path + os.sep + "attribution_patching"
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)

        print("Memory after initialization:")
        print(
            "free(Gb):",
            torch.cuda.mem_get_info()[0] / 1000000000,
            "total(Gb):",
            torch.cuda.mem_get_info()[1] / 1000000000,
        )
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")

    def get_contrastive_baseline_logit_diff(self) -> None:
        """
        get baseline logit differnece for lean and corrupted inputs
        :return:
        """
        print("logit diff of clean:")
        clean_logit_diff = self.run_and_get_logit_diff(
            self.prompt_tokens_clean,
            self.answer_tokens_clean_correct,
            self.answer_tokens_clean_wrong,
        )
        print("logit diff of corrupted:")
        corrupted_logit_diff = self.run_and_get_logit_diff(
            self.prompt_tokens_corrupted,
            self.answer_tokens_corrupted_correct,
            self.answer_tokens_corrupted_wrong,
        )

        self.clean_baseline = clean_logit_diff
        self.corrupted_baseline = corrupted_logit_diff

        clean_baseline_metric = (clean_logit_diff - self.corrupted_baseline) / (
            self.clean_baseline - self.corrupted_baseline
        )
        corrupted_baseline_metric = (corrupted_logit_diff - self.corrupted_baseline) / (
            self.clean_baseline - self.corrupted_baseline
        )

        print(f"clean baseline metric is: {clean_baseline_metric}")
        print(f"corrupted baseline metric is: {corrupted_baseline_metric}")

    def run_and_get_logit_diff(
        self, tokens, answer_tokens_correct, answer_tokens_wrong
    ):
        n_prompts = len(tokens)
        logit_diff_all = []
        for batch in tqdm(range(0, n_prompts, self.cfg.batch_size)):

            logits = self.model(tokens[batch : batch + self.cfg.batch_size])

            # get logits difference between clean and corrupted
            logit_diff = logits_to_logit_diff(
                logits,
                answer_tokens_correct[batch : batch + self.cfg.batch_size],
                answer_tokens_wrong[batch : batch + self.cfg.batch_size],
            ).item()
            logit_diff_all.append(logit_diff)

        logit_diff_all = sum(logit_diff_all) / len(
            logit_diff_all
        )  # mean logit diff of all batches
        print(f"logit diff: {logit_diff_all:.4f}")

        # clear the cache to save space
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")
        print("before delete logits:")
        print(
            "free(Gb):",
            torch.cuda.mem_get_info()[0] / 1000000000,
            "total(Gb):",
            torch.cuda.mem_get_info()[1] / 1000000000,
        )
        del logits
        gc.collect()
        torch.cuda.empty_cache()
        print("after delete logits:")
        print(
            "free(Gb):",
            torch.cuda.mem_get_info()[0] / 1000000000,
            "total(Gb):",
            torch.cuda.mem_get_info()[1] / 1000000000,
        )
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")

        return logit_diff

    def contrastive_metric(
        self, logits: Float[Tensor, "batch seq d_vocab"], batch=None, batch_size=None
    ) -> Tensor:
        """
        Scale the logit difference between 0 and 1 (clean baseline will have logit diff =1, and corrupted baseline will have logit diff =0)
        :param logits:
        :return:
        """
        if batch is None:
            return (
                logits_to_logit_diff(
                    logits,
                    self.answer_tokens_clean_correct,
                    self.answer_tokens_clean_wrong,
                )
                - self.corrupted_baseline
            ) / (self.clean_baseline - self.corrupted_baseline)
        else:
            return (
                logits_to_logit_diff(
                    logits,
                    self.answer_tokens_clean_correct[batch : batch + batch_size],
                    self.answer_tokens_clean_wrong[batch : batch + batch_size],
                )
                - self.corrupted_baseline
            ) / (self.clean_baseline - self.corrupted_baseline)

    def get_contrastive_activation_and_gradient_cache(self):
        """
        Get the activation cache and gradient cache for both the clean and corrupted prompts
        :return:
        """

        clean_cache, clean_grad_cache = self.run_and_cache_fwd_bwd_batch(
            self.prompt_tokens_corrupted, cache_bwd=False
        )

        corrupted_cache, corrupted_grad_cache = self.run_and_cache_fwd_bwd_batch(
            self.prompt_tokens_corrupted, cache_bwd=True
        )

        self.clean_cache = clean_cache
        self.clean_grad_cache = clean_grad_cache

        self.corrupted_cache = corrupted_cache
        self.corrupted_grad_cache = corrupted_grad_cache
        print(
            "free(Gb):",
            torch.cuda.mem_get_info()[0] / 1000000000,
            "total(Gb):",
            torch.cuda.mem_get_info()[1] / 1000000000,
        )
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")

    def get_contrastive_cache_plot_pca(self, pos=-1):

        # 1. Get access to cache
        activations_positive = [
            self.clean_cache[utils.get_act_name("resid_post", l)][:, pos, :].squeeze()
            for l in range(self.model.cfg.n_layers)
        ]

        activations_negative = [
            self.corrupted_cache[utils.get_act_name("resid_post", l)][
                :, pos, :
            ].squeeze()
            for l in range(self.model.cfg.n_layers)
        ]

        # stack the list of activations
        activations_positive = torch.stack(
            activations_positive
        )  # shape = [batch,d_model]
        activations_negative = torch.stack(activations_negative)

        # 2. Get pca and plot
        fig = get_pca_and_plot(
            self.cfg, activations_positive, activations_negative, self.dataset
        )
        pca_path = os.path.join(self.cfg.artifact_path(), "pca")
        if not os.path.exists(pca_path):
            os.makedirs(pca_path)
        fig.write_html(pca_path + os.sep + f"{self.cfg.model_alias}_pca.html")
        pio.write_image(
            fig, pca_path + os.sep + f"{self.cfg.model_alias}_pca.png", scale=6
        )

    def run_and_cache_fwd_bwd_batch(self, prompt_tokens, cache_bwd=True):
        cache_all, grad_cache_all = self.initialize_cache()
        n_prompts = len(prompt_tokens)
        for batch in tqdm(range(1, n_prompts, self.cfg.batch_size)):
            contrastive_metric_batch = partial(
                self.contrastive_metric, batch=batch, batch_size=self.cfg.batch_size
            )
            value, cache, grad_cache = get_cache_fwd_and_bwd(
                self.model,
                contrastive_metric_batch,
                prompt_tokens[batch : batch + self.cfg.batch_size],
                cache_bwd=cache_bwd,
            )
            for key in cache.keys():
                cache_all.cache_dict[key] = torch.cat(
                    (cache_all[key], cache[key]), dim=0
                )
            for key in grad_cache.keys():
                grad_cache_all.cache_dict[key] = torch.cat(
                    (grad_cache_all[key], grad_cache[key]), dim=0
                )
        del cache
        del grad_cache
        gc.collect()
        torch.cuda.empty_cache()
        return cache_all, grad_cache_all

    def get_attention_attribution(
        self,
    ) -> Float[Tensor, "batch layer head_inde dest src"]:
        attention_stack = torch.stack(
            [self.clean_cache["pattern", l] for l in range(self.model.cfg.n_layers)],
            dim=0,
        )
        attention_grad_stack = torch.stack(
            [
                self.clean_grad_cache["pattern", l]
                for l in range(self.model.cfg.n_layers)
            ],
            dim=0,
        )
        attention_attr = attention_grad_stack * attention_stack
        attention_attr = einops.rearrange(
            attention_attr,
            "layer batch head_index dest src -> batch layer head_index dest src",
        )
        print(
            "free(Gb):",
            torch.cuda.mem_get_info()[0] / 1000000000,
            "total(Gb):",
            torch.cuda.mem_get_info()[1] / 1000000000,
        )
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")

        return attention_attr

    def plot_attention_attr(
        self, attention_attr, tokens, top_k=20, index=0, title="", save_path=None
    ):

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
        with open(self.vis_path + os.sep + "attention_attribution.pkl", "wb") as f:
            pickle.dump(results, f)
        # vis = pysvelte.AttentionMulti(
        #     tokens=self.model.to_str_tokens(tokens),
        #     attention=attention_attr_signed.permute(1, 2, 0)[:, :, :top_k],
        #     head_labels=head_labels[:top_k],
        # ).publish(dev=True)
        # vis.publish(vis_path + os.sep + f"top20.html", dev_host="try")
        # Display results

        vis = cv.attention.attention_patterns(
            attention=attention_attr_signed[:top_k, :, :],
            tokens=self.model.to_str_tokens(tokens[:]),
            attention_head_names=head_labels[:top_k],
        )

        # save the visualization as .html file
        with open(
            self.vis_path
            + os.sep
            + f"attention_attribution_clean_{self.clean_label}.html",
            "w",
        ) as f:
            f.write(vis._repr_html_())

    def attr_patch_residual(
        self,
    ) -> TT["component", "pos"]:
        clean_residual, residual_labels = self.clean_cache.accumulated_resid(
            -1, incl_mid=True, return_labels=True
        )
        corrupted_residual = self.corrupted_cache.accumulated_resid(
            -1, incl_mid=True, return_labels=False
        )
        corrupted_grad_residual = self.corrupted_grad_cache.accumulated_resid(
            -1, incl_mid=True, return_labels=False
        )
        # clean_residual shape [component batch seq d_model]
        residual_attr = einops.reduce(
            corrupted_grad_residual * (clean_residual - corrupted_residual),
            "component batch pos d_model -> component pos",
            "sum",
        )

        # Plot
        fig = imshow(
            residual_attr,
            y=residual_labels,
            yaxis="Component",
            xaxis="Position",
            title="Residual Attribution Patching",
            return_fig=True,
        )
        # save the visualization as .html file
        fig.write_html(
            self.vis_path
            + os.sep
            + f"accumulated_layer_attribution_patching_clean_{self.clean_label}.html"
        )
        fig.write_image(
            self.vis_path
            + os.sep
            + f"accumulated_layer_attribution_patching_clean_{self.clean_label}.pdf"
        )
        string_0 = self.model.to_str_tokens(self.prompt_tokens_clean[0, :])
        print(f"example string token: {string_0}")
        print(
            "free(Gb):",
            torch.cuda.mem_get_info()[0] / 1000000000,
            "total(Gb):",
            torch.cuda.mem_get_info()[1] / 1000000000,
        )
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")
        return residual_attr, residual_labels

    def attr_patch_layer_out(
        self,
    ) -> TT["component", "pos"]:
        clean_layer_out, labels = self.clean_cache.decompose_resid(
            -1, return_labels=True
        )
        corrupted_layer_out = self.corrupted_cache.decompose_resid(
            -1, return_labels=False
        )
        corrupted_grad_layer_out = self.corrupted_grad_cache.decompose_resid(
            -1, return_labels=False
        )
        layer_out_attr = einops.reduce(
            corrupted_grad_layer_out * (clean_layer_out - corrupted_layer_out),
            "component batch pos d_model -> component pos",
            "sum",
        )

        fig = imshow(
            layer_out_attr,
            y=labels,
            yaxis="Component",
            xaxis="Position",
            title="Layer Output Attribution Patching",
            return_fig=True,
        )
        fig.write_html(
            self.vis_path
            + os.sep
            + f"decomposed_layer_attribution_patching_clean_{self.clean_label}.html"
        )
        string_0 = self.model.to_str_tokens(self.prompt_tokens_clean[0, :])
        print(f"example string token: {string_0}")
        print(
            "free(Gb):",
            torch.cuda.mem_get_info()[0] / 1000000000,
            "total(Gb):",
            torch.cuda.mem_get_info()[1] / 1000000000,
        )
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")

        return layer_out_attr, labels

    def attr_patch_head_out(
        self,
    ) -> TT["component", "pos"]:
        clean_layer_out, labels = self.clean_cache.stack_head_results(
            -1, return_labels=True
        )
        corrupted_layer_out = self.corrupted_cache.stack_head_results(
            -1, return_labels=False
        )
        corrupted_grad_layer_out = self.corrupted_grad_cache.stack_head_results(
            -1, return_labels=False
        )
        head_out_attr = einops.reduce(
            corrupted_grad_layer_out * (clean_layer_out - corrupted_layer_out),
            "component batch pos d_model -> component pos",
            "sum",
        )

        fig = imshow(
            head_out_attr,
            y=labels,
            yaxis="Component",
            xaxis="Position",
            title="Head Output Attribution Patching",
            return_fig=True,
        )
        fig.write_html(
            self.vis_path
            + os.sep
            + f"head_attribution_patching_clean_{self.clean_label}.html"
        )
        sum_head_out_attr = einops.reduce(
            head_out_attr,
            "(layer head) pos -> layer head",
            "sum",
            layer=self.model.cfg.n_layers,
            head=self.model.cfg.n_heads,
        )
        fig = imshow(
            sum_head_out_attr,
            yaxis="Layer",
            xaxis="Head Index",
            title="Head Output Attribution Patching Sum Over Pos",
            return_fig=True,
        )
        fig.write_html(
            self.vis_path
            + os.sep
            + f"head_sum_pos_attribution_patching_clean_{self.clean_label}.html"
        )
        string_0 = self.model.to_str_tokens(self.prompt_tokens_clean[0, :])
        print(f"example string token: {string_0}")

        print(
            "free(Gb):",
            torch.cuda.mem_get_info()[0] / 1000000000,
            "total(Gb):",
            torch.cuda.mem_get_info()[1] / 1000000000,
        )
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")

        return head_out_attr

    def attr_patch_head_kqv(self):
        head_vector_attr_dict = {}
        for activation_name, activation_name_full in [
            ("k", "Key"),
            ("q", "Query"),
            ("v", "Value"),
            ("z", "Mixed Value"),
        ]:
            display(
                Markdown(
                    f"#### {activation_name_full} Head Vector Attribution Patching"
                )
            )
            head_vector_attr_dict[activation_name], head_vector_labels = (
                self.attr_patch_head_vector(activation_name)
            )

            # plot all pos
            fig = imshow(
                head_vector_attr_dict[activation_name],
                y=head_vector_labels,
                yaxis="Component",
                xaxis="Position",
                title=f"{activation_name_full} Attribution Patching",
                return_fig=True,
            )
            fig.write_html(
                self.vis_path
                + os.sep
                + f"{activation_name_full}_attribution_patching_clean_{self.clean_label}.html"
            )
            # plot sum over pos
            sum_head_vector_attr = einops.reduce(
                head_vector_attr_dict[activation_name],
                "(layer head) pos -> layer head",
                "sum",
                layer=self.model.cfg.n_layers,
                head=self.model.cfg.n_heads,
            )
            fig = imshow(
                sum_head_vector_attr,
                yaxis="Layer",
                xaxis="Head Index",
                title=f"{activation_name_full} Attribution Patching Sum Over Pos",
                return_fig=True,
            )
            fig.write_html(
                self.vis_path
                + os.sep
                + f"{activation_name_full}_sum_pos_attribution_patching_clean_{self.clean_label}.html"
            )

    def attr_patch_head_vector(
        self,
        activation_name: Literal["q", "k", "v", "z"],
    ) -> TT["component", "pos"]:
        labels = self.head_names

        clean_head_vector = self.stack_head_vector_from_cache(
            self.clean_cache, activation_name
        )
        corrupted_head_vector = self.stack_head_vector_from_cache(
            self.corrupted_cache, activation_name
        )
        corrupted_grad_head_vector = self.stack_head_vector_from_cache(
            self.corrupted_grad_cache, activation_name
        )
        head_vector_attr = einops.reduce(
            corrupted_grad_head_vector * (clean_head_vector - corrupted_head_vector),
            "component batch pos d_head -> component pos",
            "sum",
        )
        return head_vector_attr

    def stack_head_vector_from_cache(
        self, cache, activation_name: Literal["q", "k", "v", "z"]
    ) -> TT["layer_and_head_index", "batch", "pos", "d_head"]:
        """Stacks the head vectors from the cache from a specific activation (key, query, value or mixed_value (z)) into a single tensor."""
        stacked_head_vectors = torch.stack(
            [cache[activation_name, l] for l in range(self.model.cfg.n_layers)], dim=0
        )
        stacked_head_vectors = einops.rearrange(
            stacked_head_vectors,
            "layer batch pos head_index d_head -> (layer head_index) batch pos d_head",
        )
        return stacked_head_vectors

    def attr_patch_head_path(
        self,
    ) -> TT["qkv", "dest_component", "src_component", "pos"]:
        """
        Computes the attribution patch along the path between each pair of heads.

        Sets this to zero for the path from any late head to any early head

        """
        start_labels = self.head_names
        end_labels = self.head_nams_qkv

        # get the corrupted gradient cache per head for kqv
        full_vector_grad_late_node = self.get_full_vector_grad_late_node(
            self.corrupted_grad_cache
        )

        # get the clean activation cache per head
        clean_head_stack_early_node = self.clean_cache.stack_head_results(-1)
        # get the corrupted activation cache per head
        corrupted_head_stack_early_node = self.corrupted_cache.stack_head_results(-1)

        # difference in clean and corrupted
        diff_head_early_node = einops.rearrange(
            clean_head_stack_early_node - corrupted_head_stack_early_node,
            "(layer head_index) batch pos d_model -> layer batch pos head_index d_model",
            layer=self.model.cfg.n_layers,
            head_index=self.model.cfg.n_heads,
        )
        # clear out memory:
        print("before del:")
        print(
            "free(Gb):",
            torch.cuda.mem_get_info()[0] / 1000000000,
            "total(Gb):",
            torch.cuda.mem_get_info()[1] / 1000000000,
        )
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")

        del self.clean_cache
        del self.clean_grad_cache
        del self.corrupted_cache
        del self.corrupted_grad_cache
        gc.collect()
        torch.cuda.empty_cache()
        print("after del:")
        print(
            "free(Gb):",
            torch.cuda.mem_get_info()[0] / 1000000000,
            "total(Gb):",
            torch.cuda.mem_get_info()[1] / 1000000000,
        )
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")

        # multiply the difference by the gradeint
        path_attr = einsum(
            "qkv layer_end batch pos head_end d_model, layer_start batch pos head_start d_model -> qkv layer_end head_end layer_start head_start pos",
            full_vector_grad_late_node,
            diff_head_early_node,
        )

        # what does this do???
        correct_layer_order_mask = (
            torch.arange(self.model.cfg.n_layers)[None, :, None, None, None, None]
            > torch.arange(self.model.cfg.n_layers)[None, None, None, :, None, None]
        ).to(path_attr.device)

        # replace late to early with zero
        zero = torch.zeros(1, device=path_attr.device)
        path_attr = torch.where(
            correct_layer_order_mask, path_attr, zero
        )  # condition, input, other

        # rearrange to 3d
        path_attr = einops.rearrange(
            path_attr,
            "qkv layer_end head_end layer_start head_start pos -> (layer_end head_end qkv) (layer_start head_start) pos",
        )
        fig = imshow(
            path_attr.sum(-1),  # sum over all position
            y=end_labels,
            yaxis="Path End (Head Input)",
            x=start_labels,
            xaxis="Path Start (Head Output)",
            title="Head Path Attribution Patching",
            return_fig=True,
        )
        fig.write_html(
            self.vis_path
            + os.sep
            + f"path_head_attribution_patching_clean_{self.clean_label}.html"
        )
        return path_attr

    def get_head_vector_grad_input_from_grad_cache(
        self,
        grad_cache: ActivationCache,
        activation_name: Literal["q", "k", "v"],
        layer: int,
    ) -> Float[Tensor, "batch pos head_index d_model"]:
        """
        We want to get the gradient with respect to key multiply by key.
        What we got is the residual stream

        key = residual_stream @ W_K + b_K, then residual_stream_grad_mediated_by_key = W_K @ key
        :param grad_cache:
        :param activation_name:
        :param layer:
        :return:
        """
        vector_grad = grad_cache[activation_name, layer]  # get access to the k,q, v
        ln_scales = grad_cache["scale", layer, "ln1"]  # get access to scale
        attn_layer_object = self.model.blocks[layer].attn
        if activation_name == "q":
            W = attn_layer_object.W_Q
        elif activation_name == "k":
            W = attn_layer_object.W_K
        elif activation_name == "v":
            W = attn_layer_object.W_V
        else:
            raise ValueError("Invalid activation name")

        if self.cfg.dtype == "bfloat16":
            return einsum(
                "batch pos head_index d_head, batch pos head_index, head_index d_model d_head -> batch pos head_index d_model",
                vector_grad.to(torch.bfloat16),
                ln_scales.squeeze(-1).to(torch.bfloat16),
                W,
            )
        else:
            return einsum(
                "batch pos head_index d_head, batch pos head_index, head_index d_model d_head -> batch pos head_index d_model",
                vector_grad,
                ln_scales.squeeze(-1),
                W,
            )

    def get_stacked_head_vector_grad_late_node(
        self, grad_cache, activation_name: Literal["q", "k", "v"]
    ) -> Float[Tensor, "layer batch pos head_index d_model"]:
        return torch.stack(
            [
                self.get_head_vector_grad_input_from_grad_cache(
                    grad_cache, activation_name, l
                )
                for l in range(self.model.cfg.n_layers)
            ],
            dim=0,
        )

    def get_full_vector_grad_late_node(
        self,
        grad_cache,
    ) -> Float[Tensor, "qkv layer batch pos head_index d_model"]:
        return torch.stack(
            [
                self.get_stacked_head_vector_grad_late_node(grad_cache, activation_name)
                for activation_name in ["q", "k", "v"]
            ],
            dim=0,
        )

    def top_heads_plot_path_attr(self, head_out_attr, head_path_attr):
        start_labels = self.head_names
        end_labels = self.head_nams_qkv

        head_out_values, head_out_indices = (
            head_out_attr.sum(-1).abs().sort(descending=True)
        )
        line(head_out_values)

        top_head_indices = head_out_indices[:22].sort().values
        top_end_indices = []
        top_end_labels = []
        top_start_indices = []
        top_start_labels = []
        for i in top_head_indices:
            i = i.item()
            top_start_indices.append(i)
            top_start_labels.append(start_labels[i])
            for j in range(3):
                top_end_indices.append(3 * i + j)
                top_end_labels.append(end_labels[3 * i + j])

        imshow(
            head_path_attr[top_end_indices, :][:, top_start_indices].sum(-1),
            y=top_end_labels,
            yaxis="Path End (Head Input)",
            x=top_start_labels,
            xaxis="Path Start (Head Output)",
            title="Head Path Attribution Patching (Filtered for Top Heads)",
        )

        for j, composition_type in enumerate(["Query", "Key", "Value"]):
            imshow(
                head_path_attr[top_end_indices, :][:, top_start_indices][j::3].sum(-1),
                y=top_end_labels[j::3],
                yaxis="Path End (Head Input)",
                x=top_start_labels,
                xaxis="Path Start (Head Output)",
                title=f"Head Path to {composition_type} Attribution Patching (Filtered for Top Heads)",
            )

        # subplots
        # todo: understand this!
        top_head_path_attr = einops.rearrange(
            head_path_attr[top_end_indices, :][:, top_start_indices].sum(-1),
            "(head_end qkv) head_start -> qkv head_end head_start",
            qkv=3,
        )
        fig = imshow(
            top_head_path_attr,
            y=[i[:-1] for i in top_end_labels[::3]],
            yaxis="Path End (Head Input)",
            x=top_start_labels,
            xaxis="Path Start (Head Output)",
            title=f"Head Path Attribution Patching (Filtered for Top Heads)",
            facet_col=0,
            facet_labels=["Query", "Key", "Value"],
            return_fig=True,
        )
        fig.write_html(
            self.vis_path
            + os.sep
            + f"top_head_path_attribution_patching_clean_{self.clean_label}.html"
        )

    def initialize_cache(self):
        # initialize cache
        contrastive_metric_batch = partial(
            self.contrastive_metric, batch=0, batch_size=1
        )
        value, cache, grad_cache = get_cache_fwd_and_bwd(
            self.model,
            contrastive_metric_batch,
            self.prompt_tokens_clean[0 : 0 + 1],
        )

        return cache, grad_cache


def run_with_cache_contrastive(model, positive_tokens, negative_tokens):

    # Run the model and cache all activations
    positive_logits, positive_cache = model.run_with_cache(positive_tokens)
    negative_logits, negative_cache = model.run_with_cache(negative_tokens)

    return positive_logits, negative_logits, positive_cache, negative_cache


def run_model_contrastive(model, positive_tokens, negative_tokens):

    # Run the model and cache all activations
    positive_logits = model(positive_tokens)
    negative_logits = model(negative_tokens)

    return positive_logits, negative_logits


def get_cache_fwd_and_bwd(model, metric, tokens, cache_bwd=True):
    # Always reset hook before you start -- good habit!
    model.reset_hooks()

    # (1) Create filter for what to cache
    filter_not_qkv_input = lambda name: "_input" not in name and (
        "attn" in name or "scale" in name or "resid_post" in name
    )

    # (2) Construct hook functions

    # forward cache hook: cache the activation during forward run
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    # backward cache hook: : cache the activation during backward run
    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    # (3) Construct hooks
    model.add_hook(filter_not_qkv_input, forward_cache_hook, "fwd")
    if cache_bwd:
        model.add_hook(filter_not_qkv_input, backward_cache_hook, "bwd")

    # (4) Run forward pass
    logits = model(tokens)

    # (5) Get loss function (our contrastive metric)
    value = metric(logits)

    # (6) Backward pass
    value.backward()  # the loss for computing the backward pass

    # Always reset hook after you are done! -- good habit!
    model.reset_hooks()

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")
    print("before delete logits:")
    print(
        "free(Gb):",
        torch.cuda.mem_get_info()[0] / 1000000000,
        "total(Gb):",
        torch.cuda.mem_get_info()[1] / 1000000000,
    )
    del logits
    gc.collect()
    torch.cuda.empty_cache()
    print("after delete logits:")
    print(
        "free(Gb):",
        torch.cuda.mem_get_info()[0] / 1000000000,
        "total(Gb):",
        torch.cuda.mem_get_info()[1] / 1000000000,
    )
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")
    return (
        value.item(),
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )
