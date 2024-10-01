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
from src.utils.hook_utils import add_hooks
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
from src.utils.utils import logits_to_ave_logit_diff
from transformer_lens import utils, HookedTransformer, ActivationCache
from rich.table import Table, Column
from rich import print as rprint
from src.utils.plotly_utils import line, scatter, bar
from src.utils.plotly_utils_neel import imshow
from src.submodules.activation_pca.activation_pca import get_pca_and_plot
import plotly.io as pio
import circuitsvis as cv  # for visualize attetnion pattern

from IPython.display import display, HTML  # for visualize attetnion pattern
from src.submodules.evaluations.evaluate_deception import (
    evaluate_generation_deception,
    plot_lying_honest_performance,
)
from src.submodules.stage_statistics.stage_statistics import get_state_quantification
from src.submodules.evaluations.evaluate_jailbreak import evaluate_generation_jailbreak
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from src.utils.hook_utils import cache_activation_hook_post, cache_residual_hook_pre

from src.utils.hook_utils import cache_activation_post_hook_pos


class ContrastiveBase:
    def __init__(self, cfg, model, dataset):
        self.cfg = cfg
        self.model = model
        self.dataset = dataset
        self.contrastive_type = cfg.contrastive_type
        self.answer_type = cfg.answer_type

        self.cache_positive = None
        self.cache_negative = None
        self.logit_diff_clean = None
        self.logit_diff_corrupted = None

        self.answer_tokens_positive_correct = None
        self.answer_tokens_positive_wrong = None
        self.answer_tokens_negative_correct = None
        self.answer_tokens_negative_wrong = None

        self.prompt_tokens_negative = None
        self.prompt_masks_negative = None
        self.prompt_tokens_positive = None
        self.prompt_masks_positive = None

        self._get_tokenization()

        print("Memory after initialization:")
        print(
            "free(Gb):",
            torch.cuda.mem_get_info()[0] / 1000000000,
            "total(Gb):",
            torch.cuda.mem_get_info()[1] / 1000000000,
        )
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")

    def _get_tokenization(self):
        if (
            self.cfg.task_name == "deception"
            or self.cfg.task_name == "jailbreak_choice"
        ):
            # tokenize prompts and answer
            (
                self.prompt_tokens_positive,
                self.answer_tokens_positive_correct,
                self.answer_tokens_positive_wrong,
                self.prompt_masks_positive,
            ) = self.dataset.tokenize_prompts_and_answers(
                self.model,
                self.dataset.prompts_positive,
                self.dataset.answers_positive_correct,
                self.dataset.answers_positive_wrong,
            )
            (
                self.prompt_tokens_negative,
                self.answer_tokens_negative_correct,
                self.answer_tokens_negative_wrong,
                self.prompt_masks_negative,
            ) = self.dataset.tokenize_prompts_and_answers(
                self.model,
                self.dataset.prompts_negative,
                self.dataset.answers_negative_correct,
                self.dataset.answers_negative_wrong,
            )

        elif self.cfg.task_name == "jailbreak":
            (
                self.prompt_tokens_positive,
                self.prompt_masks_positive,
            ) = self.dataset.tokenize_prompts_and_answers(
                self.model,
                self.dataset.prompts_positive,
            )
            (
                self.prompt_tokens_negative,
                self.prompt_masks_negative,
            ) = self.dataset.tokenize_prompts_and_answers(
                self.model,
                self.dataset.prompts_negative,
            )

    def run_and_cache_contrastive_activation(self):
        """
        Get the activation cache and gradient cache for both the positive and negative prompts
        :return:
        """

        logits_positive, cache_positive = self.run_and_cache_batch(
            self.prompt_tokens_positive, self.prompt_masks_positive
        )

        logits_negative, cache_negative = self.run_and_cache_batch(
            self.prompt_tokens_negative, self.prompt_masks_negative
        )

        self.cache_positive = cache_positive
        self.cache_negative = cache_negative

        if (
            self.cfg.task_name == "deception"
            or self.cfg.task_name == "jailbreak_choice"
        ):
            self.get_logit_diff(logits_positive, logits_negative)

        print(
            "free(Gb):",
            torch.cuda.mem_get_info()[0] / 1000000000,
            "total(Gb):",
            torch.cuda.mem_get_info()[1] / 1000000000,
        )
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")

    def get_logit_diff(self, logits_positive, logits_negative):
        if self.cfg.clean_type == "positive":
            answer_tokens_correct = self.answer_tokens_positive_correct
            answer_tokens_wrong = self.answer_tokens_positive_wrong
        else:
            answer_tokens_correct = self.answer_tokens_negative_correct
            answer_tokens_wrong = self.answer_tokens_negative_wrong

        logits_diff_positive = logits_to_ave_logit_diff(
            logits_positive,
            answer_tokens_correct,
            answer_tokens_wrong,
        )
        logits_diff_negative = logits_to_ave_logit_diff(
            logits_negative,
            answer_tokens_correct,
            answer_tokens_wrong,
        )

        if self.cfg.clean_type == "positive":

            self.logit_diff_clean = logits_diff_positive
            self.logit_diff_corrupted = logits_diff_negative
        else:
            self.logtis_diff_clean = logits_diff_negative
            self.logit_diff_corrupted = logits_diff_positive

        print(f"logit diff clean: {self.logit_diff_clean}")
        print(f"logit diff corrupted: {self.logit_diff_corrupted}")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="/n/n")

    def generate_and_contrastive_activation_cache(self):
        """
        Get the activation cache and gradient cache for both the positive and negative prompts
        :return:
        """

        cache_positive = self.generate_and_cache_batch(
            self.dataset.prompts_positive,
            self.prompt_tokens_positive,
            self.prompt_masks_positive,
            self.cfg.contrastive_type[0],
        )

        cache_negative = self.generate_and_cache_batch(
            self.dataset.prompts_negative,
            self.prompt_tokens_negative,
            self.prompt_masks_negative,
            self.cfg.contrastive_type[1],
        )

        self.cache_positive = cache_positive

        self.cache_negative = cache_negative
        print(
            "free(Gb):",
            torch.cuda.mem_get_info()[0] / 1000000000,
            "total(Gb):",
            torch.cuda.mem_get_info()[1] / 1000000000,
        )
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")

    def get_contrastive_activation_pca(self, pos=-1):

        # 1. Get access to cache
        activations_positive, activations_negative = self.get_contrastive_cache()

        # 2. Get pca and plot
        fig = get_pca_and_plot(
            self.cfg, activations_positive, activations_negative, self.dataset
        )

        # 3. Save plot
        pca_path = os.path.join(self.cfg.artifact_path(), "pca")
        if not os.path.exists(pca_path):
            os.makedirs(pca_path)
        fig.write_html(pca_path + os.sep + f"{self.cfg.model_alias}_pca.html")
        pio.write_image(fig, pca_path + os.sep + f"{self.cfg.model_alias}_pca.pdf")

        if self.cfg.task_name == "deception":
            labels = self.dataset.labels_positive_correct
        elif (
            self.cfg.task_name == "jailbreak"
            or self.cfg.task_name == "jailbreak_choice"
        ):
            labels = self.dataset.input_labels

        # 4. Save activations
        activations = {
            "activations_positive": activations_positive,
            "activations_negative": activations_negative,
            "labels": labels,
        }
        activation_path = os.path.join(self.cfg.artifact_path(), "activations")
        if not os.path.exists(activation_path):
            os.makedirs(activation_path)
        with open(
            activation_path + os.sep + f"{self.cfg.model_alias}_activations.pkl", "wb"
        ) as f:
            pickle.dump(activations, f)

    def run_and_cache_batch(self, prompt_tokens, prompt_masks):

        if self.cfg.framework == "transformer_lens":
            logits, cache = run_and_cache(
                self.cfg,
                self.model,
                prompt_tokens,
            )

        else:
            logits, cache = run_and_cache(
                self.cfg,
                self.model,
                prompt_tokens,
                prompt_masks,
            )

        return logits, cache

    def generate_and_cache_batch(
        self, prompts, prompt_tokens, prompt_masks, contrastive_type
    ):

        # 1. Gen data
        if self.cfg.task_name == "deception":
            data = self.dataset.statements
            labels = self.dataset.answers_positive_correct
            prompt_type = "statement"
            category = [""] * len(data)

        elif self.cfg.task_name == "jailbreak":
            data = self.dataset.request
            labels = self.dataset.input_labels
            prompt_type = "request"
            category = self.dataset.category

        elif self.cfg.task_name == "jailbreak_choice":
            data = self.dataset.request
            prompt_type = "request"
            category = self.dataset.category
            labels = self.dataset.input_labels

        # 2. Generation initialization
        completions = []
        output, cache_all = self.initialize_cache(
            prompt_tokens,
            prompt_masks,
            mode="generate",
        )
        # Construct generation result for saving (one batch)
        generation_toks = output[:, prompt_tokens.shape[1] :]
        completions.append(
            {
                "prompt": prompts[0],
                f"{prompt_type}": data[0],
                "response": self.model.tokenizer.decode(
                    generation_toks[0], skip_special_tokens=True
                ).strip(),
                "label": labels[0],
                "category": category[0],
                "ID": 0,
            }
        )
        # 3. Main loop
        n_prompts = len(prompt_tokens)
        for batch in tqdm(range(1, n_prompts, self.cfg.batch_size)):

            if self.cfg.framework == "transformer_lens":
                output, cache = generate_and_cache(
                    self.cfg,
                    self.model,
                    prompt_tokens[batch : batch + self.cfg.batch_size],
                )
            else:
                output, cache = generate_and_cache(
                    self.cfg,
                    self.model,
                    prompt_tokens[batch : batch + self.cfg.batch_size],
                    masks=prompt_masks[batch : batch + self.cfg.batch_size],
                )

            # concatenate cache
            if self.cfg.framework == "transformer_lens":
                for key in cache.keys():
                    cache_all.cache_dict[key] = torch.cat(
                        (cache_all[key], cache[key]), dim=0
                    )
            else:
                cache_all = torch.cat((cache_all, cache), dim=1)

            # Construct generation result for saving (one batch)
            generation_toks = output[:, prompt_tokens.shape[1] :]
            for generation_idx, generation in enumerate(generation_toks):
                completions.append(
                    {
                        f"{prompt_type}": data[batch + generation_idx],
                        "response": self.model.tokenizer.decode(
                            generation, skip_special_tokens=True
                        ).strip(),
                        "label": labels[batch + generation_idx],
                        "ID": batch + generation_idx,
                        "category": category[batch + generation_idx],
                    }
                )

        # 3. Save completion
        completion_path = os.path.join(self.cfg.artifact_path(), "completions")
        if not os.path.exists(completion_path):
            os.makedirs(completion_path)
        with open(
            completion_path
            + os.sep
            + f"{self.cfg.model_alias}_completions_{contrastive_type}.json",
            "w",
        ) as f:
            json.dump(completions, f, indent=4)

        return cache_all

    def initialize_cache(self, prompt_tokens, prompt_masks, mode="run"):

        # initialize cache
        if mode == "run":
            if self.cfg.framework == "transformer_lens":
                logits, cache = run_and_cache(
                    self.cfg, self.model, prompt_tokens[0 : 0 + 1]
                )
            else:
                logits, cache = run_and_cache(
                    self.cfg,
                    self.model,
                    prompt_tokens[0 : 0 + 1],
                    prompt_masks[0 : 0 + 1],
                )
            return logits, cache
        else:
            if self.cfg.framework == "transformer_lens":

                output, cache = generate_and_cache(
                    self.cfg, self.model, prompt_tokens[0 : 0 + 1]
                )
            else:
                output, cache = generate_and_cache(
                    self.cfg,
                    self.model,
                    prompt_tokens[0 : 0 + 1],
                    masks=prompt_masks[0 : 0 + 1],
                )

            return output, cache

    def evaluate_deception(self):
        completions_honest = evaluate_generation_deception(
            self.cfg, contrastive_type=self.cfg.contrastive_type[0]
        )
        completions_lying = evaluate_generation_deception(
            self.cfg, contrastive_type=self.cfg.contrastive_type[1]
        )
        plot_lying_honest_performance(self.cfg, completions_honest, completions_lying)

    def evaluate_jailbreak(self):
        evaluate_generation_jailbreak(
            self.cfg, contrastive_type=self.cfg.contrastive_type[0]
        )
        evaluate_generation_jailbreak(
            self.cfg, contrastive_type=self.cfg.contrastive_type[1]
        )

    def get_stage_statistics(self):
        # 1. Get access to cache
        activations_positive, activations_negative = self.get_contrastive_cache()

        # 2. get stage statistics
        if (
            self.cfg.task_name == "deception"
            or self.cfg.task_name == "jailbreak_choice"
        ):
            labels = self.dataset.labels_positive_correct
        elif self.cfg.task_name == "jailbreak":
            labels = self.dataset.input_labels
        get_state_quantification(
            self.cfg,
            activations_positive,
            activations_negative,
            labels,
            save_plot=True,
        )

    def get_contrastive_cache(self):
        if self.cfg.framework == "transformer_lens":
            activations_positive = [
                self.cache_positive[
                    utils.get_act_name(self.cfg.cache_name, l)
                ].squeeze()
                for l in range(self.model.cfg.n_layers)
            ]

            activations_negative = [
                self.cache_negative[
                    utils.get_act_name(self.cfg.cache_name, l)
                ].squeeze()
                for l in range(self.model.cfg.n_layers)
            ]
            # stack the list of activations # shape = [batch,d_model]
            activations_positive = torch.stack(activations_positive)
            activations_negative = torch.stack(activations_negative)
        else:
            activations_positive = self.cache_positive
            activations_negative = self.cache_negative
        return activations_positive, activations_negative


def run_with_cache_contrastive(model, positive_tokens, negative_tokens):

    # Run the model and cache all activations
    positive_logits, cache_positive = model.run_with_cache(positive_tokens)
    negative_logits, cache_negative = model.run_with_cache(negative_tokens)

    return positive_logits, negative_logits, cache_positive, cache_negative


def run_model_contrastive(model, positive_tokens, negative_tokens):

    # Run the model and cache all activations
    positive_logits = model(positive_tokens)
    negative_logits = model(negative_tokens)

    return positive_logits, negative_logits


def generate_and_cache(cfg, model, tokens, masks=None):
    max_new_tokens = cfg.max_new_tokens
    do_sample = cfg.do_sample

    #  (1) Initialize
    if cfg.framework == "transformer_lens":
        # Always reset hook before you start -- good habit!
        model.reset_hooks()
        cache = {}
    else:
        n_layers = model.model.config.num_hidden_layers
        d_model = model.model.config.hidden_size
        n_samples = len(tokens)
        block_modules = model.model_block_modules
        len_prompt = tokens.shape[1]
        cache = torch.zeros(
            (n_layers, n_samples, d_model),
            dtype=torch.float16,
            device=model.device,
        )

    # (2) Construct hook functions
    # forward cache hook: cache the activation during forward run
    def forward_cache_hook(act, hook, len_prompt, cfg):
        # only cache generation result
        if cfg.cache_pos == "prompt_last_token":
            if act.shape[1] == len_prompt:
                cache[hook.name] = act[:, -1, :].detach()
        elif cfg.cache_pos == "prompt_all":
            if act.shape[1] == len_prompt:
                cache[hook.name] = act.detach()

    # (3) Construct hooks
    if cfg.framework == "transformer_lens":
        # Create filter for what to cache
        # activation_filter = lambda name: list(cfg.cache_name) in name
        activation_filter = lambda name: cfg.cache_name in name
        fwd_hook = partial(forward_cache_hook, len_prompt=tokens.shape[1], cfg=cfg)
        model.add_hook(activation_filter, fwd_hook, "fwd")

    else:
        fwd_hook = [
            (
                block_modules[layer],
                cache_activation_post_hook_pos(
                    cache, layer, len_prompt=len_prompt, pos=-1
                ),
            )
            for layer in range(n_layers)
        ]

    # (4) Run forward pass
    if do_sample:
        if cfg.framework == "transformer_lens":

            output = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.1,
                freq_penalty=1.0,
                stop_at_eos=False,
                prepend_bos=model.cfg.default_prepend_bos,
            )
        else:
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.1,
                repetition_penalty=1.0,
            )
            with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hook):

                output = model.model.generate(
                    input_ids=tokens,
                    attention_mask=masks,
                    max_new_tokens=max_new_tokens,
                    generation_config=generation_config,
                )

    else:
        if cfg.framework == "transformer_lens":
            output = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        else:
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hook):

                output = model.model.generate(
                    input_ids=tokens,
                    attention_mask=masks,
                    max_new_tokens=max_new_tokens,
                    generation_config=generation_config,
                )

    # (5) Output
    if cfg.framework == "transformer_lens":
        #  Always reset hook after you are done! -- good habit!
        model.reset_hooks()
        return output, ActivationCache(cache, model)
    else:
        return output, cache


def run_and_cache(cfg, model, tokens, masks=None) -> Tuple[
    Float[Tensor, "batch pos d_vocab"],
    Union[Float[Tensor, "layer batch pos d_modle"], ActivationCache],
]:

    if cfg.framework == "transformer_lens":
        model.reset_hooks()
        n_layers = model.cfg.n_layers
        cache = {}

        def steering(
            activations,
            hook,
            cfg=None,
        ):
            # print(steering_vector.shape) # [batch, n_tokens, n_head, d_head ]
            print("hook.layer()")
            print(hook.layer())
            if cfg.cache_pos == "prompt_last_token":
                cache[hook.name] = activations[:, -1, :].detach()
            elif cfg.cache_pos == "all":
                cache[hook.name] = activations.detach()

        steering_hook = partial(
            steering,
            cfg=cfg,
        )
        # (2) Run forward pass
        # logits, cache = model.run_with_cache(tokens, names_filter=activation_filter)
        if type(cfg.cache_name) == list:

            hook = [
                (
                    utils.get_act_name(cache_name, layer),
                    steering_hook,
                )
                for cache_name in cfg.cache_name
                for layer in range(n_layers)
            ]
        else:
            hook = [
                (
                    utils.get_act_name(cfg.cache_name, layer),
                    steering_hook,
                )
                for layer in range(n_layers)
            ]

        with model.hooks(fwd_hooks=hook):
            logits = model(tokens)
            # model(tokens)

        # Always reset hook after you are done! -- good habit!
        model.reset_hooks()

        # logits = torch.zeros(
        #     (len(tokens), model.cfg.d_model),
        # )
        # return logits, ActivationCache(cache, model)

        return logits[:, -1, :], ActivationCache(cache, model)
        # todo: correct back, test memory for now

    else:
        n_layers = model.model.config.num_hidden_layers
        n_samples = tokens.shape[0]
        d_model = model.model.config.hidden_size
        seq_len = tokens.shape[1]
        if cfg.cache_pos == "prompt_last_token":
            cache = torch.zeros(
                (n_layers, n_samples, 1, d_model),
                device=model.device,
                dtype=torch.bfloat16,
            )
        elif cfg.cache_ppos == "all":
            cache = torch.zeros(
                (n_layers, n_samples, seq_len, d_model),
                device=model.device,
                dtype=torch.bfloat16,
            )
        # (1) get hook function
        if "resid" in cfg.cache_name:
            block_modules = model.model_block_modules
        elif "attn" in cfg.cache_name:
            block_modules = model.model_attn_out
        elif "mlp" in cfg.cache_name:
            block_modules = model.model_mlp_out

        fwd_hook_post = [
            (
                block_modules[layer],
                cache_activation_hook_post(cfg, cache=cache, layer=layer),
            )
            for layer in range(n_layers)
        ]
        fwd_hook_pre = []

        # (2) run with hook

        with add_hooks(
            module_forward_pre_hooks=fwd_hook_pre, module_forward_hooks=fwd_hook_post
        ):
            logit = model.model(input_ids=tokens, attention_mask=masks)

        return logit[0][:, -1, :], cache
