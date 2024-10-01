import random
import json
import pprint
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union

from gprof2dot import labels
from torch import Tensor
from jaxtyping import Float, Int, Bool
from src.utils.utils import logits_to_ave_logit_diff

import einops
from fancy_einsum import einsum
import gc
import os
from typing_extensions import Literal
from tqdm import tqdm
from functools import partial

import pickle
import numpy as np
from matplotlib.widgets import EllipseSelector
from IPython.display import HTML, Markdown
import torch
from src.submodules.dataset.task_and_dataset_deception import (
    ContrastiveDatasetDeception,
)
from src.submodules.model.load_model import load_model
from src.utils.utils import logits_to_ave_logit_diff
from transformer_lens import utils, HookedTransformer, ActivationCache

from src.utils.plotly_utils_arena import imshow
from src.submodules.activation_pca.activation_pca import get_pca_and_plot
import plotly.io as pio

from transformer_lens.hook_points import HookPoint
from src.utils.hook_utils import add_hooks
from src.utils.hook_utils import patch_residual_hook_pre, patch_activation_hook_post


class ActivationPatching:
    def __init__(self, contrastive_dataset, contrastive_base):

        self.model = contrastive_base.model
        self.cfg = contrastive_base.cfg

        if contrastive_base.cfg.clean_type == "positive":
            self.corrupted_tokens = contrastive_base.prompt_tokens_negative
            self.corrupted_masks = contrastive_base.prompt_masks_negative
            self.clean_tokens = contrastive_base.prompt_tokens_positive
            self.clean_masks = contrastive_base.prompt_masks_positive
            self.clean_cache = contrastive_base.cache_positive
            self.corrupted_cache = contrastive_base.cache_negative
            self.clean_prompts = contrastive_dataset.prompts_positive
            self.corrupted_prompts = contrastive_dataset.prompts_negative
            self.answer_tokens_correct = contrastive_base.answer_tokens_positive_correct
            self.answer_tokens_wrong = contrastive_base.answer_tokens_positive_wrong
            self.logit_diff_corrupted = contrastive_base.logit_diff_corrupted
            self.logit_diff_clean = contrastive_base.logit_diff_clean
        else:
            self.corrupted_tokens = contrastive_dataset.prompt_tokens_positive
            self.corrupted_masks = contrastive_dataset.prompt_masks_positive
            self.clean_tokens = contrastive_base.prompt_tokens_negative
            self.clean_masks = contrastive_base.prompt_masks_negative
            self.clean_cache = contrastive_base.cache_negative
            self.corrupted_cache = contrastive_base.cache_negative
            self.clean_prompts = contrastive_dataset.prompts_postive
            self.corrupted_prompts = contrastive_dataset.prompts_negative
            self.answer_tokens_correct = contrastive_base.answer_tokens_negative_correct
            self.answer_tokens_wrong = contrastive_base.answer_tokens_negative_wrong
            self.logit_diff_corrupted = contrastive_base.logit_diff_corrupted
            self.logit_diff_clean = contrastive_base.logit_diff_clean

    def get_activation_patching_block(self):
        """ "
        # patch component including: resid pre, resid mid, resid post, mlp_out, attn_out
        """

        save_path = os.path.join(
            self.cfg.artifact_path(),
            "activation_patching",
            "block",
            self.cfg.framework,
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        patching_metric = partial(
            logit_diff_metric,
            answer_tokens_correct=self.answer_tokens_correct,
            answer_tokens_wrong=self.answer_tokens_wrong,
            logit_diff_corrupted=self.logit_diff_corrupted,
            logit_diff_clean=self.logit_diff_clean,
        )

        if self.cfg.cache_name == "z":
            act_patch = get_act_patch_attn_head_out_all_pos(
                model=self.model,
                corrupted_tokens=self.corrupted_tokens,
                clean_cache=self.clean_cache,
                patching_metric=patching_metric,
            )
        elif "z" in self.cfg.cache_name and len(self.cfg.cache_name) > 1:
            act_patch_zqp = get_act_patch_attn_head_all_pos_zqp(
                model=self.model,
                corrupted_tokens=self.corrupted_tokens,
                clean_cache=self.clean_cache,
                patching_metric=patching_metric,
            )
            act_patch_kv = get_act_patch_attn_head_all_pos_kv(
                model=self.model,
                corrupted_tokens=self.corrupted_tokens,
                clean_cache=self.clean_cache,
                patching_metric=patching_metric,
            )
        else:
            if self.cfg.cache_pos == "all":
                act_patch = act_patch_block_tl(
                    model=self.model,
                    corrupted_tokens=self.corrupted_tokens,
                    clean_cache=self.clean_cache,
                    patching_metric=patching_metric,
                    cache_name=self.cfg.cache_name,
                )
            elif self.cfg.cache_pos == "prompt_last_token":
                act_patch = act_patch_block_tl_pos(
                    model=self.model,
                    corrupted_tokens=self.corrupted_tokens,
                    clean_cache=self.clean_cache,
                    patching_metric=patching_metric,
                    cache_name=self.cfg.cache_name,
                )

        # Plot
        prompt_clean = self.clean_tokens[0]
        prompt_corrupted = self.corrupted_tokens[0]

        labels_clean = [
            f"{tok} {i}" for i, tok in enumerate(self.model.to_str_tokens(prompt_clean))
        ]
        labels_corrupted = [
            f"{tok} {i}"
            for i, tok in enumerate(self.model.to_str_tokens(prompt_corrupted))
        ]
        if self.cfg.cache_name == "z" and len(self.cfg.cache_name) == 1:
            fig = imshow(
                act_patch,
                title="Logit Difference From Patched Attn Head Output",
                labels={"x": "Head", "y": "Layer"},
                width=600,
                return_fig=True,
            )
        elif "z" in self.cfg.cache_name and len(self.cfg.cache_name) > 1:
            fig_1 = imshow(
                act_patch_zqp,
                facet_col=0,
                facet_labels=["Output", "Value", "Pattern"],
                title="Activation Patching Per Head (All Pos)",
                labels={"x": "Head", "y": "Layer"},
                width=1200,
                return_fig=True,
            )
            fig_2 = imshow(
                act_patch_kv,
                facet_col=0,
                facet_labels=["Query", "Key"],
                title="Activation Patching Per Head (All Pos)",
                labels={"x": "Head", "y": "Layer"},
                width=1200,
                return_fig=True,
            )

        else:
            if self.cfg.cache_pos == "all":
                fig = imshow(
                    act_patch,
                    x=labels_clean,
                    title=f"Logit Difference From Patched {self.cfg.framework}",
                    labels={"x": "Sequence Position", "y": "Layer"},
                    width=600,  # If you remove this argument, the plot will usually fill the available space
                    return_fig=True,
                )
            elif self.cfg.cache_pos == "prompt_last_token":
                fig = imshow(
                    act_patch,
                    x=labels_clean[-2:],
                    title=f"Logit Difference From Patched {self.cfg.framework}",
                    labels={"x": "Sequence Position", "y": "Layer"},
                    width=600,  # If you remove this argument, the plot will usually fill the available space
                    return_fig=True,
                )

        # save fig
        # z q pattern v k
        if "z" in self.cfg.cache_name and len(self.cfg.cache_name) > 1:
            for ff, fig in enumerate((fig_1, fig_2)):
                pio.write_image(
                    fig,
                    save_path
                    + os.sep
                    + f"{self.cfg.model_alias}_kqv_activation_patching_{ff}.png",
                    scale=6,
                )
                pio.write_image(
                    fig,
                    save_path
                    + os.sep
                    + f"{self.cfg.model_alias}_kqv_activation_patching_{ff}.pdf",
                )
                fig.write_html(
                    save_path
                    + os.sep
                    + f"{self.cfg.model_alias}_kqv_activation_patching_{ff}.html"
                )

            # Save results
            n_layers = self.model.cfg.n_layers
            pattern_clean = [
                self.clean_cache[utils.get_act_name("pattern", layer)]
                for layer in range(n_layers)
            ]
            pattern_corrupted = [
                self.corrupted_cache[utils.get_act_name("pattern", layer)]
                for layer in range(n_layers)
            ]
            results = {
                "act_patch_zqp": act_patch_zqp,
                "act_patch_kv": act_patch_kv,
                "tokens_clean": labels_clean,
                "tokens_corrupted": labels_corrupted,
                "pattern_clean": pattern_clean,
                "pattern_corrupted": pattern_corrupted,
            }
            with open(
                save_path
                + os.sep
                + f"{self.cfg.model_alias}_{self.cfg.cache_name}_activation_patching.pkl",
                "wb",
            ) as f:
                pickle.dump(results, f)

        else:
            pio.write_image(
                fig,
                save_path
                + os.sep
                + f"{self.cfg.model_alias}_{self.cfg.cache_name}_activation_patching.png",
                scale=6,
            )
            pio.write_image(
                fig,
                save_path
                + os.sep
                + f"{self.cfg.model_alias}_{self.cfg.cache_name}_activation_patching.pdf",
            )
            fig.write_html(
                save_path
                + os.sep
                + f"{self.cfg.model_alias}_{self.cfg.cache_name}_activation_patching.html"
            )

            # Save results
            results = {
                "act_patch": act_patch,
            }
            with open(
                save_path
                + os.sep
                + f"{self.cfg.model_alias}_{self.cfg.cache_name}_activation_patching.pkl",
                "wb",
            ) as f:
                pickle.dump(results, f)

    def act_patch_block(
        self, patching_metric: Callable[[Float[Tensor, "batch d_vocab"]], float]
    ) -> Float[Tensor, "layer pos"]:
        """
        Returns an array of results of activation_patching each position at each layer in the residual
        stream, using the value from the clean cache.

        The results are calculated using the patching_metric function, which should be
        called on the model's logit output.
        """
        n_layers = self.model.cfg.n_layers
        seq_len = self.corrupted_tokens.size(-1)
        results = torch.zeros(n_layers, seq_len, device=self.model.device)

        for patch_layer in tqdm(range(n_layers)):
            for patch_pos in range(seq_len):
                patched_logit = self.patch_residual_component(
                    patch_layer=patch_layer,
                    patch_pos=patch_pos,
                )
                results[patch_layer, patch_pos] = patching_metric(patched_logit)

        return results

    def patch_residual_component(
        self,
        patch_layer: int,
        patch_pos: int,
    ) -> Float[Tensor, "batch d_model"]:
        """
        Patches a given sequence position in the residual stream, using the value
        from the clean cache.
        """

        # (1) Get forward hook
        if "resid" in self.cfg.cache_name:
            block_modules = self.model.model_block_modules
        elif "attn" in self.cfg.cache_name:
            block_modules = self.model.model_attn_out
        elif "mlp" in self.cfg.cache_name:
            block_modules = self.model.model_mlp_out
        n_layers = self.model.model.config.num_hidden_layers
        if self.cfg.cache_name == "resid_pre":
            fwd_hook_pre = [
                (
                    block_modules[layer],
                    patch_residual_hook_pre(
                        clean_cache=self.clean_cache,
                        patch_layer=patch_layer,
                        layer=layer,
                        patch_pos=patch_pos,
                    ),
                )
                for layer in range(n_layers)
            ]
            fwd_hook_post = []
        else:
            fwd_hook_post = [
                (
                    block_modules[patch_layer],
                    patch_activation_hook_post(
                        clean_activation=self.clean_cache[patch_layer],
                        patch_pos=patch_pos,
                    ),
                )
            ]
            fwd_hook_pre = []

        # (3) run with hook
        with add_hooks(
            module_forward_pre_hooks=fwd_hook_pre, module_forward_hooks=fwd_hook_post
        ):
            patched_logit = self.model.model(
                input_ids=self.corrupted_tokens, attention_mask=self.corrupted_masks
            )

        return patched_logit[0][:, -1, :]


def logit_diff_metric(
    logits: Float[Tensor, "batch d_vocab"],
    answer_tokens_correct: Float[Tensor, "batch"],
    answer_tokens_wrong: Float[Tensor, "batch"],
    logit_diff_corrupted: float,
    logit_diff_clean: float,
) -> Float[Tensor, ""]:
    """
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on corrupted input, and 1 when performance is same as on clean input.
    """
    patched_logit_diff = logits_to_ave_logit_diff(
        logits, answer_tokens_correct, answer_tokens_wrong
    )

    return (patched_logit_diff - logit_diff_corrupted) / (
        logit_diff_clean - logit_diff_corrupted
    )


def act_patch_block_tl(
    model: HookedTransformer,
    corrupted_tokens: Float[Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Callable[[Float[Tensor, "batch d_vocab"]], float],
    cache_name: str = "resid_pre",
) -> Float[Tensor, "layer pos"]:
    """
    Returns an array of results of activation_patching each position at each layer in the residual
    stream, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    """

    def patch_residual_component(
        corrupted_residual_component: Float[Tensor, "batch pos d_model"],
        hook: HookPoint,
        pos: int,
        clean_cache: ActivationCache,
    ) -> Float[Tensor, "batch pos d_model"]:
        """
        Patches a given sequence position in the residual stream, using the value
        from the clean cache.
        """
        corrupted_residual_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
        return corrupted_residual_component

    model.reset_hooks()
    seq_len = corrupted_tokens.size(-1)
    results = torch.zeros(
        model.cfg.n_layers, seq_len, device="cuda", dtype=torch.bfloat16
    )

    for layer in tqdm(range(model.cfg.n_layers)):
        for pos in range(seq_len):
            hook_fun = partial(
                patch_residual_component, pos=pos, clean_cache=clean_cache
            )
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(utils.get_act_name(cache_name, layer), hook_fun)],
            )
            results[layer, pos] = patching_metric(patched_logits)

    return results


def act_patch_block_tl_pos(
    model: HookedTransformer,
    corrupted_tokens: Float[Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Callable[[Float[Tensor, "batch d_vocab"]], float],
    cache_name: str = "resid_pre",
) -> Float[Tensor, "layer pos"]:
    """
    Returns an array of results of activation_patching each position at each layer in the residual
    stream, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    """

    def patch_residual_component(
        corrupted_residual_component: Float[Tensor, "batch pos d_model"],
        hook: HookPoint,
        pos_corrupted: int,
        pos_clean: int,
        clean_cache: ActivationCache,
    ) -> Float[Tensor, "batch pos d_model"]:
        """
        Patches a given sequence position in the residual stream, using the value
        from the clean cache.
        """
        corrupted_residual_component[:, pos_corrupted, :] = clean_cache[hook.name][
            :, pos_clean, :
        ]
        return corrupted_residual_component

    model.reset_hooks()
    seq_len = np.arange(corrupted_tokens.size(-1))
    patch_len = 2
    results = torch.zeros(
        model.cfg.n_layers, patch_len, device="cuda", dtype=torch.bfloat16
    )

    for layer in tqdm(range(model.cfg.n_layers)):
        for ii, pos in enumerate([seq_len[-2], seq_len[-1]]):
            hook_fun = partial(
                patch_residual_component,
                pos_corrupted=pos,
                pos_clean=ii,
                clean_cache=clean_cache,
            )
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(utils.get_act_name(cache_name, layer), hook_fun)],
            )
            results[layer, ii] = patching_metric(patched_logits)

    return results


def patch_head_vector(
    corrupted_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    head_index: int,
    clean_cache: ActivationCache,
) -> Float[Tensor, "batch pos head_index d_head"]:
    """
    Patches the output of a given head (before it's added to the residual stream) at
    every sequence position, using the value from the clean cache.
    """
    # print(hook.name)
    # print(clean_cache[hook.name].shape)

    corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][:, :, head_index]
    return corrupted_head_vector


def get_act_patch_attn_head_out_all_pos(
    model: HookedTransformer,
    corrupted_tokens: Float[Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Callable,
) -> Float[Tensor, "layer head"]:
    """
    Returns an array of results of activation_patching at all positions for each head in each
    layer, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    """
    results = torch.zeros(
        model.cfg.n_layers, model.cfg.n_heads, device="cuda", dtype=torch.float32
    )
    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            hook_fun = partial(
                patch_head_vector, head_index=head, clean_cache=clean_cache
            )
            patched_logits = model.run_with_hooks(
                corrupted_tokens, fwd_hooks=[(utils.get_act_name("z", layer), hook_fun)]
            )

            results[layer, head] = patching_metric(patched_logits)
    return results


def patch_attn_patterns(
    corrupted_head_vector: Float[Tensor, "batch head_index pos_q pos_k"],
    hook: HookPoint,
    head_index: int,
    clean_cache: ActivationCache,
) -> Float[Tensor, "batch pos head_index d_head"]:
    """
    Patches the attn patterns of a given head at every sequence position, using
    the value from the clean cache.
    """
    corrupted_head_vector[:, head_index] = clean_cache[hook.name][:, head_index]
    return corrupted_head_vector


def get_act_patch_attn_head_all_pos_zqp(
    model: HookedTransformer,
    corrupted_tokens: Float[Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Callable,
) -> Float[Tensor, "layer head"]:
    """
    Returns an array of results of activation_patching at all positions for each head in each
    layer (using the value from the clean cache) for output, queries, keys, values
    and attn pattern in turn.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.

    Patch z, q, p(attention pattern)
    """
    results = torch.zeros(
        3, model.cfg.n_layers, model.cfg.n_heads, device="cuda", dtype=torch.float32
    )

    # Loop over each component in turn
    for component_idx, component in enumerate(["z", "q", "pattern"]):
        for layer in tqdm(range(model.cfg.n_layers)):
            for head in range(model.cfg.n_heads):
                # Get different hook function if we're doing attention probs
                hook_fn_general = (
                    patch_attn_patterns if component == "pattern" else patch_head_vector
                )
                hook_fn = partial(
                    hook_fn_general, head_index=head, clean_cache=clean_cache
                )
                # Get patched logits
                patched_logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[(utils.get_act_name(component, layer), hook_fn)],
                    return_type="logits",
                )
                results[component_idx, layer, head] = patching_metric(patched_logits)

    return results


def get_act_patch_attn_head_all_pos_kv(
    model: HookedTransformer,
    corrupted_tokens: Float[Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Callable,
) -> Float[Tensor, "layer head"]:
    """
    Returns an array of results of activation_patching at all positions for each head in each
    layer (using the value from the clean cache) for output, queries, keys, values
    and attn pattern in turn.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    """
    results = torch.zeros(
        2, model.cfg.n_layers, model.cfg.n_heads, device="cuda", dtype=torch.float32
    )
    # Loop over each component in turn
    if "Llama" in model.cfg.model_name or "gemma" in model.cfg.model_name:
        n_key_value_heads = model.cfg.n_key_value_heads
    else:
        n_key_value_heads = model.cfg.n_heads
    for component_idx, component in enumerate(["k", "v"]):
        for layer in tqdm(range(model.cfg.n_layers)):
            for head in range(n_key_value_heads):
                # Get different hook function if we're doing attention probs
                hook_fn_general = (
                    patch_attn_patterns if component == "pattern" else patch_head_vector
                )
                hook_fn = partial(
                    hook_fn_general, head_index=head, clean_cache=clean_cache
                )
                # Get patched logits
                patched_logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[(utils.get_act_name(component, layer), hook_fn)],
                    return_type="logits",
                )
                results[component_idx, layer, head] = patching_metric(patched_logits)

    return results
