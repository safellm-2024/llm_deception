import torch
import os
import math
from tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import einops
from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor
import pickle
import json


def get_pca_and_plot(cfg, activations_positive, activations_negative, dataset):
    activations_all: Float[Tensor, "n_layers n_samples  d_model"] = torch.cat(
        (activations_positive, activations_negative), dim=1
    )

    n_contrastive_data = activations_negative.shape[1]
    n_layers = activations_negative.shape[0]
    contrastive_type = cfg.contrastive_type
    answer_type = cfg.answer_type

    if cfg.task_name == "deception":
        labels_all = dataset.labels_positive_correct + dataset.labels_positive_correct
        answers_all = (
            dataset.answers_positive_correct + dataset.answers_positive_correct
        )
    if cfg.task_name == "jailbreak" or cfg.task_name == "jailbreak_choice":
        labels_all = dataset.input_labels + dataset.input_labels
        answers_all = dataset.input_types + dataset.input_types

    symbol_sequence = []
    for data in labels_all:
        if data == 1:
            symbol_sequence.append("star")
        else:
            symbol_sequence.append("circle")

    cols = 4
    rows = math.ceil(n_layers / cols)
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=[f"layer {n}" for n in range(n_layers)]
    )

    pca = PCA(n_components=3)
    for row in range(rows):
        for ll, layer in enumerate(range(row * 4, row * 4 + 4)):
            # print(f'layer{layer}')
            if layer < n_layers:
                activations_pca = pca.fit_transform(
                    activations_all[layer, :, :].float().detach().cpu().numpy()
                )
                df = {}
                df["label"] = labels_all
                df["pca0"] = activations_pca[:, 0]
                df["pca1"] = activations_pca[:, 1]
                df["pca2"] = activations_pca[:, 2]
                df["answers_all"] = answers_all

                fig.add_trace(
                    go.Scatter(
                        x=df["pca0"][:n_contrastive_data],
                        y=df["pca1"][:n_contrastive_data],
                        # z=df['pca2'][:n_contrastive_data],
                        mode="markers",
                        name=contrastive_type[0],
                        showlegend=False,
                        marker=dict(
                            symbol=symbol_sequence[:n_contrastive_data],
                            size=8,
                            line=dict(width=1, color="darkslategrey"),
                            color="gold",
                        ),
                        text=df["answers_all"][:n_contrastive_data],
                    ),
                    row=row + 1,
                    col=ll + 1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df["pca0"][n_contrastive_data:],
                        y=df["pca1"][n_contrastive_data:],
                        # z=df['pca2'][n_contrastive_data:],
                        mode="markers",
                        name=contrastive_type[1],
                        showlegend=False,
                        marker=dict(
                            symbol=symbol_sequence[n_contrastive_data:],
                            size=8,
                            line=dict(width=1, color="darkslategrey"),
                            color="dodgerblue",
                        ),
                        text=df["answers_all"][n_contrastive_data:],
                    ),
                    row=row + 1,
                    col=ll + 1,
                )
    # legend
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            # z=[None],
            mode="markers",
            marker=dict(
                symbol="circle",
                size=5,
                line=dict(width=1, color="darkslategrey"),
                color=df["label"][n_contrastive_data:],
            ),
            name=f"{contrastive_type[0]}_{answer_type[1]}",
            marker_color="gold",
        ),
        row=row + 1,
        col=ll + 1,
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            # z=[None],
            mode="markers",
            marker=dict(
                symbol="star",
                size=5,
                line=dict(width=1, color="darkslategrey"),
                color=df["label"][n_contrastive_data:],
            ),
            name=f"{contrastive_type[0]}_{answer_type[0]}",
            marker_color="gold",
        ),
        row=row + 1,
        col=ll + 1,
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            # z=[None],
            mode="markers",
            marker=dict(
                symbol="circle",
                size=5,
                line=dict(width=1, color="darkslategrey"),
                color=df["label"][n_contrastive_data:],
            ),
            name=f"{contrastive_type[1]}_{answer_type[1]}",
            marker_color="dodgerblue",
        ),
        row=row + 1,
        col=ll + 1,
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            # z=[None],
            mode="markers",
            marker=dict(
                symbol="star",
                size=5,
                line=dict(width=1, color="darkslategrey"),
                color=df["label"][n_contrastive_data:],
            ),
            name=f"{contrastive_type[1]}_{answer_type[0]}",
            marker_color="dodgerblue",
        ),
        row=row + 1,
        col=ll + 1,
    )
    fig.update_layout(
        height=rows * 250,
        width=cols * 200 + 50,
    )

    fig.update_layout(
        title=dict(
            text=cfg.model_alias,
            font=dict(size=40),
        )
    )
    fig.show()
    # fig.write_html('honest_lying_pca.html')

    return fig
