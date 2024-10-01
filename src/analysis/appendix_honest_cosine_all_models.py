import os
import argparse

import pickle
import plotly.io as pio

import plotly.express as px
import pandas as pd
import json
from sklearn.metrics import confusion_matrix

import csv
import math
from tqdm import tqdm

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import einops
from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor
import plotly.express as px

import os
import argparse

import pickle
import plotly.io as pio

import plotly.express as px
import pandas as pd

import csv
import math
from tqdm import tqdm

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import einops
from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor
import plotly.express as px
from src.submodules.evaluations.evaluate_deception import (
    plot_performance_confusion_matrix,
)
from pipeline.configs.config_generation import Config

model_path = [
    "Qwen/Qwen-1_8B-Chat",
    "Qwen/Qwen-14B-Chat",
    # "Qwen/Qwen-72B-Chat",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2-57B-A14B-Instruct",
    "01-ai/Yi-6B-Chat",
    "01-ai/Yi-34B-Chat",
    "01-ai/Yi-1.5-6B-Chat",
    "01-ai/Yi-1.5-9B-Chat",
    "01-ai/Yi-1.5-34B-Chat",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "Meta-Llama-3-8B-Instruct",
    "Meta-Llama-3-70B-Instruct",
    "google/gemma-2b-it",
    "google/gemma-7b-it",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
]


models = [
    "Qwen-1.8b",
    "Qwen-14b",
    # "Qwen-72b",
    "Qwen-2-1.5b",
    "Qwen-2-7b",
    "Qwen-2-57b",
    "Yi-1-6b",
    "Yi-1-34b",
    "Yi-1.5-6b",
    "Yi-1.5-9b",
    "Yi-1.5-34b",
    "llama-2-7b",
    "llama-2-13b",
    "llama-2-70b",
    "llama-3-8b",
    "llama-3-70b",
    "gemma-1-2b",
    "gemma-1-7b",
    "gemma-2-2b",
    "gemma-2-9b",
    "gemma-2-27b",
]
# size = [1.8, 14, 72, 1.5, 7, 57, 6, 34, 6, 9, 34, 7, 13, 70, 8, 70, 2, 7, 2, 9, 27]

size = [1.8, 14, 1.5, 7, 57, 6, 34, 6, 9, 34, 7, 13, 70, 8, 70, 2, 7, 2, 9, 27]
size_plot = [ss * 4 for ss in size]
data_category = "facts"
Role = ["Honest", "Lying"]
metric = ["lying_score", "cosine_sim"] * len(model_path)
save_path = os.path.join("D:", "\Data", "deception", "figures")
if not os.path.exists(os.path.join("D:\Data\deception", "figures")):
    os.makedirs(save_path)

lying_scores = []
cosine_sims = []

n_col = 4
n_row = math.ceil(len(models) / n_col)


fig = make_subplots(rows=n_row, cols=n_col, subplot_titles=[name for name in (models)])
mm = 0

line_width = 3


for row in range(n_row):
    for col in range(n_col):
        print(mm)
        model_alias = os.path.basename(model_path[mm])
        artifact_path = os.path.join(
            "D:\Data\deception", "runs", "activation_pca", model_alias
        )

        # cosine similarity
        filename = (
            artifact_path
            + os.sep
            + "stage_stats"
            + os.sep
            + model_alias
            + "_stage_stats.pkl"
        )
        with open(filename, "rb") as file:
            data_cos = pickle.load(file)

        # cosine similarity of last layer
        cosine_sim = data_cos["stage_3"]["cos_positive_negative"]

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(cosine_sim)),
                y=cosine_sim,
                marker=dict(size=7),
                mode="lines+markers",
                showlegend=False,
                line=dict(color="royalblue", width=line_width),
            ),
            row=row + 1,
            col=col + 1,
        )

        fig.update_yaxes(
            title="Layer",
            tickvals=[-1, 0, 1],
            ticktext=[-1, 0, 1],
            row=row + 1,
            col=col + 1,
        )
        fig.update_yaxes(title="cos", row=row + 1, col=col + 1, range=[-1.01, 1.1])

        fig.update_layout(
            font=dict(size=15, color="black"), title=dict(font=dict(size=20))
        )
        mm = mm + 1


fig.update_layout(height=300 * 5, width=300 * 4)
fig.update_layout(
    font=dict(size=20, color="black"),
    title=dict(font=dict(size=20)),
)

fig.show()


pio.write_image(fig, save_path + os.sep + "apdx_lying_cosine_all_models.pdf")
pio.write_image(fig, save_path + os.sep + "apdx_lying_cosine_all_models.png", scale=6)
