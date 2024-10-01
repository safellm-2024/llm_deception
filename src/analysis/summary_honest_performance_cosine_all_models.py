import os
import argparse

import pickle
import plotly.io as pio

import plotly.express as px
import pandas as pd
import json
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

for mm in model_path:
    print(mm)
    model_alias = os.path.basename(mm)
    artifact_path = os.path.join(
        "D:\Data\deception", "runs", "activation_pca", model_alias
    )

    # performance
    filename = (
        artifact_path
        + os.sep
        + "completions"
        + os.sep
        + model_alias
        + "_completions_lying.json"
    )
    with open(filename, "r") as file:
        data = json.load(file)
    lying_score = data["lying_score"]

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
        data = pickle.load(file)
    # cosine similarity of last layer
    cosine_sim = -data["stage_3"]["cos_positive_negative"][-1]

    lying_scores.append(lying_score)
    cosine_sims.append(cosine_sim)


data_plot = {
    "lying_scores": lying_scores,
    "cosine_sims": cosine_sims,
    "model_name": models,
    "size": size_plot,
    "color": ["salmon"] * len(models),
}
df = pd.DataFrame(data_plot)


fig = px.scatter(
    x=df["cosine_sims"],
    y=df["lying_scores"],
    text=df["model_name"],
    size=df["size"],
    # color_discrete_sequence=df["color"],
)


# fig.update_layout(height=400, width=500)
fig.update_layout(
    # title="Plot Title",
    height=800,
    width=800,
    xaxis_title="Stage 3 progress ",
    yaxis_title="Lying score",
    # legend_title="Model",
    font=dict(
        family="Myriad Pro",
        size=20,
        color="black",
    ),
    xaxis=dict(
        tickmode="array",  # change 1
        tickvals=[-1, -0.5, 0, 0.5],  # change 2
        ticktext=[0, 0.25, 0.75, 1],  # change 3
    ),
)
fig.show()

pio.write_image(fig, save_path + os.sep + "lying_performance_cos_all_models.pdf")
pio.write_image(
    fig, save_path + os.sep + "lying_performance_cos_all_models.png", scale=6
)
