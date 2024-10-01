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

jailbreak = "evil_confidant"
model_path = [
    # "Qwen/Qwen-1_8B-Chat",
    # "Qwen/Qwen-14B-Chat",
    # "Qwen/Qwen-72B-Chat",
    # "Qwen/Qwen2-1.5B-Instruct",
    # "Qwen/Qwen2-7B-Instruct",
    # "Qwen/Qwen2-57B-A14B-Instruct",
    # "01-ai/Yi-6B-Chat",
    # "01-ai/Yi-34B-Chat",
    "01-ai/Yi-1.5-6B-Chat",
    "01-ai/Yi-1.5-9B-Chat",
    "01-ai/Yi-1.5-34B-Chat",
    # "meta-llama/Llama-2-7b-chat-hf",
    # "meta-llama/Llama-2-13b-chat-hf",
    # "meta-llama/Llama-2-70b-chat-hf",
    # "Meta-Llama-3-8B-Instruct",
    # "Meta-Llama-3-70B-Instruct",
    # "google/gemma-2b-it",
    # "google/gemma-7b-it",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
]


models = [
    # "Qwen-1.8b",
    # "Qwen-14b",
    # "Qwen-72b",
    # "Qwen-2-1.5b",
    # "Qwen-2-7b",
    # "Qwen-2-57b",
    "Yi-1.5-6b",
    "Yi-1.5-9b",
    "Yi-1.5-34b",
    # "llama-3-8b",
    # "llama-3-70b",
    "gemma-2-2b",
    "gemma-2-9b",
    "gemma-2-27b",
]
# size = [1.8, 14, 72, 1.5, 7, 57, 6, 34, 6, 9, 34, 7, 13, 70, 8, 70, 2, 7, 2, 9, 27]

# size = [1.5, 7, 57, 6, 9, 34, 8, 70, 2, 9, 27]

size = [6, 9, 34, 8, 70, 2, 9, 27]
size = [6, 9, 34, 2, 9, 27]

Role = ["Honest", "Lying"]
save_path = os.path.join("D:", "\Data", "jailbreak", "figures")
if not os.path.exists(os.path.join("D:\Data\jailbreak", "figures")):
    os.makedirs(save_path)

refusal_scores_jailbreak = []
refusal_scores_hhh = []

for mm in model_path:
    print(mm)
    model_alias = os.path.basename(mm)
    artifact_path = os.path.join(
        "D:\Data\jailbreak", "runs", "activation_pca", model_alias
    )

    # performance jallbreak
    filename = (
        artifact_path
        + os.sep
        + "completions"
        + os.sep
        + model_alias
        + f"_completions_{jailbreak}.json"
    )
    with open(filename, "r") as file:
        data = json.load(file)
    refusal_score_harmful = data["refusal_score_harmful"]
    refusal_scores_jailbreak.append(refusal_score_harmful)

    # HHH
    filename = (
        artifact_path
        + os.sep
        + "completions"
        + os.sep
        + model_alias
        + f"_completions_HHH.json"
    )
    with open(filename, "r") as file:
        data = json.load(file)
    refusal_score_harmful = data["refusal_score_harmful"]
    refusal_scores_hhh.append(refusal_score_harmful)

    # # cosine similarity
    # filename = (
    #     artifact_path
    #     + os.sep
    #     + "stage_stats"
    #     + os.sep
    #     + model_alias
    #     + "_stage_stats.pkl"
    # )
    # with open(filename, "rb") as file:
    #     data = pickle.load(file)
    # # cosine similarity of last layer
    # cosine_sim = -data["stage_3"]["cos_positive_negative"][-1]
    #
    # cosine_sims.append(cosine_sim)


data_plot = {
    "refusal_scores_hhh": refusal_scores_hhh,
    "refusal_scores_jailbreak": refusal_scores_jailbreak,
    "model_name": models,
    "size": size,
    "color": ["salmon"] * len(models),
}
df = pd.DataFrame(data_plot)


# plot
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=models,
        y=refusal_scores_hhh,
        name="HHH",
        marker_color="#C8B8D3",
    )
)

fig.add_trace(
    go.Bar(
        x=models,
        y=refusal_scores_jailbreak,
        name=f"{jailbreak}",
        marker_color="#A9BBAF",
    )
)

fig.update_layout(
    # title="Plot Title",
    height=600,
    width=800,
    # xaxis_title="Stage 3 progress ",
    yaxis_title="Refusal score",
    # legend_title="Model",
    font=dict(
        family="myriad pro",
        size=20,
        color="black",
    ),
)
fig.show()

# save
fig.write_image(
    fig,
    save_path + os.sep + f"jailbreak_performance_all_models_{jailbreak}.pdf",
    scale=6,
)
pio.write_image(
    fig,
    save_path + os.sep + f"jailbreak_performance_all_models_{jailbreak}.png",
    scale=6,
)
