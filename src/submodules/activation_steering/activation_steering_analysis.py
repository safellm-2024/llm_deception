import os
import pickle
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_cos_sim_and_acc_layer(cfg):
    # 1. Load acc
    score_path = os.path.join(
        cfg.artifact_path(), "activation_steering", "positive_addition", "performance"
    )

    with open(
        score_path + os.sep + f"{cfg.model_alias}_honest_score_all_layers.pkl",
        "rb",
    ) as f:
        data = pickle.load(f)

    score = data["score_negative_all"][:]  # last layer only

    # 2. Load cosine similarity (baseline)
    stage_path = os.path.join(cfg.artifact_path(), "stage_stats")
    with open(stage_path + os.sep + f"{cfg.model_alias}_stage_stats.pkl", "rb") as f:
        cos_data = pickle.load(f)
    cos = cos_data["stage_3"]["cos_positive_negative"]
    cos_pca = cos_data["stage_3"]["cos_positive_negative_pca"]

    # load cosine similarity (intervention)

    # 3. plot
    n_layers = len(cos)
    line_width = 3
    size = 5
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_layers),
            y=score,
            name="Performance",
            mode="lines+markers",
            marker=dict(color="dodgerblue", size=size),
            line=dict(color="dodgerblue", width=line_width),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_layers),
            y=cos,
            name="Cosine Similarity",
            mode="lines+markers",
            marker=dict(color="gold", size=size),
            yaxis="y3",
            line=dict(color="gold", width=line_width),
        )
    )
    fig.update_xaxes(title_text="Layer")
    # fig.update_yaxes(title_text="Logit Difference", secondary_y=False)
    # fig.update_yaxes(title_text="Cosine Similarity", secondary_y=True)

    fig.update_layout(
        height=300,
        width=500,
        yaxis=dict(
            title="Performance",
            titlefont=dict(color="dodgerblue"),
            tickfont=dict(color="dodgerblue"),
            tickmode="array",
            tickvals=[0, 0.5, 1],
            range=[-0.01, 1.11],
        ),
        yaxis3=dict(
            title="Cosine Similarity",
            titlefont=dict(color="gold"),
            tickfont=dict(color="gold"),
            anchor="x",
            overlaying="y",
            side="right",
            tickmode="array",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            range=[-1.01, 1.11],
        ),
        font=dict(family="verdana", size=16, color="black"),
    )

    fig.show()

    fig.write_html(
        score_path
        + os.sep
        + f"{cfg.cache_name}_steering_cos_sim_performance_layer"
        + ".html"
    )
    pio.write_image(
        fig,
        score_path
        + os.sep
        + f"{cfg.cache_name}_steering_cos_sim_performance_layer"
        + ".png",
        scale=6,
    )
    pio.write_image(
        fig,
        score_path
        + os.sep
        + f"{cfg.cache_name}_steering_cos_sim_performance_layer"
        + ".pdf",
    )
