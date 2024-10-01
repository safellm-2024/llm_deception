import os
import pickle
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_cos_sim_and_logit_diff_layer(cfg):
    # 1. Load logit diff (resid)
    patch_path = os.path.join(
        cfg.artifact_path(), "activation_patching", "block", "transformer_lens"
    )
    with open(
        patch_path + os.sep + f"{cfg.model_alias}_resid_post_activation_patching.pkl",
        "rb",
    ) as f:
        patching_data = pickle.load(f)
    logit_score = patching_data["act_patch"][:, -1]  # last layer only
    logit_score = logit_score.float().detach().cpu().numpy()
    # 2. Load cosine similarity
    stage_path = os.path.join(cfg.artifact_path(), "stage_stats")
    with open(stage_path + os.sep + f"{cfg.model_alias}_stage_stats.pkl", "rb") as f:
        cos_data = pickle.load(f)
    cos = cos_data["stage_3"]["cos_positive_negative"]
    cos_pca = cos_data["stage_3"]["cos_positive_negative_pca"]

    # 3. plot
    n_layers = len(cos)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_layers),
            y=logit_score,
            name="Logit Difference (Normalized)",
            mode="lines+markers",
            marker=dict(color="dodgerblue"),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_layers),
            y=cos,
            name="Cosine Similarity",
            mode="lines+markers",
            marker=dict(color="gold"),
            yaxis="y3",
        )
    )

    fig.update_xaxes(title_text="Layer")
    # fig.update_yaxes(title_text="Logit Difference", secondary_y=False)
    # fig.update_yaxes(title_text="Cosine Similarity", secondary_y=True)
    fig.update_layout(
        height=500,
        width=800,
        yaxis=dict(
            title="Logit Difference",
            titlefont=dict(color="dodgerblue"),
            tickfont=dict(color="dodgerblue"),
        ),
        yaxis3=dict(
            title="Cosine Similarity",
            titlefont=dict(color="gold"),
            tickfont=dict(color="gold"),
            anchor="x",
            overlaying="y",
            side="right",
        ),
        font=dict(family="sans-serif", size=20, color="black"),
    )
    fig.show()
    fig.write_html(patch_path + os.sep + "residual_patch_cos_sim_layer" + ".html")
    pio.write_image(
        fig,
        patch_path + os.sep + "residual_patch_cos_sim_layer" + ".png",
        scale=6,
    )
    pio.write_image(
        fig,
        patch_path + os.sep + "residual_patch_cos_sim_layer" + ".pdf",
    )
