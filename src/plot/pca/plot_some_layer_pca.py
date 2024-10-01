import os
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.io as pio
from sklearn.decomposition import PCA
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor


def plot_contrastive_activation_pca_layer(
    activations_positive,
    activations_negative,
    contrastive_type,
    input_label=None,
    input_type=["true", "false"],
    layers=[0, 1],
):

    activations_all: Float[Tensor, "n_layers n_samples  d_model"] = torch.cat(
        (activations_positive, activations_negative), dim=1
    )

    n_contrastive_data = activations_negative.shape[1]
    n_layers = len(layers)
    input_label_all = input_label + input_label
    # true or false label
    input_label_tf = []
    for ll in input_label:
        if ll == 0:
            input_label_tf.append(input_type[1])
        elif ll == 1:
            input_label_tf.append(input_type[0])

    symbol_sequence = []
    for data in input_label_all:
        if data == 1:
            symbol_sequence.append("star")
        else:
            symbol_sequence.append("circle")

    label_text = []
    for ii in range(n_contrastive_data):
        label_text = np.append(
            label_text, f"{contrastive_type[0]}_{input_label_tf[ii]}_{ii}"
        )
    for ii in range(n_contrastive_data):
        label_text = np.append(
            label_text, f"{contrastive_type[1]}_{input_label_tf[ii]}_{ii}"
        )

    n_col = 4
    n_row = math.ceil(n_layers / n_col)
    cols = list(np.arange(n_col)) * n_row
    layers = np.array(layers)
    layers_grid = np.reshape(layers, (n_row, n_col))
    fig = make_subplots(
        rows=n_row, cols=n_col, subplot_titles=[f"layer {n}" for n in (layers)]
    )
    pca = PCA(n_components=3)
    for row in range(n_row):
        for cc in range(n_col):
            # print(f'layer{layer}')
            layer = layers_grid[row, cc]

            activations_pca = pca.fit_transform(activations_all[layer, :, :].cpu())
            df = {}
            df["label"] = input_label_all
            df["pca0"] = activations_pca[:, 0]
            df["pca1"] = activations_pca[:, 1]
            df["pca2"] = activations_pca[:, 2]

            df["label_text"] = label_text

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
                        size=15,
                        line=dict(width=1, color="DarkSlateGrey"),
                        color="gold",
                    ),
                    text=df["label_text"][:n_contrastive_data],
                ),
                row=row + 1,
                col=cc + 1,
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
                        size=15,
                        line=dict(width=1, color="DarkSlateGrey"),
                        color="dodgerblue",
                    ),
                    text=df["label_text"][n_contrastive_data:],
                ),
                row=row + 1,
                col=cc + 1,
            )
    # legend
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            # z=[None],
            mode="markers",
            marker=dict(
                symbol="star",
                size=8,
                line=dict(width=1, color="DarkSlateGrey"),
                color="gold",
            ),
            name=f"{contrastive_type[0]}_{input_type[0]}",
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
                size=8,
                line=dict(width=1, color="DarkSlateGrey"),
                color="gold",
            ),
            name=f"{contrastive_type[0]}_{input_type[1]}",
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
                size=8,
                line=dict(width=1, color="DarkSlateGrey"),
                color="dodgerblue",
            ),
            name=f"{contrastive_type[1]}_{input_type[0]}",
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
                symbol="circle",
                size=8,
                line=dict(width=1, color="DarkSlateGrey"),
                color="dodgerblue",
            ),
            name=f"{contrastive_type[1]}_{input_type[1]}",
            marker_color="dodgerblue",
        ),
        row=row + 1,
        col=ll + 1,
    )

    fig.update_layout(height=330 * n_row, width=1000)
    fig.show()
    # fig.write_html("honest_lying_pca.html")

    return fig


def plot_one_layer_3d(
    activations_1,
    activations_2,
    input_label_1=None,
    input_label_2=None,
    contrastive_name=["honest", "lying"],
    probability_1=None,
    probability_2=None,
    color_by="label",
    layer=16,
):

    n_contrastive_data = activations_2.shape[0]

    if color_by == "label":
        if input_label_1 is not None and input_label_2 is None:
            input_label_all = input_label_1 + input_label_1
            input_label_tf = []
            for ll in input_label_1:
                if ll == 0:
                    input_label_tf.append("false")
                elif ll == 1:
                    input_label_tf.append("true")
            for ll in input_label_2:
                if ll == 0:
                    input_label_tf.append("false")
                elif ll == 1:
                    input_label_tf.append("true")
        elif input_label_1 is not None and input_label_2 is not None:
            input_label_all = input_label_1 + input_label_2
            # true or false label
            input_label_tf = []
            for ll in input_label_1:
                if ll == 0:
                    input_label_tf.append("false")
                elif ll == 1:
                    input_label_tf.append("true")
            for ll in input_label_1:
                if ll == 0:
                    input_label_tf.append("false")
                elif ll == 1:
                    input_label_tf.append("true")
    elif color_by == "probability":
        input_label_all = probability_1 + probability_2
        input_label_tf = []
        for ll in input_label_1:
            if ll == 0:
                input_label_tf.append("false")
            elif ll == 1:
                input_label_tf.append("true")
        for ll in input_label_2:
            if ll == 0:
                input_label_tf.append("false")
            elif ll == 1:
                input_label_tf.append("true")

    label_text = []
    for ii in range(n_contrastive_data):
        label_text = np.append(
            label_text,
            f"{contrastive_name[0]}_{input_label_tf[ii]}_{ii}_{probability_1[ii]}",
        )
    for ii in range(n_contrastive_data):
        label_text = np.append(
            label_text,
            f"{contrastive_name[1]}_{input_label_tf[ii + n_contrastive_data]}_{ii}_{probability_2[ii]}",
        )

    activations_all: Float[Tensor, "n_samples n_layers d_model"] = torch.cat(
        (activations_1, activations_2), dim=0
    )

    pca = PCA(n_components=3)
    activations_pca_all = pca.fit_transform(activations_all[:, layer, :].cpu())
    fig = make_subplots(
        rows=1, cols=1, subplot_titles=[f"layer {layer}"], specs=[[{"type": "scene"}]]
    )
    df = {}
    df["label"] = input_label_all
    df["pca0"] = activations_pca_all[:, 0]
    df["pca1"] = activations_pca_all[:, 1]
    df["pca2"] = activations_pca_all[:, 2]
    df["label_text"] = label_text
    fig.add_trace(
        go.Scatter3d(
            x=df["pca0"][:n_contrastive_data],
            y=df["pca1"][:n_contrastive_data],
            z=df["pca2"][:n_contrastive_data],
            mode="markers",
            showlegend=False,
            marker=dict(
                symbol="cross",
                size=10,
                line=dict(width=1, color="DarkSlateGrey"),
                color=df["label"][:n_contrastive_data],
                colorscale="Viridis",
                colorbar=dict(title="probability"),
            ),
            text=df["label_text"][:n_contrastive_data],
        ),
        row=1,
        col=1,
    )

    fig.update_layout(height=500, width=500)
    fig.write_html(f"pca_layer_3d_layer_{layer}.html")
    return fig


def plot_one_layer_with_centroid_and_vector(
    activations_pca_all,
    centroid_honest_true,
    centroid_honest_false,
    centroid_lying_true,
    centroid_lying_false,
    centroid_vector_honest,
    centroid_vector_lying,
    input_label,
    save_path,
    contrastive_type=["honest", "lying"],
    layer=16,
):

    n_samples = len(input_label)
    if input_label is not None:
        input_label_all = input_label + input_label
        # true or false label
        input_label_tf = []
        for ll in input_label:
            if ll == 0:
                input_label_tf.append("false")
            elif ll == 1:
                input_label_tf.append("true")

    label_text = []
    for ii in range(n_samples):
        label_text = np.append(
            label_text, f"{contrastive_type[0]}_{input_label_tf[ii]}_{ii}"
        )
    for ii in range(n_samples):
        label_text = np.append(
            label_text, f"{contrastive_type[1]}_{input_label_tf[ii]}_{ii}"
        )

    df = {}
    df["label"] = input_label_all
    df["pca0"] = activations_pca_all[layer, :, 0]
    df["pca1"] = activations_pca_all[layer, :, 1]
    df["pca2"] = activations_pca_all[layer, :, 2]
    df["label_text"] = label_text

    # plot the centeroid vector
    # x y of quiver is the origin of the vector, u v is the end point of the vector
    fig1 = ff.create_quiver(
        x=[centroid_honest_false[layer, 0], centroid_lying_false[layer, 0]],
        y=[centroid_honest_false[layer, 1], centroid_lying_false[layer, 1]],
        u=[centroid_vector_honest[layer, 0], centroid_vector_lying[layer, 0]],
        v=[centroid_vector_honest[layer, 1], centroid_vector_lying[layer, 1]],
        line=dict(width=3, color="black"),
        scale=1,
    )
    fig1.add_trace(
        go.Scatter(
            x=df["pca0"][:n_samples],
            y=df["pca1"][:n_samples],
            mode="markers",
            showlegend=False,
            marker=dict(
                symbol="star",
                size=8,
                line=dict(width=1, color="DarkSlateGrey"),
                color="gold",
                opacity=0.6,
            ),
            text=df["label_text"][:n_samples],
        ),
    )
    fig1.add_trace(
        go.Scatter(
            x=df["pca0"][n_samples : n_samples * 2],
            y=df["pca1"][n_samples : n_samples * 2],
            mode="markers",
            showlegend=False,
            marker=dict(
                symbol="circle",
                size=5,
                line=dict(width=1, color="DarkSlateGrey"),
                color="gold",
                opacity=0.6,
            ),
            text=df["label_text"][n_samples : n_samples * 2],
        ),
    )
    # plot centroid: centroid honest, true
    fig1.add_trace(
        go.Scatter(
            x=[centroid_honest_true[layer, 0]],
            y=[centroid_honest_true[layer, 1]],
            marker=dict(symbol="star", size=10, color="orange"),
            name="hoenst_true_centeroid",
        )
    )

    # plot centroid: centroid honest, false
    fig1.add_trace(
        go.Scatter(
            x=[centroid_honest_false[layer, 0]],
            y=[centroid_honest_false[layer, 1]],
            marker=dict(symbol="star", size=10, color="blue"),
            name="hoenst_false_centeroid",
        )
    )
    # plot centroid: centroid lying, true
    fig1.add_trace(
        go.Scatter(
            x=[centroid_lying_true[layer, 0]],
            y=[centroid_lying_true[layer, 1]],
            marker=dict(symbol="circle", size=10, color="orange"),
            name="lying_true_centeroid",
        )
    )
    # plot centroid: centroid lying, false
    fig1.add_trace(
        go.Scatter(
            x=[centroid_lying_false[layer, 0]],
            y=[centroid_lying_false[layer, 1]],
            marker=dict(symbol="circle", size=10, color="blue"),
            name="lying_false_centeroid",
        )
    )
    fig1.show()

    fig1.write_html(save_path + os.sep + f"pca_centroid_layer_{layer}.html")
    pio.write_image(
        fig1, save_path + os.sep + f"pca_centroid_layer_{layer}.png", scale=6
    )
    pio.write_image(
        fig1, save_path + os.sep + f"pca_centroid_layer_{layer}.pdf", scale=6
    )

    return fig1


def plot_layers_with_centroid_and_vector(
    activations_positive,
    activations_negative,
    stage_stats,
    contrastive_type=["honest", "lying"],
    input_label=None,
    input_type=["true", "false"],
    layers=[0, 1],
):
    centroid = stage_stats["stage_3"]
    activations_all: Float[Tensor, "n_layers n_samples  d_model"] = torch.cat(
        (activations_positive, activations_negative), dim=1
    )

    n_contrastive_data = activations_negative.shape[1]
    n_layers = len(layers)
    input_label_all = input_label + input_label
    # true or false label
    input_label_tf = []
    for ll in input_label:
        if ll == 0:
            input_label_tf.append(input_type[1])
        elif ll == 1:
            input_label_tf.append(input_type[0])

    symbols = []
    for data in input_label_all:
        if data == 1:
            symbols.append("star")
        else:
            symbols.append("circle")

    label_text = []
    for ii in range(n_contrastive_data):
        label_text = np.append(
            label_text, f"{contrastive_type[0]}_{input_label_tf[ii]}_{ii}"
        )
    for ii in range(n_contrastive_data):
        label_text = np.append(
            label_text, f"{contrastive_type[1]}_{input_label_tf[ii]}_{ii}"
        )
    if (
        activations_all.dtype == torch.bfloat16
        or activations_all.dtype == torch.float64
        or activations_all.dtype == torch.float32
    ):
        activations_all = activations_all.float()
        activations_all = activations_all.detach().cpu().numpy()
    else:
        activations_all = activations_all.detach().cpu().numpy()

    n_col = 4
    n_row = math.ceil(n_layers / n_col)
    layers = np.array(layers)
    layers_grid = np.reshape(layers, (n_row, n_col))
    fig = make_subplots(
        rows=n_row, cols=n_col, subplot_titles=[f"layer {n}" for n in (layers)]
    )
    pca = PCA(n_components=3)
    ii = 0
    for row in range(n_row):
        for cc in range(n_col):
            layer = layers_grid[row, cc]

            activations_pca = pca.fit_transform(activations_all[layer, :, :])
            df = {}
            df["label"] = input_label_all
            df["pca0"] = activations_pca[:, 0]
            df["pca1"] = activations_pca[:, 1]
            df["pca2"] = activations_pca[:, 2]
            df["label_text"] = label_text
            df["symbols"] = symbols

            if ii == 0:
                plot_legend = False
            else:
                plot_legend = False

            # plot scatter
            plot_scatter(
                df,
                centroid,
                layer,
                n_contrastive_data,
                fig,
                row=row + 1,
                col=cc + 1,
                plot_legend=plot_legend,
            )

            # plot vector
            fig_vector = plot_vector(centroid, layer, showlegend=plot_legend)
            fig.add_trace(
                fig_vector.data[0],
                row=row + 1,
                col=cc + 1,
            )
            fig.layout.update(fig_vector.layout)

            ii = ii + 1

    fig.update_layout(height=330 * n_row, width=330 * 4)
    fig.show()
    return fig
    # fig.write_html("honest_lying_pca.html")


def plot_vector(centroid, layer, showlegend):

    centroid_honest_false = centroid["centroid_positive_false_pca_all"]
    centroid_lying_false = centroid["centroid_negative_false_pca_all"]
    centroid_vector_honest = centroid["centroid_positive_vector_pca_all"]
    centroid_vector_lying = centroid["centroid_negative_vector_pca_all"]

    fig1 = ff.create_quiver(
        x=[centroid_honest_false[layer, 0], centroid_lying_false[layer, 0]],
        y=[centroid_honest_false[layer, 1], centroid_lying_false[layer, 1]],
        u=[centroid_vector_honest[layer, 0], centroid_vector_lying[layer, 0]],
        v=[centroid_vector_honest[layer, 1], centroid_vector_lying[layer, 1]],
        line=dict(width=3, color="black"),
        scale=1,
        showlegend=showlegend,
    )

    return fig1


def plot_scatter(df, centroid, layer, n_samples, fig1, row, col, plot_legend=False):

    centroid_honest_false = centroid["centroid_positive_false_pca_all"]
    centroid_honest_true = centroid["centroid_positive_true_pca_all"]
    centroid_lying_false = centroid["centroid_negative_false_pca_all"]
    centroid_lying_true = centroid["centroid_negative_true_pca_all"]

    fig1.add_trace(
        go.Scatter(
            x=df["pca0"][:n_samples],
            y=df["pca1"][:n_samples],
            mode="markers",
            showlegend=False,
            marker=dict(
                symbol=df["symbols"],
                size=15,
                line=dict(width=1, color="DarkSlateGrey"),
                color="gold",
                opacity=1,
            ),
            text=df["label_text"][:n_samples],
        ),
        row=row,
        col=col,
    )
    fig1.add_trace(
        go.Scatter(
            x=df["pca0"][n_samples : n_samples * 2],
            y=df["pca1"][n_samples : n_samples * 2],
            mode="markers",
            showlegend=False,
            marker=dict(
                symbol=df["symbols"],
                size=15,
                line=dict(width=1, color="DarkSlateGrey"),
                color="dodgerblue",
                opacity=1,
            ),
            text=df["label_text"][n_samples : n_samples * 2],
        ),
        row=row,
        col=col,
    )
    #############################################################

    # plot centroid: centroid honest, true
    fig1.add_trace(
        go.Scatter(
            x=[centroid_honest_true[layer, 0]],
            y=[centroid_honest_true[layer, 1]],
            showlegend=plot_legend,
            marker=dict(
                symbol="star",
                size=18,
                color="orange",
                line=dict(width=1, color="DarkSlateGrey"),
                opacity=1,
            ),
            name="hoenst_true_centeroid",
        ),
        row=row,
        col=col,
    )

    # plot centroid: centroid honest, false
    fig1.add_trace(
        go.Scatter(
            x=[centroid_honest_false[layer, 0]],
            y=[centroid_honest_false[layer, 1]],
            showlegend=plot_legend,
            marker=dict(
                symbol="circle",
                size=18,
                color="orange",
                line=dict(width=1, color="DarkSlateGrey"),
                opacity=1,
            ),
            name="hoenst_false_centeroid",
        ),
        row=row,
        col=col,
    )
    # plot centroid: centroid lying, true
    fig1.add_trace(
        go.Scatter(
            x=[centroid_lying_true[layer, 0]],
            y=[centroid_lying_true[layer, 1]],
            showlegend=plot_legend,
            marker=dict(
                symbol="star",
                size=18,
                color="dodgerblue",
                line=dict(width=1, color="DarkSlateGrey"),
                opacity=1,
            ),
            name="lying_true_centeroid",
        ),
        row=row,
        col=col,
    )
    # plot centroid: centroid lying, false
    fig1.add_trace(
        go.Scatter(
            x=[centroid_lying_false[layer, 0]],
            y=[centroid_lying_false[layer, 1]],
            showlegend=plot_legend,
            marker=dict(
                symbol="circle",
                size=18,
                color="dodgerblue",
                line=dict(width=1, color="DarkSlateGrey"),
                opacity=1,
            ),
            name="lying_false_centeroid",
        ),
        row=row,
        col=col,
    )
    # fig1.show()
    return fig1


def plot_contrastive_activation_pca_one_layer_jailbreaks(
    cfg,
    activations_all,
    contrastive_types_all,
    contrastive_type,
    input_types_all,
    prompt_type,
    layer_plot=10,
):
    print("plot")
    n_layers = activations_all.shape[1]
    # layers = np.arange(n_layers)
    n_contrastive_data = cfg.n_train
    n_contrastive_groups = int(len(activations_all) / n_contrastive_data / 2)
    colors = ["yellow", "red", "blue"]

    label_text = []
    for ii in range(len(activations_all)):
        if int(input_types_all[ii]) == 0:
            label_text = np.append(
                label_text, f"{contrastive_types_all[ii]}_{prompt_type[0]}"
            )
        if int(input_types_all[ii]) == 1:
            label_text = np.append(
                label_text, f"{contrastive_types_all[ii]}_{prompt_type[1]}"
            )

    cols = 4
    # rows = math.ceil(n_layers/cols)
    fig = make_subplots(rows=1, cols=1, subplot_titles=[f"layer {layer_plot}"])

    pca = PCA(n_components=3)

    # print(f'layer{layer}')
    activations_pca = pca.fit_transform(activations_all[:, layer_plot, :].cpu())
    df = {}
    df["label"] = contrastive_types_all
    df["pca0"] = activations_pca[:, 0]
    df["pca1"] = activations_pca[:, 1]
    df["pca2"] = activations_pca[:, 2]
    df["label_text"] = label_text

    for ii in range(n_contrastive_groups):
        fig.add_trace(
            go.Scatter(
                x=df["pca0"][
                    ii * n_contrastive_data * 2 : ii * n_contrastive_data * 2
                    + n_contrastive_data
                ],
                y=df["pca1"][
                    ii * n_contrastive_data * 2 : ii * n_contrastive_data * 2
                    + n_contrastive_data
                ],
                # z=df['pca2'][:n_contrastive_data],
                mode="markers",
                showlegend=False,
                marker=dict(
                    symbol="star",
                    size=20,
                    line=dict(width=1, color="DarkSlateGrey"),
                    color=colors[ii],
                ),
                text=df["label_text"][
                    ii * n_contrastive_data * 2 : ii * n_contrastive_data * 2
                    + n_contrastive_data
                ],
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df["pca0"][
                    ii * n_contrastive_data * 2
                    + n_contrastive_data : (ii + 1) * n_contrastive_data * 2
                ],
                y=df["pca1"][
                    ii * n_contrastive_data * 2
                    + n_contrastive_data : (ii + 1) * n_contrastive_data * 2
                ],
                # z=df['pca2'][:n_contrastive_data],
                mode="markers",
                showlegend=False,
                marker=dict(
                    symbol="circle",
                    size=8,
                    line=dict(width=1, color="DarkSlateGrey"),
                    color=colors[ii],
                ),
                text=df["label_text"][
                    ii * n_contrastive_data * 2
                    + n_contrastive_data : (ii + 1) * n_contrastive_data * 2
                ],
            ),
            row=1,
            col=1,
        )

    # legend
    for ii in range(n_contrastive_groups):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                # z=[None],
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=5,
                    line=dict(width=1, color="DarkSlateGrey"),
                    color=colors[0],
                ),
                name=f"{label_text[ii*n_contrastive_data*2]}",
                marker_color=colors[ii],
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        height=600,
        width=600,
        title=dict(
            text=f"Layer {layer_plot}",
            font=dict(size=30),
            automargin=True,
            yref="paper",
        ),
    )
    fig.show()

    return fig
