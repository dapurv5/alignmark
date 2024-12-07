import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pub_ready_plots as prp
import ternary
from fire import Fire
from matplotlib.patches import ConnectionPatch


def extract_model_name(filepath):
    filename = str(filepath)
    if "Meta-Llama" in filename:
        return "LLaMA-8B-Inst"
    elif "Mistral" in filename:
        return "Mistral-7B-Inst"
    elif "gemma" in filename:
        return "Gemma-2-9B-Inst"
    elif "Phi-3" in filename:
        return "Phi-3-Mini-Inst"
    elif "Qwen2-7B-Instruct" in filename:
        return "Qwen2-7B-Inst"
    return "Unknown"


def setup_ternary_plot(ax):
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=1.0)
    ax.set_axis_off()
    tax.boundary(linewidth=1.0)
    tax.get_axes().set_facecolor("white")
    return tax


def add_vertex_labels(tax):
    fontsize = 12
    offset = 0.02
    tax.annotate(
        "Safe", (1.0 + offset, -offset, 0.0), fontsize=fontsize, ha="left", va="center"
    )
    tax.annotate(
        "Unsafe",
        (offset, 1.0 + offset, 0.0),
        fontsize=fontsize,
        ha="right",
        va="bottom",
    )
    tax.annotate(
        "Overrefusal",
        (-offset, offset, 1.0 + offset),
        fontsize=fontsize,
        ha="right",
        va="top",
    )
    tax.clear_matplotlib_ticks()


def create_legends(ax, markers, colors):
    model_elements = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="gray",
            label=extract_model_name(model),
            markersize=8,
            linestyle="None",
        )
        for model, marker in markers.items()
    ]

    setting_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            label=setting,
            markersize=8,
            linestyle="None",
        )
        for setting, color in colors.items()
    ]

    leg1 = ax.legend(
        handles=model_elements,
        bbox_to_anchor=(1.0, 1.0),
        loc="upper right",
        borderaxespad=0.0,
        ncol=1,
        frameon=False,
        fontsize=12,
        title_fontsize=12,
        alignment="right",
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=setting_elements,
        bbox_to_anchor=(-0.08, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        ncol=1,
        frameon=False,
        fontsize=12,
        title_fontsize=12,
    )


def add_arrows(ax, model_points, models_to_connect):
    for model in models_to_connect:
        if model not in model_points:
            continue

        start = model_points[model]["Unwatermarked"]
        kgw = model_points[model].get("KGW")
        gumbel = model_points[model].get("Gumbel")

        if not (kgw and gumbel):
            continue

        for end_point, rad in [(kgw, 0.6), (gumbel, -0.3)]:
            distance = (
                (end_point[0] - start[0]) ** 2 + (end_point[1] - start[1]) ** 2
            ) ** 0.5
            offset = max(0.01 * (1 / distance), 0.005)
            adjusted_end = (
                end_point[0] - (end_point[0] - start[0]) * offset,
                end_point[1] - (end_point[1] - start[1]) * offset,
            )

            arrow = ConnectionPatch(
                xyA=start,
                xyB=adjusted_end,
                coordsA="data",
                coordsB="data",
                axesA=ax,
                axesB=ax,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="->",
                color="black",
                linewidth=1,
                zorder=5,
            )
            ax.add_patch(arrow)


def plot(df: pd.DataFrame, markers: dict[str, str]):
    # Normalize data
    metrics = ["Safe", "Unsafe", "Overrefusal"]
    for idx in df.index:
        total = df.loc[idx, metrics].sum()
        df.loc[idx, metrics] = df.loc[idx, metrics] / total

    colors = {"KGW": "#ff7f0e", "Gumbel": "#2ca02c", "Unwatermarked": "#1f77b4"}

    with prp.get_context(layout=prp.Layout.ICML, single_col=True) as (fig, ax):
        tax = setup_ternary_plot(ax)

        # Plot points and collect coordinates
        model_points = {}
        for model in markers:
            model_points[model] = {}
            for setting in colors:
                mask = (df["Model Name"] == model) & (df["Setting"] == setting)
                if not mask.any():
                    continue

                point = df[mask].iloc[0]
                coords = (point["Safe"], point["Unsafe"], point["Overrefusal"])
                scatter_points = tax.scatter(
                    [coords],
                    marker=markers[model],
                    color=colors[setting],
                    s=100,
                    label=f"{model} ({setting})",
                    zorder=10,
                )

                if ax.collections:
                    last_collection = ax.collections[-1]
                    model_points[model][setting] = tuple(
                        last_collection.get_offsets()[0]
                    )

        add_arrows(ax, model_points, ["Qwen2-7B-Instruct", "Phi-3-mini-4k-instruct"])
        add_vertex_labels(tax)
        create_legends(ax, markers, colors)
        plt.tight_layout(rect=[-0.1, 0, 1, 1])
        plt.show()


def filter_data_by_model(df: pd.DataFrame, model_name: str = None) -> pd.DataFrame:
    """Filter the dataframe to keep only the specified model's data.

    Args:
        df: Input dataframe with model data
        model_name: Model name to filter by

    Returns:
        Filtered dataframe
    """

    mask = df["Model Name"].str.contains(model_name, case=False)
    filtered_df = df[mask].copy()

    if len(filtered_df) == 0:
        raise ValueError(f"Model {model_name} not found in data")

    return filtered_df


def main(input_path: str, model_name: str = None):
    df = pd.read_csv(input_path, sep="\t")
    if model_name:
        df = filter_data_by_model(df, model_name)
    # Define markers for different models
    markers = {
        "Qwen2-7B-Instruct": "o",
        "Phi-3-mini-4k-instruct": "s",
        "Meta-Llama-3.1-8B-Instruct": "^",
        "Mistral-7B-Instruct-v0.3": "D",
    }
    # Filter the markers to only include the models in the dataframe
    markers = {k: v for k, v in markers.items() if k in df["Model Name"].unique()}
    plot(df, markers)


if __name__ == "__main__":
    Fire(main)
