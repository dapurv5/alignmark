import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pub_ready_plots as prp
import ternary
from fire import Fire
from matplotlib.patches import ConnectionPatch
from scipy.stats import dirichlet

from plots.plot_utils import get_color, get_short_model_name, get_short_watermark_name


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


def create_legends(ax, markers, watermark_types):
    model_elements = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="gray",
            label=get_short_model_name(model),
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
            color=get_color(watermark_type),
            label=get_short_watermark_name(watermark_type),
            markersize=8,
            linestyle="None",
        )
        for watermark_type in watermark_types
    ]

    leg1 = ax.legend(
        handles=model_elements,
        bbox_to_anchor=(1.0, 1.0),
        loc="upper right",
        borderaxespad=0.0,
        ncol=1,
        frameon=False,
        fontsize=16,
        title_fontsize=12,
        alignment="right",
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=sorted(
            setting_elements, key=lambda x: len(x.get_label()), reverse=True
        ),
        bbox_to_anchor=(-0.12, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        ncol=1,
        frameon=False,
        fontsize=18,
        title_fontsize=12,
    )


def plot_dirichlet_samples(tax, point_counts, color, n_samples=20, alpha_scale=256.0):
    """
    Plot Dirichlet samples to show uncertainty region.
    """
    # Convert to numpy array and ensure float type
    counts = np.array(
        [
            float(point_counts["Safe"]),
            float(point_counts["Unsafe"]),
            float(point_counts["Overrefusal"]),
        ]
    )

    # Add small constant to avoid zeros
    alpha = counts * alpha_scale + 1.0

    # Generate samples
    samples = dirichlet.rvs(alpha, size=n_samples)

    # Plot samples as uncertainty region
    tax.scatter(samples, color=color, alpha=0.1, s=10, zorder=5)


def plot(df: pd.DataFrame, markers: dict[str, str]):
    # Keep original counts for Dirichlet
    df_counts = df.copy()

    # Normalize data
    metrics = ["Safe", "Unsafe", "Overrefusal"]
    for idx in df.index:
        total = df.loc[idx, metrics].sum()
        df.loc[idx, metrics] = df.loc[idx, metrics] / total
    watermark_types = df["Setting"].unique()

    with prp.get_context(layout=prp.Layout.ICML, single_col=True) as (fig, ax):
        tax = setup_ternary_plot(ax)

        # Plot points and uncertainty regions
        model_points = {}
        for model in markers:
            model_points[model] = {}

            # First plot uncertainty regions
            for watermark_type in watermark_types:
                mask = (df_counts["Model Name"] == model) & (
                    df_counts["Setting"] == watermark_type
                )
                if not mask.any():
                    continue

                # Get point counts and plot Dirichlet samples
                point_counts = df_counts[mask].iloc[0][metrics]
                plot_dirichlet_samples(tax, point_counts, get_color(watermark_type))

            # Then plot the actual points
            for watermark_type in watermark_types:
                mask = (df["Model Name"] == model) & (df["Setting"] == watermark_type)
                if not mask.any():
                    continue

                point = df[mask].iloc[0]
                coords = (point["Safe"], point["Unsafe"], point["Overrefusal"])
                scatter_points = tax.scatter(
                    [coords],
                    marker=markers[model],
                    color=get_color(watermark_type),
                    s=30,
                    label=f"{model} ({watermark_type})",
                    zorder=10,
                )

                if ax.collections:
                    last_collection = ax.collections[-1]
                    model_points[model][watermark_type] = tuple(
                        last_collection.get_offsets()[0]
                    )

        # Add arrows after plotting points but before legends
        add_arrows(
            ax,
            model_points,
            ["Qwen2-7B-Instruct", "Phi-3-mini-4k-instruct", "Qwen2.5-7B-Instruct"],
        )

        add_vertex_labels(tax)
        create_legends(ax, markers, watermark_types)
        plt.tight_layout(rect=[-0.1, 0, 1, 1])
        plt.show()


def add_arrows(ax, model_points, models_to_connect):
    for model in models_to_connect:
        if model not in model_points:
            continue

        start = model_points[model].get("Unwatermarked")
        if "KGW" in model_points[model] and "Gumbel" in model_points[model]:
            kgw = model_points[model].get("KGW")
            gumbel = model_points[model].get("Gumbel")
        elif (
            "KGW (Distort)" in model_points[model]
            and "Gumbel (Dist-Free)" in model_points[model]
        ):
            kgw = model_points[model].get("KGW (Distort)")
            gumbel = model_points[model].get("Gumbel (Dist-Free)")
        elif (
            "KGW (BoN-2)" in model_points[model]
            and "Gumbel (BoN-2)" in model_points[model]
        ):
            kgw = model_points[model].get("KGW (BoN-2)")
            gumbel = model_points[model].get("Gumbel (BoN-2)")
        elif (
            "KGW (BoN-4)" in model_points[model]
            and "Gumbel (BoN-4)" in model_points[model]
        ):
            kgw = model_points[model].get("KGW (BoN-4)")
            gumbel = model_points[model].get("Gumbel (BoN-4)")

        if not all([start, kgw, gumbel]):
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


def filter_data_by_model(df: pd.DataFrame, model_name: str = None) -> pd.DataFrame:
    mask = df["Model Name"].str.contains(model_name, case=False)
    filtered_df = df[mask].copy()
    if len(filtered_df) == 0:
        raise ValueError(f"Model {model_name} not found in data")
    return filtered_df


def main(input_path: str, model_name: str = None):
    df = pd.read_csv(input_path, sep="\t")
    if model_name:
        df = filter_data_by_model(df, model_name)

    markers = {
        "Qwen2-7B-Instruct": "o",
        "Qwen2.5-7B-Instruct": "o",
        "Phi-3-mini-4k-instruct": "s",
        "Meta-Llama-3.1-8B-Instruct": "^",
        "Llama-3.1-8B-Instruct": "^",
        "Mistral-7B-Instruct-v0.3": "D",
    }
    markers = {k: v for k, v in markers.items() if k in df["Model Name"].unique()}
    plot(df, markers)


if __name__ == "__main__":
    Fire(main)
