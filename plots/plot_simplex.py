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


def plot(df: pd.DataFrame, markers: dict[str, str]):
    # Normalize each row to create probability distributions
    metrics = ["Safe", "Unsafe", "Overrefusal"]
    for idx in df.index:
        total = df.loc[idx, metrics].sum()
        df.loc[idx, metrics] = df.loc[idx, metrics] / total

    # Define colors for different settings
    colors = {"KGW": "#ff7f0e", "Gumbel": "#2ca02c", "Unwatermarked": "#1f77b4"}

    with prp.get_context(layout=prp.Layout.ICML, single_col=True) as (fig, ax):
        tax = ternary.TernaryAxesSubplot(ax=ax, scale=1.0)

        # Plot points for each model and setting
        for model in markers:
            for setting in colors:
                mask = (df["Model Name"] == model) & (df["Setting"] == setting)
                if mask.any():
                    point = df[mask].iloc[0]
                    # Get normalized coordinates
                    coords = (point["Safe"], point["Unsafe"], point["Overrefusal"])
                    tax.scatter(
                        [coords],
                        marker=markers[model],
                        color=colors[setting],
                        s=100,
                        label=f"{model} ({setting})",
                        zorder=10,
                    )  # Ensure points are above grid lines

        # Store points for arrows
        model_points = {}
        for model in markers:
            model_points[model] = {}
            for setting in colors:
                mask = (df["Model Name"] == model) & (df["Setting"] == setting)
                if mask.any():
                    point = df[mask].iloc[0]
                    coords = (point["Safe"], point["Unsafe"], point["Overrefusal"])
                    # Plot points and get the collection object directly
                    scatter_points = tax.scatter(
                        [coords],
                        marker=markers[model],
                        color=colors[setting],
                        s=100,
                        label=f"{model} ({setting})",
                        zorder=10,
                    )
                    # Get the last added collection which contains our point
                    collections = ax.collections
                    if collections:
                        last_collection = collections[-1]
                        model_points[model][setting] = (
                            last_collection.get_offsets()[0][0],
                            last_collection.get_offsets()[0][1],
                        )

        # Add curved arrows for Qwen and Phi models
        models_to_connect = ["Qwen2-7B-Instruct", "Phi-3-mini-4k-instruct"]
        for model in models_to_connect:
            if model in model_points:
                start = model_points[model]["Unwatermarked"]
                kgw = model_points[model].get("KGW")
                gumbel = model_points[model].get("Gumbel")

                if kgw and gumbel:
                    # Create curved arrows with different curvatures
                    arrow_params = [
                        (kgw, 0.6),  # More curved for KGW
                        (gumbel, -0.3),  # Opposite curve for Gumbel
                    ]

                    for end_point, rad in arrow_params:
                        # Calculate distance between points
                        distance = (
                            (end_point[0] - start[0]) ** 2
                            + (end_point[1] - start[1]) ** 2
                        ) ** 0.5

                        # Adjust offset based on distance
                        # Smaller distance -> larger offset (up to 0.25)
                        # Larger distance -> smaller offset (down to 0.1)
                        base_offset = 0.01
                        min_offset = 0.005
                        offset = max(base_offset * (1 / distance), min_offset)

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

        # Remove the square boundary by setting the axis off
        ax.set_axis_off()

        # Keep the triangular boundary
        tax.boundary(linewidth=1.0)
        # tax.gridlines(multiple=0.1, color="gray", linewidth=0.5, alpha=0.3)

        # Remove the background by setting it to white
        tax.get_axes().set_facecolor("white")

        # Add vertex labels with larger font and smaller offset
        fontsize = 12
        offset = 0.02  # Reduced from 0.1 to 0.05
        tax.annotate(
            "Safe",
            (1.0 + offset, -offset, 0.0),
            fontsize=fontsize,
            ha="left",  # Changed to left align
            va="center",
        )
        tax.annotate(
            "Unsafe",
            (offset, 1.0 + offset, 0.0),
            fontsize=fontsize,
            ha="right",  # Changed to right align
            va="bottom",
        )
        tax.annotate(
            "Overrefusal",
            (-offset, offset, 1.0 + offset),
            fontsize=fontsize,
            ha="right",  # Changed to right align
            va="top",
        )

        # Remove axis labels since we're using vertex labels
        tax.clear_matplotlib_ticks()

        # # Add title
        # plt.title(
        #     "Normalized Response Rates in Probability Simplex", y=1.05, fontsize=14
        # )

        # Create separate legend elements for models and settings
        model_legend_elements = []
        settings_legend_elements = []

        # Model markers
        for model, marker in markers.items():
            model_name = extract_model_name(model)
            model_legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="gray",
                    label=model_name,
                    markersize=8,
                    linestyle="None",
                )
            )

        # Setting colors
        for setting, color in colors.items():
            settings_legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=color,
                    label=setting,
                    markersize=8,
                    linestyle="None",
                )
            )

        # Add two separate legends
        # First legend (Models)
        leg1 = ax.legend(
            handles=model_legend_elements,
            bbox_to_anchor=(1.0, 1.0),
            loc="upper right",
            borderaxespad=0.0,
            ncol=1,
            frameon=False,
            fontsize=12,
            title_fontsize=12,
            alignment="right",
        )

        # Add the first legend manually to the axis
        ax.add_artist(leg1)

        # Second legend (Settings)
        ax.legend(
            handles=settings_legend_elements,
            bbox_to_anchor=(-0.08, 1.0),
            loc="upper left",
            borderaxespad=0.0,
            ncol=1,
            frameon=False,
            fontsize=12,
            title_fontsize=12,
        )

        # Adjust layout to center the figure
        plt.tight_layout(
            rect=[-0.1, 0, 1, 1]
        )  # Shifts everything left by adjusting the left margin
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
