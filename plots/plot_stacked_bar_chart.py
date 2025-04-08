import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pub_ready_plots as prp
from fire import Fire

from plots.plot_utils import (
    get_color,
    get_pattern,
    get_short_model_name,
    get_short_watermark_name,
)


def calculate_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate deltas from unwatermarked baseline for each model."""
    metrics = ["Unsafe", "Overrefusal"]
    delta_rows = []
    watermark_types = list(set(df["Setting"].unique()) - {"Unwatermarked"})

    for model in df["Model Name"].unique():
        model_data = df[df["Model Name"] == model]
        baseline = model_data[model_data["Setting"] == "Unwatermarked"].iloc[0]

        for watermark_type in watermark_types:
            if not model_data[model_data["Setting"] == watermark_type].empty:
                watermarked = model_data[model_data["Setting"] == watermark_type].iloc[
                    0
                ]

                # Calculate deltas
                deltas = {
                    "Model Name": model,
                    "Setting": watermark_type,
                }

                # Calculate absolute differences
                for metric in metrics:
                    deltas[f"Delta_{metric}"] = watermarked[metric] - baseline[metric]

                delta_rows.append(deltas)

    return pd.DataFrame(delta_rows)


def plot(df: pd.DataFrame, output_path: str):
    # Calculate deltas
    delta_df = calculate_deltas(df)
    watermark_types = list(set(df["Setting"].unique()) - {"Unwatermarked"})

    with prp.get_context(layout=prp.Layout.ICML, single_col=True) as (fig, ax):
        # Clear the main axis as we'll create our own subplots
        ax.remove()

        # Increase figure size
        fig.set_size_inches(20, 6)

        # Create two subplots with more space between them
        gs = fig.add_gridspec(
            1, 2, hspace=0.3, wspace=0.5
        )  # Increased wspace from 0.4 to 0.5
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        # Sort models by average delta unsafe
        model_order = (
            delta_df.groupby("Model Name")["Delta_Unsafe"]
            .mean()
            .sort_values(ascending=True)
            .index
        )

        # Sort models by size (uncomment for scaling plots)
        # model_order = sorted(
        #     model_order,
        # )

        # Plot Unsafe changes
        x = np.arange(len(model_order))
        width = 0.25  # IMPORTANT: 0.35 is the default, use 0.20 for BoN plots

        for i, watermark_type in enumerate(watermark_types):
            mask = delta_df["Setting"] == watermark_type
            data = [
                delta_df[mask & (delta_df["Model Name"] == model)]["Delta_Unsafe"].iloc[
                    0
                ]
                for model in model_order
            ]

            ax1.bar(
                x + i * width,
                data,
                width,
                label=get_short_watermark_name(watermark_type),
                color=get_color(watermark_type),
                edgecolor="black",
                linewidth=0.5,
                alpha=0.7,
                hatch=get_pattern(watermark_type),
            )

        ax1.set_ylabel("Δ Unsafe Responses", fontsize=24)
        ax1.set_title("Change in Unsafe Responses", fontsize=22)

        # Plot Overrefusal changes
        for i, watermark_type in enumerate(watermark_types):
            mask = delta_df["Setting"] == watermark_type
            data = [
                delta_df[mask & (delta_df["Model Name"] == model)][
                    "Delta_Overrefusal"
                ].iloc[0]
                for model in model_order
            ]

            ax2.bar(
                x + i * width,
                data,
                width,
                label=get_short_watermark_name(watermark_type),
                color=get_color(watermark_type),
                edgecolor="black",
                linewidth=0.5,
                alpha=0.7,
                hatch=get_pattern(watermark_type),
            )

        ax2.set_ylabel("Δ Overrefusal Count", fontsize=24)
        ax2.set_title("Change in Overrefusal", fontsize=22)

        # Customize both subplots with improved formatting
        for ax in [ax1, ax2]:
            # IMPORTANT: width/2 is the default, use 3 * width/2 for BoN plots
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(
                [
                    get_short_model_name(model).replace("-Inst", "")
                    for model in model_order
                ],
                ha="center",
                fontsize=18,
            )
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
            ax.grid(True, axis="y", linestyle="--", alpha=0.7)

            # Increase y-axis tick label size
            ax.tick_params(axis="y", labelsize=16)

        # Update legend formatting and position
        ax1.legend(
            bbox_to_anchor=(1.2, 1.15),
            loc="center",
            fontsize=14,
            title_fontsize=15,
            ncol=2,
            bbox_transform=ax1.transAxes,
        )

        # Adjust layout with more space at the top for the legend
        plt.tight_layout(rect=[0, 0, 1, 0.90])

        # Save the figure with higher quality
        plt.savefig(output_path, bbox_inches="tight", dpi=300)

        # Display the plot
        plt.show()

    # Print the actual delta values
    print("\nDelta values from baseline:")
    pd.set_option("display.float_format", "{:.1f}".format)
    print(delta_df.to_string(index=False))


def main(input_path: str, output_path: str):
    df = pd.read_csv(input_path, sep="\t")
    plot(df, output_path)


if __name__ == "__main__":
    Fire(main)
