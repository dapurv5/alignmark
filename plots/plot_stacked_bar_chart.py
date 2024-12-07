import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pub_ready_plots as prp
from fire import Fire


def extract_model_name(filepath):
    filename = str(filepath)
    if "Meta-Llama" in filename:
        return "LLaMA-8B-Inst"
    elif "Mistral" in filename:
        return "Mistral-7B-Inst"
    elif "Phi-3" in filename:
        return "Phi-3-Mini-Inst"
    elif "Qwen2" in filename:
        return "Qwen2-7B-Inst"
    return "Unknown"


def calculate_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate deltas from unwatermarked baseline for each model."""
    metrics = ["Unsafe", "Overrefusal"]
    delta_rows = []

    for model in df["Model Name"].unique():
        model_data = df[df["Model Name"] == model]
        baseline = model_data[model_data["Setting"] == "Unwatermarked"].iloc[0]

        for setting in ["KGW", "Gumbel"]:
            if not model_data[model_data["Setting"] == setting].empty:
                watermarked = model_data[model_data["Setting"] == setting].iloc[0]

                # Calculate deltas
                deltas = {
                    "Model Name": model,
                    "Setting": setting,
                }

                # Calculate absolute differences
                for metric in metrics:
                    deltas[f"Delta_{metric}"] = watermarked[metric] - baseline[metric]

                delta_rows.append(deltas)

    return pd.DataFrame(delta_rows)


def plot(df: pd.DataFrame):
    # Calculate deltas
    delta_df = calculate_deltas(df)

    # Colors for different settings
    colors = {"KGW": "#ff7f0e", "Gumbel": "#2ca02c"}

    with prp.get_context(layout=prp.Layout.ICML, single_col=True) as (fig, ax):
        # Clear the main axis as we'll create our own subplots
        ax.remove()

        # Create two subplots using the existing figure
        gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        # Sort models by average delta unsafe
        model_order = (
            delta_df.groupby("Model Name")["Delta_Unsafe"]
            .mean()
            .sort_values(ascending=True)
            .index
        )

        # Plot Unsafe changes
        x = np.arange(len(model_order))
        width = 0.35

        for i, setting in enumerate(["KGW", "Gumbel"]):
            mask = delta_df["Setting"] == setting
            data = [
                delta_df[mask & (delta_df["Model Name"] == model)]["Delta_Unsafe"].iloc[
                    0
                ]
                for model in model_order
            ]

            ax1.bar(x + i * width, data, width, label=setting, color=colors[setting])

        ax1.set_ylabel("Δ Unsafe Responses")
        ax1.set_title("Change in Unsafe Responses")

        # Plot Overrefusal changes
        for i, setting in enumerate(["KGW", "Gumbel"]):
            mask = delta_df["Setting"] == setting
            data = [
                delta_df[mask & (delta_df["Model Name"] == model)][
                    "Delta_Overrefusal"
                ].iloc[0]
                for model in model_order
            ]

            ax2.bar(x + i * width, data, width, label=setting, color=colors[setting])

        ax2.set_ylabel("Δ Overrefusal Count")
        ax2.set_title("Change in Overrefusal")

        # Customize both subplots
        for ax in [ax1, ax2]:
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(
                [extract_model_name(model) for model in model_order],
                rotation=45,
                ha="right",
            )
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        # Add legend to the first subplot only
        ax1.legend(title="Watermarking", bbox_to_anchor=(1.05, 1), loc="upper left")

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        plt.savefig("watermarking_effects.pdf", bbox_inches="tight", dpi=300)

        # Display the plot
        plt.show()

    # Print the actual delta values
    print("\nDelta values from baseline:")
    pd.set_option("display.float_format", "{:.1f}".format)
    print(delta_df.to_string(index=False))


def main(input_path: str):
    df = pd.read_csv(input_path, sep="\t")
    plot(df)


if __name__ == "__main__":
    Fire(main)
