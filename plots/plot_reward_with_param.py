import json
import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pub_ready_plots as prp
from fire import Fire


def read_jsonl(file_path: str) -> list[dict]:
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def parse_filename(filename: str) -> dict:
    filename_parts = filename.split("_")
    dataset_name = filename_parts[1]
    model_name = filename_parts[2]
    watermark_type = filename_parts[3]
    params = {}
    idx = 3
    for part in filename_parts[4:]:
        idx += 1
        if part.startswith(("delta", "gamma", "ngram", "seed", "temperature")):
            key, value = part, filename_parts[idx + 1]
            params[key] = float(value) if key != "ngram" else int(value)

    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "watermark_type": watermark_type,
    } | params


def process_files(
    input_dir: str,
    watermark_type_to_plot: str,
    param_name_to_plot: str,
) -> dict[tuple[str, str], list[tuple[float, list[float], list[float]]]]:
    """
    Process all the files in the input directory and return a dictionary
    with (model_name, dataset_name) as keys and a list of tuples containing
    the param_name, watermarked reward scores, and unwatermarked reward scores.

    Args:
        input_dir (str): The directory containing the reward files.
        watermark_type_to_plot (str): The type of watermark to plot.
        param_name_to_plot (str): The name of the parameter which varies.
    Returns:
        dict[tuple[str, str], list[tuple[float, list[float], list[float]]]]: A dictionary
        with (model_name, dataset_name) as keys and a list of tuples containing
        the param_name, watermarked reward scores, and unwatermarked reward scores.
    """
    data: dict[tuple[str, str], list[tuple[float, list[float], list[float]]]] = {}
    for filename in os.listdir(input_dir):
        if filename.endswith("_rewards.jsonl"):
            file_path = os.path.join(input_dir, filename)
            parsed_info = parse_filename(filename)

            if (
                param_name_to_plot in parsed_info
                and parsed_info["watermark_type"] == watermark_type_to_plot
            ):
                param_name = parsed_info[param_name_to_plot]
                dataset_name = parsed_info["dataset_name"]
                model_name = parsed_info["model_name"]

                blobs = read_jsonl(file_path)
                watermarked_sc = [
                    blob["watermarked_text_reward_score"] for blob in blobs
                ]
                unwatermarked_sc = [
                    blob["unwatermarked_text_reward_score"] for blob in blobs
                ]
                key = (model_name, dataset_name)
                if key not in data:
                    data[key] = []
                data[key].append((float(param_name), watermarked_sc, unwatermarked_sc))

    # Sort the lists for each key by param_name
    for key in data:
        data[key] = sorted(data[key], key=lambda x: x[0])

    return data


def get_short_model_name(model_name: str) -> str:
    return {
        "Mistral-7B-Instruct-v0.3": "Mistral-7B-Inst",
        "Meta-Llama-3.1-8B-Instruct": "Meta-Llama-8B-Inst",
    }.get(model_name, model_name)


def plot_reward_diff(
    input_dir: str,
    output_file: str = "rewards_plot.pdf",
    watermark_type_to_plot: str = "openai",  # openai, maryland
    param_name_to_plot: str = "temperature",
):
    data: dict[tuple[str, str], list[tuple[float, list[float], list[float]]]] = (
        process_files(input_dir, watermark_type_to_plot, param_name_to_plot)
    )
    with prp.get_context(layout=prp.Layout.ICML, single_col=True) as (
        fig,
        axs,
    ):
        # plt.rcParams.update({"font.size": 6})
        colors = cycle(
            ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        )  # Colors chosen for clarity and distinction in publication
        for (model_name, dataset_name), values in data.items():
            print(f"Reading data for {model_name} on {dataset_name} and plotting")
            wm_strengths, watermarked_rewards, unwatermarked_rewards = zip(*values)

            color = next(colors)

            # Plot watermarked scores
            wm_means = [np.mean(w) for w in watermarked_rewards]
            axs.plot(
                wm_strengths,
                wm_means,
                label=f"{get_short_model_name(model_name)} (W)",
                marker="o",
                markersize=2,
                markerfacecolor=color,
                markeredgecolor=color,
                color=color,
                alpha=0.7,
                markeredgewidth=1,
                linestyle="-",
            )
            # axs.fill_between(
            #     wm_strengths,
            #     [np.mean(w) - np.std(w) for w in watermarked_rewards],
            #     [np.mean(w) + np.std(w) for w in watermarked_rewards],
            #     color=color,
            #     alpha=0.2,
            # )
            unwm_means = [np.mean(u) for u in unwatermarked_rewards]
            # Plot unwatermarked scores
            axs.plot(
                wm_strengths,
                unwm_means,
                label=f"{get_short_model_name(model_name)} (U)",
                marker="s",
                markersize=2,
                markerfacecolor="none",
                markeredgecolor=color,
                color=color,
                alpha=0.7,
                markeredgewidth=1,
                linestyle="dashed",
            )
            # axs.fill_between(
            #     wm_strengths,
            #     [np.mean(u) - np.std(u) for u in unwatermarked_rewards],
            #     [np.mean(u) + np.std(u) for u in unwatermarked_rewards],
            #     color=color,
            #     alpha=0.1,
            # )

        axs.set_xlabel(f"{param_name_to_plot} →", fontsize=6)
        axs.set_ylabel("Reward Score", fontsize=6, labelpad=3)
        axs.tick_params(axis="both", which="major", labelsize=5)
        axs.set_title("Reward Scores with Temperature", fontsize=6)
        axs.legend(loc="best", fontsize=5)

        axs.set_xlim(left=0.2, right=1)
        axs.grid(True, linestyle="--", alpha=0.7)

        fig.savefig(output_file, format="pdf", bbox_inches="tight")
        print(f"Plot saved as {output_file}")


def main(
    input_dir: str,
    output_file: str = "rewards_plot.pdf",
    watermark_type_to_plot: str = "openai",
    param_name_to_plot: str = "temperature",
):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_reward_diff(
        input_dir,
        output_file,
        watermark_type_to_plot,
        param_name_to_plot,
    )


if __name__ == "__main__":
    Fire(main)
