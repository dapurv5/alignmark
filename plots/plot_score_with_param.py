import json
import os
from collections import defaultdict
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
    seed = filename_parts[4]
    params = {}
    idx = 4
    for part in filename_parts[5:]:
        idx += 1
        if part.startswith(("delta", "gamma", "ngram", "temperature")):
            key, value = part, filename_parts[idx + 1]
            params[key] = float(value) if key != "ngram" else int(value)

    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "watermark_type": watermark_type,
        "seed": seed,
    } | params


def process_files(
    input_dir: str,
    model_name_to_plot: str,
    param_name_to_plot: str,
    score_name: str,
) -> dict[tuple[str, str], dict[str, list[tuple[float, list[float], list[float]]]]]:
    """
    Process all the files in the input directory and return a dictionary
    with (watermark_type, dataset_name) as keys and a list of tuples containing
    the param_name, watermarked reward scores, and unwatermarked reward scores.

    Args:
        input_dir (str): The directory containing the reward files.
        watermark_type_to_plot (str): The type of watermark to plot.
        param_name_to_plot (str): The name of the parameter which varies.
    Returns:
        dict[tuple[str, str], dict[str, list[tuple[float, list[float], list[float]]]]: A dictionary
        with (watermark_type, dataset_name) as keys and a list of tuples containing
        the param_name, watermarked reward scores, and unwatermarked reward scores.
    """
    data: dict[
        tuple[str, str], dict[str, list[tuple[float, list[float], list[float]]]]
    ] = {}
    for filename in os.listdir(input_dir):
        if "truthful_qa" in filename:
            filename_ = filename.replace(
                "truthful_qa", "truthfulqa"
            )  # no _ allowed in dataset name
        if filename.endswith(f"_{score_name}.jsonl"):
            print(f"Processing {filename}")
            file_path = os.path.join(input_dir, filename)
            parsed_info = parse_filename(filename_)
            if (
                param_name_to_plot in parsed_info
                and parsed_info["model_name"] == model_name_to_plot
            ):
                param_value = parsed_info[param_name_to_plot]
                dataset_name = parsed_info["dataset_name"]
                watermark_type = parsed_info["watermark_type"]
                seed = parsed_info["seed"]

                blobs = read_jsonl(file_path)
                watermarked_sc = [
                    blob[f"watermarked_{score_name}_score"] for blob in blobs
                ]
                unwatermarked_sc = [
                    blob[f"unwatermarked_{score_name}_score"] for blob in blobs
                ]
                key = (watermark_type, dataset_name)
                if key not in data:
                    data[key] = {}
                if seed not in data[key]:
                    data[key][seed] = []
                data[key][seed].append((param_value, watermarked_sc, unwatermarked_sc))

    # Sort the lists for each key, seed by param_name
    for key in data:
        for seed in data[key]:
            data[key][seed] = sorted(data[key][seed], key=lambda x: x[0])
    return data


def get_short_model_name(model_name: str) -> str:
    return {
        "Mistral-7B-Instruct-v0.3": "Mistral-7B-Inst",
        "Meta-Llama-3.1-8B-Instruct": "Meta-Llama-8B-Inst",
    }.get(model_name, model_name)


def get_short_watermark_name(watermark_type: str) -> str:
    return {
        "maryland": "KGW (Distort)",
        "openai": "Gumbel (Dist-Free)",
    }.get(watermark_type, watermark_type)


def plot_scores(
    input_dir: str,
    output_file: str = "rewards_plot.pdf",
    model_name_to_plot: str = "Mistral-7B-Instruct-v0.3",  # Mistral-7B-Instruct-v0.3, Meta-Llama-3.1-8B-Instruct
    param_name_to_plot: str = "temperature",
    score_name: str = "rewards",
):
    data: dict[
        tuple[str, str], dict[str, list[tuple[float, list[float], list[float]]]]
    ] = process_files(input_dir, model_name_to_plot, param_name_to_plot, score_name)
    with prp.get_context(layout=prp.Layout.ICML, single_col=True) as (
        fig,
        axs,
    ):
        # plt.rcParams.update({"font.size": 6})
        colors = cycle(
            ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        )  # Colors chosen for clarity and distinction in publication
        markers = cycle(["o", "s", "D", "P", "X", "v", "^", "<", ">", "1", "2", "3"])
        # Plot unwatermarked scores first by averaging over all watermark_types
        # Based on plots these lines mostly coincide across watermark_types because the
        # text was generated from the same model without any watermark
        unwm_means = defaultdict(list)
        for (watermark_type, dataset_name), _ in data.items():
            for seed, values in data[(watermark_type, dataset_name)].items():
                wm_strengths, _, unwatermarked_rewards = zip(*values)
                unwm_means[seed].append([np.mean(u) for u in unwatermarked_rewards])
        color = next(colors)
        marker = next(markers)
        # Now average over all seeds
        unwm_means_avg = np.mean(list(unwm_means.values()), axis=0)
        # Now average over all watermark_types
        unwm_means_avg_avg = np.mean(unwm_means_avg, axis=0)
        # Compute the standard deviation
        unwm_stds = np.std(list(unwm_means.values()), axis=0)
        unwm_stds_avg = np.mean(unwm_stds, axis=0)
        axs.plot(
            wm_strengths,
            unwm_means_avg_avg,
            label="Unwatermarked",
            marker=marker,
            markersize=2,
            markerfacecolor="none",
            markeredgecolor=color,
            color=color,
            alpha=0.7,
            markeredgewidth=1,
            linestyle="dashed",
        )

        for (watermark_type, dataset_name), _ in data.items():
            print(f"Reading data for {watermark_type} on {dataset_name} and plotting")
            # Average over all seeds
            wm_means = defaultdict(list)
            for seed, values in data[(watermark_type, dataset_name)].items():
                wm_strengths, watermarked_rewards, unwatermarked_rewards = zip(*values)
                wm_means[seed].append([np.mean(w) for w in watermarked_rewards])
            wm_means_avg = np.mean(list(wm_means.values()), axis=0)
            wm_stds = np.std(list(wm_means.values()), axis=0)
            wm_means_avg = wm_means_avg.squeeze()
            wm_stds = wm_stds.squeeze()
            import pdb

            pdb.set_trace()
            color = next(colors)
            marker = next(markers)
            # Plot watermarked scores
            axs.plot(
                wm_strengths,
                wm_means_avg,
                label=f"{get_short_watermark_name(watermark_type)}",
                marker=marker,
                markersize=2,
                markerfacecolor=color,
                markeredgecolor=color,
                color=color,
                alpha=0.7,
                markeredgewidth=1,
                linestyle="-",
            )

        axs.set_xlabel(f"{param_name_to_plot} →", fontsize=6)
        axs.set_ylabel("Reward Score", fontsize=6, labelpad=3)
        axs.tick_params(axis="both", which="major", labelsize=5)
        axs.set_title(
            f"{score_name.capitalize()} Scores with Temperature for {get_short_model_name(model_name_to_plot)}",
            fontsize=6,
        )
        axs.legend(loc="best", fontsize=5)

        axs.set_xlim(left=0.2, right=1)
        axs.grid(True, linestyle="--", alpha=0.7)

        fig.savefig(output_file, format="pdf", bbox_inches="tight")
        print(f"Plot saved as {output_file}")


def main(
    input_dir: str,
    output_file: str = "rewards_plot.pdf",
    model_name_to_plot: str = "gpt-3",
    param_name_to_plot: str = "temperature",
    score_name: str = "rewards",  # or "truthfulness"
):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Assuming plot_score is a function that can handle multiple watermark types
    plot_scores(
        input_dir, output_file, model_name_to_plot, param_name_to_plot, score_name
    )


if __name__ == "__main__":
    Fire(main)
