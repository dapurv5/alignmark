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


def parse_filename(filename: str) -> tuple[float, str, str, str, str]:
    filename_parts = filename.split("_")
    dataset_name = filename_parts[1]
    model_name = filename_parts[2]
    watermark_type = filename_parts[3]
    wm_strength_param_name = filename_parts[4]
    wm_strength_param = float(filename_parts[5])
    return (
        wm_strength_param,  # E.g. 2.0, 2.5, 3.0
        wm_strength_param_name,  # E.g. delta
        dataset_name,
        model_name,
        watermark_type,
    )


def process_files(
    input_dir: str,
    watermark_type_to_plot: str,
    wm_strength_param_name_to_plot: str,
) -> dict[tuple[str, str], list[tuple[float, list[float], list[float]]]]:
    """
    Process all the files in the input directory and return a dictionary
    with (model_name, dataset_name) as keys and a list of tuples containing
    the wm_strength_param, watermarked reward scores, and unwatermarked reward scores.

    Args:
        input_dir (str): The directory containing the reward files.
        watermark_type_to_plot (str): The type of watermark to plot.
        wm_strength_param_name_to_plot (str): The name of the watermark strength parameter.
    Returns:
        dict[tuple[str, str], list[tuple[float, list[float], list[float]]]]: A dictionary
        with (model_name, dataset_name) as keys and a list of tuples containing
        the wm_strength_param, watermarked reward scores, and unwatermarked reward scores.
    """
    data: dict[tuple[str, str], list[tuple[float, list[float], list[float]]]] = {}
    for filename in os.listdir(input_dir):
        if filename.endswith("_rewards.jsonl"):
            file_path = os.path.join(input_dir, filename)
            (
                wm_strength_param,
                wm_strength_param_name,
                dataset_name,
                model_name,
                watermark_type,
            ) = parse_filename(filename)
            if (
                wm_strength_param is not None
                and wm_strength_param_name == wm_strength_param_name_to_plot
                and watermark_type == watermark_type_to_plot
            ):
                blobs = read_jsonl(file_path)
                watermarked = [blob["watermarked_text_reward_score"] for blob in blobs]
                unwatermarked = [
                    blob["unwatermarked_text_reward_score"] for blob in blobs
                ]
                key = (model_name, dataset_name)
                if key not in data:
                    data[key] = []
                data[key].append((float(wm_strength_param), watermarked, unwatermarked))

    # Sort the lists for each key by wm_strength_param
    for key in data:
        data[key] = sorted(data[key], key=lambda x: x[0])

    return data


def plot_reward_diff(
    input_dir: str,
    output_file: str = "rewards_plot.pdf",
    watermark_type_to_plot: str = "KGW",
    wm_strength_param_name_to_plot: str = "delta",
):
    data: dict[
        tuple[str, str], list[tuple[float, list[float], list[float]]]
    ] = process_files(input_dir, watermark_type_to_plot, wm_strength_param_name_to_plot)
    with prp.get_context(layout=prp.Layout.ICML, single_col=True) as (
        fig,
        axs,
    ):
        plt.rcParams.update({"font.size": 6})
        plt.rcParams.update({"font.size": 6})
        colors = cycle(plt.cm.tab10.colors)  # Create a color cycle
        for (model_name, dataset_name), values in data.items():
            print(f"Reading data for {model_name} on {dataset_name} and plotting")
            wm_strengths, watermarked_rewards, unwatermarked_rewards = zip(*values)
            # wm_strengths is a tuple of floats
            # watermarked_rewards is a tuple of list of floats
            # unwatermarked_rewards is a tuple of list of floats
            reward_diffs = []
            reward_diff_stds = []
            for idx, wm_strength in enumerate(wm_strengths):
                reward_diff_arrays = [
                    np.array(w) - np.array(u)
                    for w, u in zip(
                        watermarked_rewards[idx], unwatermarked_rewards[idx]
                    )
                ]
                reward_diffs.append(np.mean(reward_diff_arrays))
                reward_diff_stds.append(np.std(reward_diff_arrays))
            color = next(colors)
            axs.plot(
                wm_strengths,
                reward_diffs,
                # label=f"{model_name} on {dataset_name}",
                label=f"{model_name}",
                marker={
                    "Meta-Llama-3.1-8B-Instruct": "v",
                    "Meta-Llama-3.1-34B-Instruct": "s",
                    "Meta-Llama-3.1-70B-Instruct": "o",
                }.get(model_name, "o"),
                markersize=2,
                markerfacecolor=color,
                markeredgecolor=color,
                color=color,
                alpha=0.7,
                markeredgewidth=1,
                linestyle="--",  # Add this line to make the line dotted
            )
            axs.errorbar(
                wm_strengths,
                reward_diffs,
                yerr=reward_diff_stds,
                fmt="none",  # Remove connecting lines
                ecolor=color,
                capsize=2,
                alpha=0.7,
                elinewidth=0.5,
            )

        axs.set_xlabel(
            f"Watermark Strength ({wm_strength_param_name_to_plot}) →", fontsize=6
        )
        axs.set_ylabel(r"$R_w - R_u$", fontsize=6, labelpad=3)
        # reduce the size of ticks on x and y axis
        axs.tick_params(axis="both", which="major", labelsize=5)
        axs.set_title(
            r"Reward score gap with increasing watermark strength", fontsize=6
        )
        axs.legend(loc="best", fontsize=5)

        axs.set_xlim(left=0, right=1.0)  # Set x-axis to start at 2
        axs.grid(True, linestyle="--", alpha=0.7)  # Add grid lines

        # plt.tight_layout()
        fig.savefig(output_file, format="pdf", bbox_inches="tight")
        print(f"Plot saved as {output_file}")


def main(
    input_dir: str,
    output_file: str = "rewards_plot.pdf",
    watermark_type_to_plot: str = "KGW",
    wm_strength_param_name_to_plot: str = "delta",
):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_reward_diff(
        input_dir,
        output_file,
        watermark_type_to_plot,
        wm_strength_param_name_to_plot,
    )


if __name__ == "__main__":
    Fire(main)
