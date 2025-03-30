import math
import os
from collections import defaultdict
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pub_ready_plots as prp
from fire import Fire

from plots.plot_utils import (
    get_color,
    get_short_model_name,
    get_short_param_name_to_plot,
    get_short_watermark_name,
    process_files,
)


def plot_scores(
    input_dir: str,
    output_file: str = "rewards_plot.pdf",
    model_name_to_plot: str = "Mistral-7B-Instruct-v0.3",  # Mistral-7B-Instruct-v0.3, Meta-Llama-3.1-8B-Instruct
    param_name_to_plot: str = "temperature",
    score_name: str = "rewards",
    plot_theoretical_sqrt_log: bool = False,
):
    data: dict[
        tuple[str, str], dict[str, list[tuple[float, list[float], list[float]]]]
    ] = process_files(input_dir, model_name_to_plot, param_name_to_plot, score_name)
    with prp.get_context(layout=prp.Layout.ICML, single_col=True) as (
        fig,
        axs,
    ):
        # plt.rcParams.update({"font.size": 6})
        markers = cycle(["o", "s", "D", "P", "X", "v", "^", "<", ">", "1", "2", "3"])
        # Plot unwatermarked scores first by averaging over all watermark_types
        # Based on plots these lines mostly coincide across watermark_types because the
        # text was generated from the same model without any watermark
        # collect over seeds and dataset_names
        unwm_means = defaultdict(lambda: defaultdict(list))
        all_unwm_scores = []
        for (watermark_type, dataset_name), _ in data.items():
            for seed, values in data[(watermark_type, dataset_name)].items():
                print(f"Processing seed {seed} for {watermark_type} on {dataset_name}")
                wm_strengths, _, unwatermarked_scores = zip(*values)
                unwm_means[dataset_name][seed].append(
                    [np.mean(u) for u in unwatermarked_scores]
                )
                for u in unwatermarked_scores:
                    all_unwm_scores.extend(u)
        for dataset_name in unwm_means:
            # color = next(colors)
            marker = next(markers)
            # Now average over all seeds
            unwm_means_avg = np.mean(list(unwm_means[dataset_name].values()), axis=0)
            # average over all watermark_types because unwatermarked text is generated for each watermark_type
            unwm_means_avg_avg = np.mean(unwm_means_avg, axis=0)
            # Compute the standard deviation
            unwm_stds = np.std(list(unwm_means[dataset_name].values()), axis=0)
            # average over all watermark_types because unwatermarked text is generated for each watermark_type
            unwm_stds_avg = np.mean(unwm_stds, axis=0)
            axs.plot(
                wm_strengths,
                unwm_means_avg_avg,
                label="Unwatermarked",
                marker=marker,
                markersize=2,
                markerfacecolor="none",
                markeredgecolor=get_color("unwatermarked"),
                color=get_color("unwatermarked"),
                alpha=0.7,
                markeredgewidth=1,
                linestyle="dashed",
            )

        std_wm_scores_by_watermark_type = {}
        for (watermark_type, dataset_name), _ in data.items():
            print(f"Reading data for {watermark_type} on {dataset_name} and plotting")
            # Average over all seeds
            all_watermarked_scores = []
            wm_means = defaultdict(list)
            for seed, values in data[(watermark_type, dataset_name)].items():
                wm_strengths, watermarked_scores, _ = zip(*values)
                for w in watermarked_scores:
                    all_watermarked_scores.extend(w)
                wm_means[seed].append([np.mean(w) for w in watermarked_scores])
            wm_means_avg = np.mean(list(wm_means.values()), axis=0)
            wm_stds = np.std(list(wm_means.values()), axis=0)
            wm_means_avg = wm_means_avg.squeeze()
            wm_stds = wm_stds.squeeze()
            marker = next(markers)
            # Plot watermarked scores
            axs.plot(
                wm_strengths,
                wm_means_avg,
                label=f"{get_short_watermark_name(watermark_type)}",
                marker=marker,
                markersize=2,
                markerfacecolor=get_color(watermark_type),
                markeredgecolor=get_color(watermark_type),
                color=get_color(watermark_type),
                alpha=0.7,
                markeredgewidth=1,
                linestyle="-",
            )

            if plot_theoretical_sqrt_log:
                std_wm_scores_by_watermark_type[watermark_type] = np.std(
                    all_watermarked_scores, axis=0
                )
                std_unwm_scores = np.std(all_unwm_scores, axis=0)
                baseline_degradation = wm_means_avg[0] - unwm_means_avg_avg[0]
                N_arr = np.array(wm_strengths)
                # y_pred = (
                #     unwm_means_avg_avg[0]
                #     + baseline_degradation
                #     + std_wm_scores_by_watermark_type[watermark_type]
                #     * (1 - np.exp(1 - 9 / N_arr) ** np.ceil(N_arr / 3))
                #     * np.sqrt(np.log(N_arr))
                # )
                y_pred = (
                    unwm_means_avg_avg[0]
                    + baseline_degradation
                    + std_wm_scores_by_watermark_type[watermark_type]
                    * 0.67
                    * np.sqrt(np.log(N_arr))
                )
                print(
                    f"Baseline degradation for {watermark_type}: {baseline_degradation}"
                )
                print(
                    f"std of {watermark_type}: {std_wm_scores_by_watermark_type[watermark_type]}"
                )
                axs.plot(
                    wm_strengths,
                    y_pred,
                    label=f"{get_short_watermark_name(watermark_type + '-theoretical')}",
                    marker=marker,
                    markersize=2,
                    markerfacecolor=get_color(watermark_type),
                    markeredgecolor=get_color(watermark_type),
                    color=get_color(watermark_type),
                    alpha=0.7,
                    markeredgewidth=1,
                    linestyle="dotted",
                )

        axs.set_xlabel(
            f"{get_short_param_name_to_plot(param_name_to_plot)} →", fontsize=7
        )
        axs.set_ylabel(f"{score_name.capitalize()} Score", fontsize=7, labelpad=3)
        axs.tick_params(axis="both", which="major", labelsize=5)
        axs.set_title(
            f"{score_name.capitalize()} Scores with {get_short_param_name_to_plot(param_name_to_plot)} for {get_short_model_name(model_name_to_plot)}",
            fontsize=5,
        )
        axs.legend(loc="best", fontsize=5)
        # Get the range of the x-axis
        x_range = axs.get_xlim()
        # Set the x-axis range to be from 0.2 to 1
        # axs.set_xlim(left=0.2, right=1)  # this was for temperature
        # Get first x value from wm_strengths
        first_x = min(wm_strengths)
        last_x = max(wm_strengths)
        axs.set_xlim(left=first_x, right=last_x)
        # Get current ticks
        ticks = axs.get_xticks()
        # Keep only every other tick
        axs.set_xticks(ticks[::2])
        axs.grid(True, linestyle="--", alpha=0.7)

        fig.savefig(output_file, format="pdf", bbox_inches="tight")
        print(f"Plot saved as {output_file}")


def main(
    input_dir: str,
    output_file: str = "rewards_plot.pdf",
    model_name_to_plot: str = "gpt-3",
    param_name_to_plot: str = "temperature",
    score_name: str = "rewards",  # or "truthfulness"
    plot_theoretical_sqrt_log: bool = False,
):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Assuming plot_score is a function that can handle multiple watermark types
    plot_scores(
        input_dir,
        output_file,
        model_name_to_plot,
        param_name_to_plot,
        score_name,
        plot_theoretical_sqrt_log,
    )


if __name__ == "__main__":
    Fire(main)
