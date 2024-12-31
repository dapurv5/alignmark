from pathlib import Path

import fire
from plot_utils import (
    group_files_by_model,
    parse_filename,
    read_jsonl,
)


def plot_score_comparison(all_models_data, output_file, score_name):
    models = list(all_models_data.keys())
    watermark_types = list(all_models_data.values())[0].keys()

    import numpy as np
    import pub_ready_plots as prp
    from plot_utils import get_color, get_short_model_name, get_short_watermark_name

    # Use NeurIPS style
    with prp.get_context(layout=prp.Layout.NEURIPS, width_frac=1, height_frac=0.3) as (
        fig,
        ax,
    ):
        # Calculate positions
        n_groups = len(models)
        n_bars = len(watermark_types)
        bar_width = 0.10
        group_width = bar_width * n_bars
        group_positions = np.arange(n_groups) * (
            group_width + 0.3
        )  # Add gap between groups

        # Add horizontal grid lines
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)  # Ensure grid lines are below the bars

        # Plot bars for each model
        for model_idx, model in enumerate(models):
            for wm_idx, wm_type in enumerate(watermark_types):
                x_pos = group_positions[model_idx] + wm_idx * bar_width
                height = all_models_data[model][wm_type]
                ax.bar(
                    x_pos,
                    height,
                    bar_width,
                    label=get_short_watermark_name(wm_type) if model_idx == 0 else "",
                    alpha=0.7,
                    color=get_color(wm_type),
                )

        # Customize the plot
        ax.set_ylabel(f"{score_name.capitalize()} Score", fontsize=12)
        ax.set_xticks(group_positions + (group_width - bar_width) / 2)
        ax.set_xticklabels(
            [get_short_model_name(model) for model in models], rotation=0, ha="center"
        )

        # Add legend inside the plot
        ax.legend(loc="upper right")

        # Adjust layout to prevent label cutoff
        fig.tight_layout()

        # Save the plot
        fig.savefig(output_file, bbox_inches="tight", dpi=300)


def get_scores(data, score_name, prefix="watermarked"):
    scores = []
    for item in data:
        scores.append(item[f"{prefix}_text_{score_name}_score"])
    return scores


def main(input_dir: str, output_dir: str = None, score_name: str = "reward"):
    """
    Generate score comparison plots for model outputs with different watermarking methods.

    Args:
        input_dir: Directory containing the safety score JSONL files
        output_dir: Directory to save the output plots (defaults to input_dir if not specified)
        score_name: Name of the score to plot (defaults to "reward", can also be "truthfulness")
    """
    input_path = Path(input_dir)
    if not output_dir:
        output_dir = input_dir
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all score files
    if score_name == "reward":
        # Hack because reward score files are annoyingly named _rewards.jsonl instead of _reward.jsonl
        files = list(input_path.glob("*{score_name}s.jsonl"))
    else:
        files = list(input_path.glob(f"*{score_name}.jsonl"))

    # Create a dictionary to store data for all models
    all_models_data = {}
    all_watermark_types = set()
    for file in files:
        filename = file.name
        parsed_info = parse_filename(filename)
        all_watermark_types.add(parsed_info["watermark_type"])

    for model_files in group_files_by_model(files):
        parsed_info = parse_filename(model_files[0].name)
        model_name = parsed_info["model_name"]
        model_data_by_wm = {wm: {} for wm in all_watermark_types}
        model_data_by_wm["unwatermarked"] = {}

        for filepath in model_files:
            data = read_jsonl(filepath)
            filename = filepath.name
            parsed_info = parse_filename(filename)
            watermark_type = parsed_info["watermark_type"]
            # Each model should only have one file per watermark type (in other words no sweep across temperature, etc)
            assert not model_data_by_wm[watermark_type]
            if not model_data_by_wm[watermark_type]:
                model_data_by_wm[watermark_type] = []
            model_data_by_wm[watermark_type].extend(
                get_scores(data, score_name, "watermarked")
            )
            # Only collect unwatermarked data from first watermark type's file
            if not model_data_by_wm["unwatermarked"]:
                model_data_by_wm["unwatermarked"] = []
                model_data_by_wm["unwatermarked"].extend(
                    get_scores(data, score_name, "unwatermarked")
                )

        all_models_data[model_name] = model_data_by_wm
    # Average all scores
    for model_name, model_data in all_models_data.items():
        for watermark_type, scores in model_data.items():
            model_data[watermark_type] = sum(scores) / len(scores)

    output_file = output_path / f"score_comparison_all_models_{score_name}.png"
    plot_score_comparison(all_models_data, output_file, score_name)
    print(f"Generated combined plot at {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
