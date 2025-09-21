from pathlib import Path
from typing import Any

import fire
from plot_utils import (
    get_color,
    get_model_size_for_sort,
    get_pattern,
    get_short_model_name,
    get_short_watermark_name,
    group_files_by_model,
    parse_filename,
    read_jsonl,
)


def plot_score_comparison(
    all_models_data: dict[str, dict[str, float]], output_file: Path, score_name: str
) -> None:
    # Sort models by numeric size (e.g., 0.5B < 1.5B < 3B < 7B < 14B)
    models: list[str] = sorted(
        list(all_models_data.keys()),
        key=lambda m: (get_model_size_for_sort(m), get_short_model_name(m).lower()),
    )
    watermark_types = list(all_models_data.values())[0].keys()
    watermark_types_sorted = sorted(
        list(watermark_types), key=lambda wm: get_short_watermark_name(wm).lower()
    )

    import numpy as np
    import pub_ready_plots as prp

    # Use NeurIPS style
    with prp.get_context(layout=prp.Layout.NEURIPS, width_frac=1, height_frac=0.3) as (
        fig,
        ax_obj,
    ):
        # Inform the type checker that we have a single Axes object, not an ndarray
        from typing import cast

        from matplotlib.axes import Axes

        ax: Axes = cast(Axes, ax_obj)
        # Calculate positions
        n_groups = len(models)
        n_bars = len(watermark_types_sorted)
        bar_width = 0.10
        group_width = bar_width * n_bars
        group_positions = np.arange(n_groups) * (
            group_width + 0.3
        )  # Add gap between groups

        # Add horizontal grid lines
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)  # Ensure grid lines are below the bars

        # Plot bars for each model
        for model_idx, model in enumerate(models):
            for wm_idx, wm_type in enumerate(watermark_types_sorted):
                x_pos = group_positions[model_idx] + wm_idx * bar_width
                height = all_models_data[model][wm_type]
                ax.bar(
                    x_pos,
                    height,
                    bar_width,
                    label=get_short_watermark_name(wm_type) if model_idx == 0 else "",
                    alpha=0.7,
                    color=get_color(wm_type),
                    hatch=get_pattern(wm_type),
                    edgecolor="black",
                    linewidth=0.5,
                )

        # Customize the plot
        ax.set_ylabel(f"{score_name.capitalize()} Score", fontsize=14)
        ax.set_xticks(group_positions + (group_width - bar_width) / 2)
        ax.set_xticklabels(
            [get_short_model_name(model) for model in models],
            rotation=0,
            ha="center",
            fontsize=10,
        )

        # Calculate optimal column distribution
        handles, labels = ax.get_legend_handles_labels()
        total_items = len(labels)

        if total_items <= 3:
            # Single row for 3 or fewer items
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.2),
                ncol=total_items,
                fontsize=9,
                handletextpad=0.5,
                columnspacing=1.0,
            )
        else:
            # Split into two rows for more than 3 items
            import math

            cols_first_row = math.ceil(total_items / 2)
            items_first_row = cols_first_row

            legend1 = ax.legend(
                handles[:items_first_row],
                labels[:items_first_row],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.35),
                ncol=cols_first_row,
                fontsize=9,
                handletextpad=0.5,
                columnspacing=1.0,
            )
            ax.add_artist(legend1)

            ax.legend(
                handles[items_first_row:],
                labels[items_first_row:],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.2),
                ncol=total_items - items_first_row,
                fontsize=9,
                handletextpad=0.5,
                columnspacing=1.0,
            )

        # Adjust layout to prevent label cutoff
        fig.tight_layout()

        # Save the plot with increased top margin to accommodate legend
        fig.savefig(output_file, bbox_inches="tight", dpi=300, pad_inches=0.3)

        # Print table of values to terminal
        from math import isnan

        header = ["Model"] + [
            get_short_watermark_name(wm_type) for wm_type in watermark_types_sorted
        ]
        rows: list[list[str]] = []
        for model in models:
            row: list[str] = [get_short_model_name(model)]
            for wm_type in watermark_types_sorted:
                value = all_models_data[model][wm_type]
                cell = "-" if isnan(value) else f"{value:.3f}"
                row.append(cell)
            rows.append(row)

        # Compute column widths
        col_widths = [len(col) for col in header]
        for row in rows:
            for idx, cell in enumerate(row):
                if len(cell) > col_widths[idx]:
                    col_widths[idx] = len(cell)

        def format_row(cells: list[str]) -> str:
            return "  ".join(
                cell.ljust(col_widths[idx]) for idx, cell in enumerate(cells)
            )

        print("\n" + format_row(header))
        print(format_row(["-" * w for w in col_widths]))
        for row in rows:
            print(format_row(row))


def get_scores(
    data: list[dict[str, Any]], score_name: str, prefix: str = "watermarked"
) -> list[float]:
    scores: list[float] = []
    for item in data:
        scores.append(item[f"{prefix}_text_{score_name}_score"])  # type: ignore[index]
    return scores


def main(
    input_dir: str, output_dir: str | None = None, score_name: str = "reward"
) -> None:
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
        files = list(input_path.glob(f"*{score_name}s.jsonl"))
    else:
        files = list(input_path.glob(f"*{score_name}.jsonl"))
    # Sort files so files are sorted by watermark type.
    files = sorted(files, key=lambda f: parse_filename(f.name)["watermark_type"])

    # Create a dictionary to store data for all models
    # Collect raw scores for each model and watermark type
    all_models_scores: dict[str, dict[str, list[float]]] = {}
    all_watermark_types = set()
    for file in files:
        filename = file.name
        parsed_info = parse_filename(filename)
        all_watermark_types.add(parsed_info["watermark_type"])

    for model_files in group_files_by_model(files):
        parsed_info = parse_filename(model_files[0].name)
        model_name = parsed_info["model_name"]
        model_scores_by_wm: dict[str, list[float]] = {
            wm: [] for wm in all_watermark_types
        }
        model_scores_by_wm["unwatermarked"] = []

        for filepath in model_files:
            data = read_jsonl(filepath)
            filename = filepath.name
            parsed_info = parse_filename(filename)
            watermark_type = parsed_info["watermark_type"]
            # Each model should only have one file per watermark type (in other words no sweep across temperature, etc)
            assert not model_scores_by_wm[watermark_type]
            model_scores_by_wm[watermark_type].extend(
                get_scores(data, score_name, "watermarked")
            )
            # Only collect unwatermarked data from first watermark type's file
            if not model_scores_by_wm["unwatermarked"]:
                model_scores_by_wm["unwatermarked"].extend(
                    get_scores(data, score_name, "unwatermarked")
                )

        all_models_scores[model_name] = model_scores_by_wm
    # Average all scores into a separate structure for plotting
    all_models_data: dict[str, dict[str, float]] = {}
    for model_name, model_data in all_models_scores.items():
        averaged: dict[str, float] = {}
        for watermark_type, scores in model_data.items():
            averaged[watermark_type] = (
                (sum(scores) / len(scores)) if scores else float("nan")
            )
        all_models_data[model_name] = averaged

    output_file = output_path / f"score_comparison_all_models_{score_name}.png"
    # Sort models by numeric size for consistent table ordering
    all_models_data = dict(
        sorted(
            all_models_data.items(),
            key=lambda x: (
                get_model_size_for_sort(x[0]),
                get_short_model_name(x[0]).lower(),
            ),
        )
    )
    plot_score_comparison(all_models_data, output_file, score_name)
    print(f"Generated combined plot at {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
