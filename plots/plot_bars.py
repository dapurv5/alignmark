from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from plot_utils import (
    get_color,
    get_pattern,
    get_short_model_name,
    get_short_watermark_name,
    group_files_by_model,
    parse_filename,
    read_jsonl,
)


def process_categories(category_str):
    if not category_str:
        return []
    return [cat.strip() for cat in category_str.replace('"', "").split(",")]


def calculate_category_percentages(data):
    # Define all possible safety categories
    safety_categories = {
        "S1": "Illegal Activity",
        "S2": "Hate/Harass/Discrim/Violence",
        "S3": "Malware",
        "S4": "Physical Harm",
        "S5": "Economic Harm",
        "S6": "Fraud/Deception",
        "S7": "Adult Content",
        "S8": "Political Campaigning",
        "S9": "Privacy Violation",
        "S10": "Unauthorized Legal Advice",
        "S11": "Tailored Financial Advice",
        "S12": "Unauthorized Medical Advice",
        "S13": "High-Risk Gov. Decision",
        "S14": "Sexual Content",
        "S15": "Misinfo & Disinfo",
    }

    category_counts = {cat: 0 for cat in safety_categories.values()}

    for item in data:
        if "watermarked_unsafe_category" in item:
            watermarked_cats = process_categories(
                item.get("watermarked_unsafe_category", "")
            )
        else:
            watermarked_cats = process_categories(
                item.get("watermarked_text_unsafe_category", "")
            )
        for cat in watermarked_cats:
            if cat in safety_categories:
                category_counts[safety_categories[cat]] += 1

    return category_counts


def plot_safety_comparison(all_models_data, output_path):
    categories = list(list(all_models_data.values())[0]["unwatermarked"].keys())
    models = list(all_models_data.keys())
    watermark_types = list(all_models_data.values())[0].keys()

    import pub_ready_plots as prp

    # Use NeurIPS style
    with prp.get_context(layout=prp.Layout.NEURIPS, width_frac=1, height_frac=0.3) as (
        fig,
        ax,
    ):
        # Create a figure with horizontal subplots
        fig, axes = plt.subplots(1, len(models), constrained_layout=True)
        if len(models) == 1:
            axes = [axes]

        fig.suptitle(
            "Change in Number of Unsafe Responses with Watermarking",
            fontsize=10,
            y=0.97,
            x=0.55,
        )

        bar_width = 0.25  # 0.35
        # # Sort all_models_data by model size obtained by parsing the model name e.g. Qwen2.5-72B-Instruct -> 72B, Qwen2.5-14B -> 14B
        # all_models_data = dict(
        #     sorted(
        #         all_models_data.items(),
        #         key=lambda x: float(
        #             "".join(
        #                 c
        #                 for c in next(p for p in x[0].split("-") if "B" in p)
        #                 if c.isdigit() or c == "."
        #             )
        #         ),
        #     )
        # )
        # Sort all_models_data by model size obtained by parsing the model name e.g. Qwen2.5-72B-Instruct -> 72B, Qwen2.5-14B -> 14B
        all_models_data = dict(
            sorted(
                all_models_data.items(),
                key=lambda x: float(
                    "".join(
                        c
                        for c in next(p for p in x[0].split("-") if "B" in p)
                        if c.isdigit() or c == "."
                    )
                ),
            )
        )
        for idx, (model_name, model_data) in enumerate(all_models_data.items()):
            ax = axes[idx]

            # Calculate absolute increases for each watermark type
            increases = {}
            for watermark_type in watermark_types:
                if watermark_type != "unwatermarked":
                    increases[watermark_type] = []
                    for cat in categories:
                        baseline = model_data["unwatermarked"][cat]
                        inc = model_data[watermark_type][cat] - baseline
                        increases[watermark_type].append(inc)

            # Plot bars for absolute increases
            for i, watermark_type in enumerate(watermark_types):
                if watermark_type != "unwatermarked":
                    r = [x + i * bar_width for x in np.arange(len(categories))]
                    ax.barh(
                        r,
                        increases[watermark_type],
                        bar_width,
                        label=get_short_watermark_name(watermark_type),
                        alpha=0.7,
                        color=get_color(watermark_type),
                        hatch=get_pattern(watermark_type),
                        edgecolor="black",
                        linewidth=0.3,
                    )
            ax.set_xlabel(
                f"{get_short_model_name(model_name)}", fontsize=10, labelpad=5
            )
            if idx == 0:
                ax.set_ylabel("Safety Categories", fontsize=12)
                ax.set_yticks([r + bar_width / 2 for r in range(len(categories))])
                ax.set_yticklabels(categories, ha="right", fontsize=6)
            else:
                ax.set_yticks([r + bar_width / 2 for r in range(len(categories))])
                ax.set_yticklabels([])
            if idx == len(all_models_data) - 1:  # Only add legend to last subplot
                ax.legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc="lower right")
            ax.tick_params(axis="both", which="major", labelsize=6)

            # Add grid
            ax.grid(True, linestyle="--", alpha=0.7)

            # Add vertical line at 0
            ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()


def main(input_dir: str, output_dir: str = None):
    """
    Generate safety comparison plots for model outputs with different watermarking methods.

    Args:
        input_dir: Directory containing the safety score JSONL files
        output_dir: Directory to save the output plots (defaults to input_dir if not specified)
    """
    input_path = Path(input_dir)
    if not output_dir:
        output_dir = input_dir
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all safety score files
    files = list(input_path.glob("*safety_scores*.jsonl"))

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
            model_data_by_wm[watermark_type] = calculate_category_percentages(data)

            # Only collect unwatermarked data from first watermark type's file
            if not model_data_by_wm["unwatermarked"]:
                model_data_by_wm["unwatermarked"] = calculate_category_percentages(
                    [
                        (
                            {
                                "watermarked_unsafe_category": d[
                                    "unwatermarked_unsafe_category"
                                ]
                            }
                            if "watermarked_unsafe_category" in d
                            else {
                                "watermarked_text_unsafe_category": d[
                                    "unwatermarked_text_unsafe_category"
                                ]
                            }
                        )
                        for d in data
                    ]
                )

        all_models_data[model_name] = model_data_by_wm

    output_file = output_path / "safety_comparison_all_models.png"
    plot_safety_comparison(all_models_data, output_file)
    print(f"Generated combined plot at {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
