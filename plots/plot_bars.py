import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_and_process_jsonl(filepath):
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


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
        watermarked_cats = process_categories(
            item.get("watermarked_unsafe_category", "")
        )
        for cat in watermarked_cats:
            if cat in safety_categories:
                category_counts[safety_categories[cat]] += 1

    return category_counts


def plot_safety_comparison(all_models_data, output_path):
    categories = list(list(all_models_data.values())[0]["openai"].keys())
    models = list(all_models_data.keys())

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
            fontsize=8,
            y=0.95,
            x=0.55,
        )

        bar_width = 0.35
        r1 = np.arange(len(categories))
        r2 = [x + bar_width for x in r1]

        for idx, (model_name, model_data) in enumerate(all_models_data.items()):
            ax = axes[idx]

            # Calculate absolute increases
            openai_increases = []
            maryland_increases = []
            for cat in categories:
                baseline = model_data["unwatermarked"][cat]
                openai_inc = model_data["openai"][cat] - baseline
                maryland_inc = model_data["maryland"][cat] - baseline
                openai_increases.append(openai_inc)
                maryland_increases.append(maryland_inc)

            # Plot bars for absolute increases
            ax.barh(
                r1,
                openai_increases,
                bar_width,
                label="Gumbel (Dist-Free)",
                alpha=0.7,
                color="#2ca02c",
            )
            ax.barh(
                r2,
                maryland_increases,
                bar_width,
                label="KGW (Distort)",
                alpha=0.7,
                color="#ff7f0e",
            )
            ax.set_xlabel(f"{model_name}", fontsize=8, labelpad=5)
            if idx == 0:
                ax.set_ylabel("Safety Categories", fontsize=8)
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


def group_files_by_model(files):
    files_dict = {}
    for file in files:
        model_name = extract_model_name(file)
        if model_name not in files_dict:
            files_dict[model_name] = []
        files_dict[model_name].append(file)
    return files_dict.values()


def extract_model_name(filepath):
    filename = str(filepath)
    if "Meta-Llama" in filename:
        return "LLaMA-8B-Inst"
    elif "Mistral" in filename:
        return "Mistral-7B-Inst"
    elif "gemma" in filename:
        return "Gemma-2-9B-Inst"
    return "Unknown"


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
    files = input_path.glob("*safety_scores*.jsonl")

    # Create a dictionary to store data for all models
    all_models_data = {}

    for model_files in group_files_by_model(files):
        model_name = extract_model_name(model_files[0])
        model_data = {"openai": {}, "maryland": {}, "unwatermarked": {}}

        for filepath in model_files:
            data = load_and_process_jsonl(filepath)

            if "openai" in str(filepath):
                model_data["openai"] = calculate_category_percentages(data)
                # Process unwatermarked data from the same file
                model_data["unwatermarked"] = calculate_category_percentages(
                    [
                        {
                            "watermarked_unsafe_category": d[
                                "unwatermarked_unsafe_category"
                            ]
                        }
                        for d in data
                    ]
                )
            elif "maryland" in str(filepath):
                model_data["maryland"] = calculate_category_percentages(data)

        all_models_data[model_name] = model_data

    output_file = output_path / "safety_comparison_all_models.png"
    plot_safety_comparison(all_models_data, output_file)
    print(f"Generated combined plot at {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
