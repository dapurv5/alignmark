import glob
import json
import os

import numpy as np
from fire import Fire

from plots.plot_utils import get_short_watermark_name, parse_filename


def get_files(dir: str, pattern: str):
    return glob.glob(os.path.join(dir, pattern))


def init_empty_row():
    return {"Safe": [], "Unsafe": [], "Overrefusal": []}


def get_refusal_counts(
    refusal_file: str, normalize: bool = False, baseline: bool = False
):
    counts = {
        "wm": {"refusal": 0, "total": 0},
        "unwm": {"refusal": 0, "total": 0},
        "nr_of_lines": 0,
    }

    with open(refusal_file, "r") as input_fp:
        for line in input_fp:
            blob = json.loads(line)
            counts["nr_of_lines"] += 1

            if baseline:
                # Handle baseline case with multiple evaluations per text
                for prefix in ["watermarked", "unwatermarked"]:
                    key = "wm" if prefix == "watermarked" else "unwm"
                    if f"{prefix}_texts_refusal_eval" in blob:
                        for eval in blob[f"{prefix}_texts_refusal_eval"]:
                            counts[key]["total"] += 1
                            if eval == 1:
                                counts[key]["refusal"] += 1
            else:
                # Handle regular case with single evaluation per text
                for prefix in ["watermarked", "unwatermarked"]:
                    key = "wm" if prefix == "watermarked" else "unwm"
                    counts[key]["total"] += 1
                    refusal_eval_key = f"{prefix}_text_refusal_eval"
                    if refusal_eval_key not in blob:
                        refusal_eval_key = f"{prefix}_refusal_eval"
                    eval = blob[refusal_eval_key]
                    if eval == 1:
                        counts[key]["refusal"] += 1

    avg_wm_texts_per_line = counts["wm"]["total"] / counts["nr_of_lines"]
    avg_unwm_texts_per_line = counts["unwm"]["total"] / counts["nr_of_lines"]
    counts["wm"]["refusal"] = counts["wm"]["refusal"] / avg_wm_texts_per_line
    counts["unwm"]["refusal"] = counts["unwm"]["refusal"] / avg_unwm_texts_per_line
    if normalize:
        return (
            counts["wm"]["refusal"] / counts["nr_of_lines"],
            counts["unwm"]["refusal"] / counts["nr_of_lines"],
        )

    return counts["wm"]["refusal"], counts["unwm"]["refusal"]


def get_safety_counts(
    safety_file: str, normalize: bool = False, baseline: bool = False
):
    counts = {
        "wm": {"safe": 0, "unsafe": 0, "total": 0},
        "unwm": {"safe": 0, "unsafe": 0, "total": 0},
        "nr_of_lines": 0,
    }

    with open(safety_file, "r") as input_fp:
        for line in input_fp:
            blob = json.loads(line)
            counts["nr_of_lines"] += 1
            if baseline:
                # Handle baseline case with multiple evaluations per text
                for prefix in ["watermarked", "unwatermarked"]:
                    key = "wm" if prefix == "watermarked" else "unwm"
                    if f"{prefix}_texts_safety_eval" in blob:
                        for eval in blob[f"{prefix}_texts_safety_eval"]:
                            if eval in ["safe", "unsafe"]:
                                counts[key][eval] += 1
                                counts[key]["total"] += 1
            else:
                # Handle regular case with single evaluation per text
                for prefix in ["watermarked", "unwatermarked"]:
                    key = "wm" if prefix == "watermarked" else "unwm"
                    safety_eval_key = f"{prefix}_text_safety_eval"
                    if safety_eval_key not in blob:
                        safety_eval_key = f"{prefix}_safety_eval"
                    eval = blob[safety_eval_key]
                    if eval in ["safe", "unsafe"]:
                        counts[key][eval] += 1
                        counts[key]["total"] += 1

    avg_wm_texts_per_line = counts["wm"]["total"] / counts["nr_of_lines"]
    avg_unwm_texts_per_line = counts["unwm"]["total"] / counts["nr_of_lines"]
    counts["wm"]["safe"] = counts["wm"]["safe"] / avg_wm_texts_per_line
    counts["wm"]["unsafe"] = counts["wm"]["unsafe"] / avg_wm_texts_per_line
    counts["unwm"]["safe"] = counts["unwm"]["safe"] / avg_unwm_texts_per_line
    counts["unwm"]["unsafe"] = counts["unwm"]["unsafe"] / avg_unwm_texts_per_line
    if normalize:
        return (
            counts["wm"]["safe"] / counts["nr_of_lines"],
            counts["wm"]["unsafe"] / counts["nr_of_lines"],
            counts["unwm"]["safe"] / counts["nr_of_lines"],
            counts["unwm"]["unsafe"] / counts["nr_of_lines"],
        )

    return (
        counts["wm"]["safe"],
        counts["wm"]["unsafe"],
        counts["unwm"]["safe"],
        counts["unwm"]["unsafe"],
    )


def populate_safety_column(
    table: dict,
    safety_files: list[str],
    normalize: bool = False,
    baseline: bool = False,
):
    for safety_file in safety_files:
        filename = os.path.basename(safety_file)
        parsed_info = parse_filename(filename)
        model_name = parsed_info["model_name"]
        watermark_type = parsed_info["watermark_type"]
        cnt_wm_safe, cnt_wm_unsafe, cnt_unwm_safe, cnt_unwm_unsafe = get_safety_counts(
            safety_file, normalize=normalize, baseline=baseline
        )
        table_key = (model_name, watermark_type)
        if table_key not in table:
            table[table_key] = init_empty_row()
        table[table_key]["Safe"].append(cnt_wm_safe)
        table[table_key]["Unsafe"].append(cnt_wm_unsafe)
        table_key = (model_name, "Unwatermarked")
        if table_key not in table:
            table[table_key] = init_empty_row()
        table[table_key]["Safe"].append(cnt_unwm_safe)
        table[table_key]["Unsafe"].append(cnt_unwm_unsafe)


def populate_refusal_column(
    table: dict,
    refusal_files: list[str],
    normalize: bool = False,
    baseline: bool = False,
):
    for refusal_file in refusal_files:
        filename = os.path.basename(refusal_file)
        parsed_info = parse_filename(filename)
        model_name = parsed_info["model_name"]
        watermark_type = parsed_info["watermark_type"]
        cnt_wm_refusals, cnt_unwm_refusals = get_refusal_counts(
            refusal_file, normalize=normalize, baseline=baseline
        )
        table_key = (model_name, watermark_type)
        if table_key not in table:
            table[table_key] = init_empty_row()
        table[table_key]["Overrefusal"].append(cnt_wm_refusals)
        table_key = (model_name, "Unwatermarked")
        if table_key not in table:
            table[table_key] = init_empty_row()
        table[table_key]["Overrefusal"].append(cnt_unwm_refusals)


def main(
    safety_dir: str,
    refusal_dir: str,
    output_path: str,
    normalize: bool = False,
    baseline: bool = False,
):
    safety_files = get_files(safety_dir, "*_safety_scores.jsonl")
    refusal_files = get_files(refusal_dir, "*_refusal_scores.jsonl")
    table = {}
    populate_safety_column(table, safety_files, normalize=normalize, baseline=baseline)
    populate_refusal_column(
        table, refusal_files, normalize=normalize, baseline=baseline
    )
    # Sort the table by model name
    table = dict(sorted(table.items(), key=lambda x: x[0][0]))
    with open(output_path, "w") as output_fp:
        output_fp.write("Model Name\tSetting\tSafe\tUnsafe\tOverrefusal\n")
        for (model_name, watermark_type), row in table.items():
            # if any row has empty np array, skip it
            if any(len(values) == 0 for values in row.values()):
                continue
            for key, values in row.items():
                row[key] = np.mean(values)
            output_fp.write(
                f"{model_name}\t{get_short_watermark_name(watermark_type)}\t{row['Safe']}\t{row['Unsafe']}\t{row['Overrefusal']}\n"
            )


if __name__ == "__main__":
    Fire(main)
