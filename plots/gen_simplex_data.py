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


def get_refusal_counts(refusal_file: str, normalize: bool = False):
    cnt_wm_refusals = 0
    cnt_unwm_refusals = 0
    cnt_total_lines = 0
    with open(refusal_file, "r") as input_fp:
        for line in input_fp:
            blob = json.loads(line)
            if blob["watermarked_text_refusal_eval"] == 1:
                cnt_wm_refusals += 1
            if blob["unwatermarked_text_refusal_eval"] == 1:
                cnt_unwm_refusals += 1
            cnt_total_lines += 1
    if normalize:
        return (
            cnt_wm_refusals / cnt_total_lines,
            cnt_unwm_refusals / cnt_total_lines,
        )
    return cnt_wm_refusals, cnt_unwm_refusals


def get_safety_counts(safety_file: str, normalize: bool = False):
    cnt_wm_safe = 0
    cnt_wm_unsafe = 0
    cnt_unwm_safe = 0
    cnt_unwm_unsafe = 0
    cnt_total_lines = 0
    with open(safety_file, "r") as input_fp:
        for line in input_fp:
            blob = json.loads(line)
            if blob["watermarked_text_safety_eval"] == "safe":
                cnt_wm_safe += 1
            elif blob["watermarked_text_safety_eval"] == "unsafe":
                cnt_wm_unsafe += 1
            if blob["unwatermarked_text_safety_eval"] == "safe":
                cnt_unwm_safe += 1
            elif blob["unwatermarked_text_safety_eval"] == "unsafe":
                cnt_unwm_unsafe += 1
            cnt_total_lines += 1
    if normalize:
        return (
            cnt_wm_safe / cnt_total_lines,
            cnt_wm_unsafe / cnt_total_lines,
            cnt_unwm_safe / cnt_total_lines,
            cnt_unwm_unsafe / cnt_total_lines,
        )
    return cnt_wm_safe, cnt_wm_unsafe, cnt_unwm_safe, cnt_unwm_unsafe


def populate_safety_column(
    table: dict,
    safety_files: list[str],
    normalize: bool = False,
):
    for safety_file in safety_files:
        filename = os.path.basename(safety_file)
        parsed_info = parse_filename(filename)
        model_name = parsed_info["model_name"]
        watermark_type = parsed_info["watermark_type"]
        cnt_wm_safe, cnt_wm_unsafe, cnt_unwm_safe, cnt_unwm_unsafe = get_safety_counts(
            safety_file, normalize=normalize
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
):
    for refusal_file in refusal_files:
        filename = os.path.basename(refusal_file)
        parsed_info = parse_filename(filename)
        model_name = parsed_info["model_name"]
        watermark_type = parsed_info["watermark_type"]
        cnt_wm_refusals, cnt_unwm_refusals = get_refusal_counts(
            refusal_file, normalize=normalize
        )
        table_key = (model_name, watermark_type)
        if table_key not in table:
            table[table_key] = init_empty_row()
        table[table_key]["Overrefusal"].append(cnt_wm_refusals)
        table_key = (model_name, "Unwatermarked")
        if table_key not in table:
            table[table_key] = init_empty_row()
        table[table_key]["Overrefusal"].append(cnt_unwm_refusals)


def main(safety_dir: str, refusal_dir: str, output_path: str, normalize: bool = False):
    safety_files = get_files(safety_dir, "*_safety_scores.jsonl")
    refusal_files = get_files(refusal_dir, "*_refusal_scores.jsonl")
    table = {}
    populate_safety_column(table, safety_files, normalize=normalize)
    populate_refusal_column(table, refusal_files, normalize=normalize)
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
