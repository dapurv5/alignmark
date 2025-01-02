import glob
import json
import os
from typing import Any, Optional

import numpy as np
from fire import Fire

from plots.plot_utils import parse_filename


def create_output_filename(input_filename: str, n: int) -> str:
    watermark_type = parse_filename(input_filename)["watermark_type"]
    output_filename = input_filename.replace(
        watermark_type, f"{watermark_type}-BoN-{n}"
    )
    return output_filename


def get_top_score_index(
    candidate_scores: list[float],
    rng: np.random.Generator,
    preferred_categorical_label: Optional[Any] = None,
) -> int:
    if preferred_categorical_label is not None:
        candidate_indices = [
            i
            for i, score in enumerate(candidate_scores)
            if score == preferred_categorical_label
        ]
        if len(candidate_indices) == 0:
            return rng.choice(range(len(candidate_scores)))
        else:
            return rng.choice(candidate_indices)
    return np.argmax(candidate_scores)


def main(
    input_dir: str,
    output_dir: str,
    score_field: str,
    src_fields: list[str],
    tgt_score_field: str,
    tgt_fields: list[str],
    n: int,
    filename_pattern: str,
    preferred_categorical_label: Optional[str] = None,
):
    """Generate best of n samples from the input directory.
    Picks the top scoring index from the first n samples in the score fields.
    The scores fields are each a list of scores >= n samples
    (obtained by running run_generate.py with beam size > 1).

    The tgt_score_field is overwritten by the top score at the top score index.
    The tgt_associated_fields are overwritten by the associated fields of the top score index.
    Args:
        input_dir (str): The directory from output of run_generate.py.
        output_dir (str): The directory to save the best of n samples.
        score_field (str): The field to use for scoring. This field value is a list of scores. e.g.
            "watermarked_texts_reward_score"
        src_fields (list[str]): The fields to associate with the score field. e.g.
            [
                "watermarked_texts",
                "watermarked_texts_safety_eval",
                "watermarked_texts_unsafe_category",
            ]
            These field values are lists of strings.
        tgt_score_field (str): The field to overwrite with the top score. e.g.
            "watermarked_text_reward_score"
        tgt_fields (list[str]): The fields to overwrite with the top associated fields. e.g.
            [
                "watermarked_text",
                "watermarked_text_safety_eval",
                "watermarked_text_unsafe_category",
            ]
            These field values are just plain strings.
        n (int, optional): The number of samples to choose from in BoN.
        filename_pattern (str): The pattern to match the files in the input directory.
            e.g. "*_rewards.jsonl"
        preferred_categorical_label (str, optional): The preferred categorical label to
            pick from the score field. Sometimes the score field values might not be numeric.
            It might be categorical. e.g. "safe" or "unsafe".
            If preferred_categorical_label is provided, the top score index is picked from the
            score field values that match the preferred_categorical_label.
            e.g. "safe"
    """
    # If the output directory doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if isinstance(src_fields, str):
        src_fields = src_fields.split(",")
    if isinstance(tgt_fields, str):
        tgt_fields = tgt_fields.split(",")
    assert len(src_fields) == len(tgt_fields)

    # Get all the files in the input directory that match the filename pattern
    input_filepaths = glob.glob(os.path.join(input_dir, filename_pattern))
    input_filenames = [os.path.basename(filepath) for filepath in input_filepaths]
    output_filenames = [create_output_filename(fname, n) for fname in input_filenames]
    output_filepaths = [os.path.join(output_dir, fname) for fname in output_filenames]
    output_filepaths = [os.path.normpath(filepath) for filepath in output_filepaths]
    rng = np.random.default_rng(42)

    # Iterate over the input files
    for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
        with open(input_filepath, "r") as input_fp:
            with open(output_filepath, "w") as output_fp:
                for line in input_fp:
                    line = line.strip()
                    blob = json.loads(line)
                    candidate_scores = blob[score_field][:n]
                    top_score_index = get_top_score_index(
                        candidate_scores, rng, preferred_categorical_label
                    )
                    blob[tgt_score_field] = blob[score_field][top_score_index]
                    for i, src_field in enumerate(src_fields):
                        tgt_field = tgt_fields[i]
                        assert isinstance(blob[src_field], list)
                        assert (
                            isinstance(blob[tgt_field], str)
                            or isinstance(blob[tgt_field], float)
                            or isinstance(blob[tgt_field], int)
                        )
                        blob[tgt_field] = blob[src_field][top_score_index]
                    output_fp.write(json.dumps(blob) + "\n")


if __name__ == "__main__":
    Fire(main)
