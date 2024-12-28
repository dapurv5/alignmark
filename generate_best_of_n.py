import glob
import json
import os

import numpy as np
from fire import Fire

from plots.plot_utils import parse_filename


def create_output_filename(input_filename: str, n: int) -> str:
    watermark_type = parse_filename(input_filename)["watermark_type"]
    output_filename = input_filename.replace(
        watermark_type, f"{watermark_type}-BoN-{n}"
    )
    # import pdb

    # pdb.set_trace()
    return output_filename


def main(
    input_dir: str,
    output_dir: str,
    score_field: str,
    src_fields: list[str],
    tgt_score_field: str,
    tgt_fields: list[str],
    n: int,
    filename_pattern: str,
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
        score_fields (list[str]): The fields to use for scoring. e.g.
            [
                "watermarked_texts_reward_score",
                "watermarked_texts_truthfulness_score"
            ]
        associated_fields (list[list[str]]): The fields to associate with the score fields. e.g.
            [
                [
                    "watermarked_texts",
                    "watermarked_texts_safety_eval",
                    "watermarked_texts_unsafe_category",
                ]
            ],
        tgt_score_field (str): The field to overwrite with the top score. e.g.
            "watermarked_text_reward_score"
        tgt_associated_fields (list[str]): The fields to overwrite with the top associated fields. e.g.
            "watermarked_text"
        n (int, optional): The number of samples to choose from in BoN.
        filename_pattern (str): The pattern to match the files in the input directory.
            e.g. "*_rewards.jsonl"
    """
    # If the output directory doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if isinstance(src_fields, str):
        src_fields = [src_fields]
    if isinstance(tgt_fields, str):
        tgt_fields = [tgt_fields]
    assert len(src_fields) == len(tgt_fields)

    # Get all the files in the input directory that match the filename pattern
    input_filepaths = glob.glob(os.path.join(input_dir, filename_pattern))
    input_filenames = [os.path.basename(filepath) for filepath in input_filepaths]
    output_filenames = [create_output_filename(fname, n) for fname in input_filenames]
    output_filepaths = [os.path.join(output_dir, fname) for fname in output_filenames]
    output_filepaths = [os.path.normpath(filepath) for filepath in output_filepaths]
    import pdb

    pdb.set_trace()

    # Iterate over the input files
    for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
        with open(input_filepath, "r") as input_fp:
            with open(output_filepath, "w") as output_fp:
                for line in input_fp:
                    line = line.strip()
                    blob = json.loads(line)
                    candidate_scores = blob[score_field][:n]
                    top_score_index = np.argmax(candidate_scores)
                    blob[tgt_score_field] = [blob[score_field][top_score_index]]
                    for i, src_field in enumerate(src_fields):
                        assert isinstance(blob[src_field], list)
                        assert isinstance(blob[tgt_fields[i]], str)
                        blob[tgt_fields[i]] = [blob[src_field][top_score_index]]
                    output_fp.write(json.dumps(blob) + "\n")


if __name__ == "__main__":
    Fire(main)
