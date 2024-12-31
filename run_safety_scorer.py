import glob
import os

import fire

from safety_scorer import SafetyScorerRegistry


def run_safety_scorer(
    input_path: str,
    output_path: str,
    scorer_name: str = "llama-guard",
    batch_size: int = 16,
):
    scorer = SafetyScorerRegistry.get(scorer_name)(batch_size=batch_size)
    # If input_path and output_path are files, then we use them directly
    # Otherwise, we assume they are directories and process each file in them
    if os.path.isfile(input_path) and os.path.isfile(output_path):
        scorer.compute_safety_scores(input_path, output_path)
    else:
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        for input_file in glob.glob(os.path.join(input_path, "*.jsonl")):
            if not input_file.endswith("_safety_scores.jsonl"):
                input_filename = os.path.basename(input_file)
                output_filename = (
                    os.path.splitext(input_filename)[0]
                    + "_safety_scores"
                    + os.path.splitext(input_filename)[1]
                )
                output_filename = os.path.join(output_path, output_filename)
                if not os.path.exists(output_filename) or (
                    os.path.exists(output_filename)
                    and sum(1 for _ in open(input_file))
                    != sum(1 for _ in open(output_filename))
                ):
                    scorer.compute_safety_scores(input_file, output_filename)


if __name__ == "__main__":
    fire.Fire(run_safety_scorer)
