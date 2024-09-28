import glob
import os

import fire
from reward_scorer import RewardScorerRegistry


def run_reward_scorer(
    input_path: str, output_path: str, reward_model: str = "llm-blender/PairRM"
):
    scorer = RewardScorerRegistry.get(reward_model)()
    # If input_path and output_path are files, then we use them directly
    # Otherwise, we assume they are directories and process each file in them
    if os.path.isfile(input_path) and os.path.isfile(output_path):
        scorer.compute_rewards(input_path, output_path)
    else:
        for input_file in glob.glob(os.path.join(input_path, "*.jsonl")):
            if not input_file.endswith("_rewards.jsonl"):
                input_filename = os.path.basename(input_file)
                output_filename = (
                    os.path.splitext(input_filename)[0]
                    + "_rewards"
                    + os.path.splitext(input_filename)[1]
                )
                output_filename = os.path.join(output_path, output_filename)
                if not os.path.exists(output_filename) or (
                    os.path.exists(output_filename)
                    and sum(1 for _ in open(input_file))
                    != sum(1 for _ in open(output_filename))
                ):
                    scorer.compute_rewards(input_file, output_filename)


if __name__ == "__main__":
    fire.Fire(run_reward_scorer)
