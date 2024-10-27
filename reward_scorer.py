import json
import logging
import os
from abc import abstractmethod
from collections import OrderedDict

import llm_blender
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RewardScorerBase:
    @abstractmethod
    def get_reward_score(self, prompt: str, texts: list[str]) -> list[float]:
        raise NotImplementedError("Subclasses must implement this method")

    def compute_rewards(self, input_path: str, output_path: str):
        logger.info(f"Computing rewards for {input_path} and writing to {output_path}")
        # If the output file exists and has the same number of lines as the input file, skip
        if os.path.exists(output_path) and sum(1 for _ in open(output_path)) == sum(
            1 for _ in open(input_path)
        ):
            logger.info(
                f"Output file {output_path} already exists and has the same number of lines as the input file, skipping"
            )
            return
        with open(input_path, "r") as input_fp, open(output_path, "w") as output_fp:
            for line in tqdm(input_fp):
                data = json.loads(line)
                scores = OrderedDict(
                    [
                        ("watermarked_text", data["watermarked_text"]),
                        ("unwatermarked_text", data["unwatermarked_text"]),
                        ("chosen", data["chosen"]),
                        ("rejected", data["rejected"]),
                    ]
                )
                texts_to_score = [text for key, text in scores.items()]
                numerical_scores = self.get_reward_score(data["prompt"], texts_to_score)
                for idx, (key, text) in enumerate(scores.items()):
                    data[f"{key}_reward_score"] = float(numerical_scores[idx])

                json.dump(data, output_fp)
                output_fp.write("\n")
                output_fp.flush()


class RewardScorerRegistry:
    _scorers: dict[str, type[RewardScorerBase]] = {}

    @classmethod
    def register(cls, name):
        def decorator(scorer_class):
            cls._scorers[name] = scorer_class
            return scorer_class

        return decorator

    @classmethod
    def get(cls, name):
        return cls._scorers.get(name)


@RewardScorerRegistry.register("llm-blender/PairRM")
class BlenderRewardScorer(RewardScorerBase):
    def __init__(self):
        super().__init__()
        reward_model = "llm-blender/PairRM"
        logger.info(f"Loading reward model: {reward_model}")
        self.blender = llm_blender.Blender()
        self.blender.loadranker(reward_model)

    def get_reward_score(self, prompt: str, texts: list[str]) -> list[float]:
        # This is list of lists, because blender did not work with plain strings
        # or list of strings of size 1.
        return self.blender.rank([prompt], [texts], return_scores=True)[0]
