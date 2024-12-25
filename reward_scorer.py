import json
import logging
import os
from abc import abstractmethod
from collections import OrderedDict

import llm_blender
from tqdm import tqdm

from cleanup_utils import cleanup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROLE_TAGS = ["\n\nHuman:", "\n\nAssistant:"]
REMOVE_TOKENS = ["&quot;", "&quot"]


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
                data = cleanup(data, ROLE_TAGS, REMOVE_TOKENS, "prompt")

                # Score single texts
                single_text_keys = [
                    "watermarked_text",
                    "unwatermarked_text",
                    "chosen",
                    "rejected",
                ]
                data = self._score_single_texts(data, single_text_keys)

                # Score text lists
                list_text_keys = ["watermarked_texts", "unwatermarked_texts"]
                data = self._score_text_lists(data, list_text_keys)

                # Write results
                json.dump(data, output_fp)
                output_fp.write("\n")
                output_fp.flush()

    def _score_single_texts(self, data: dict, keys_to_score: list[str]) -> dict:
        """Score individual text fields and add scores to data"""
        scores = OrderedDict()
        for key in keys_to_score:
            if key in data:
                scores[key] = data[key]

        if scores:
            texts_to_score = [text for text in scores.values()]
            numerical_scores = self.get_reward_score(data["prompt"], texts_to_score)

            for idx, key in enumerate(scores):
                data[f"{key}_reward_score"] = float(numerical_scores[idx])

        return data

    def _score_text_lists(self, data: dict, keys_to_score: list[str]) -> dict:
        """Score lists of texts and add scores to data"""
        for key in keys_to_score:
            if key in data:
                numerical_scores = self.get_reward_score(data["prompt"], data[key])
                data[f"{key}_reward_score"] = [
                    float(score) for score in numerical_scores
                ]

        return data


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
    def __init__(self, device: str = "cpu"):
        super().__init__()
        reward_model = "llm-blender/PairRM"
        logger.info(f"Loading reward model: {reward_model}")
        self.blender = llm_blender.Blender()
        self.blender.loadranker(reward_model, device=device)

    def get_reward_score(self, prompt: str, texts: list[str]) -> list[float]:
        # This is list of lists, because blender did not work with plain strings
        # or list of strings of size 1.
        return self.blender.rank([prompt], [texts], return_scores=True)[0]
