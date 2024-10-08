import json
import logging
from abc import abstractmethod
from collections import OrderedDict

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TruthfulnessScorerBase:
    @abstractmethod
    def get_truthfulness_score(self, prompt: str, texts: list[str]) -> list[float]:
        raise NotImplementedError("Subclasses must implement this method")

    def compute_truthfulness_scores(self, input_path: str, output_path: str):
        logger.info(
            f"Computing truthfulness scores for {input_path} and writing to {output_path}"
        )
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
                numerical_scores = self.get_truthfulness_score(
                    data["prompt"], texts_to_score
                )
                for idx, (key, text) in enumerate(scores.items()):
                    data[f"{key}_truthfulness_score"] = float(numerical_scores[idx])

                json.dump(data, output_fp)
                output_fp.write("\n")
                output_fp.flush()


class TruthfulnessScorerRegistry:
    _scorers: dict[str, type[TruthfulnessScorerBase]] = {}

    @classmethod
    def register(cls, name):
        def decorator(scorer_class):
            cls._scorers[name] = scorer_class
            return scorer_class

        return decorator

    @classmethod
    def get(cls, name):
        return cls._scorers.get(name)


@TruthfulnessScorerRegistry.register("bleurt")
class BleurtTruthfulnessScorer(TruthfulnessScorerBase):
    def __init__(self):
        super().__init__()

    def get_truthfulness_score(
        self,
        text: list[str],
        true_ref_answers: list[list[str]],
        false_ref_answers: list[list[str]],
    ) -> list[float]:
        pass
