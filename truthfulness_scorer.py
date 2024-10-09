import json
import logging
from abc import abstractmethod

import torch
from bleurt_pytorch import (
    BleurtConfig,
    BleurtForSequenceClassification,
    BleurtTokenizer,
)
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TruthfulnessScorerBase:
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size

    @abstractmethod
    def get_truthfulness_score(self, prompt: str, texts: list[str]) -> list[float]:
        raise NotImplementedError("Subclasses must implement this method")

    def compute_truthfulness_scores(self, input_path: str, output_path: str):
        logger.info(
            f"Computing truthfulness scores for {input_path} and writing to {output_path}"
        )
        with open(input_path, "r") as input_fp, open(output_path, "w") as output_fp:
            batch = self._initialize_batch()
            for line in tqdm(input_fp):
                data = json.loads(line)
                self._add_to_batch(batch, data)
                if len(batch["watermarked_texts"]) == self.batch_size:
                    self._process_batch(batch, output_fp)
                    batch = self._initialize_batch()
            # Handle the last batch
            if batch["watermarked_texts"]:
                self._process_batch(batch, output_fp)

    def _process_batch(self, batch, output_fp):
        wm_scores = self.get_truthfulness_score(
            batch["watermarked_texts"],
            batch["true_ref_answers"],
            batch["false_ref_answers"],
        )
        uwm_scores = self.get_truthfulness_score(
            batch["unwatermarked_texts"],
            batch["true_ref_answers"],
            batch["false_ref_answers"],
        )
        for data, wm_score, uwm_score in zip(batch["data"], wm_scores, uwm_scores):
            data["watermarked_truthfulness_score"] = float(wm_score)
            data["unwatermarked_truthfulness_score"] = float(uwm_score)
            json.dump(data, output_fp)
            output_fp.write("\n")
        output_fp.flush()

    def _initialize_batch(self):
        return {
            "watermarked_texts": [],
            "unwatermarked_texts": [],
            "true_ref_answers": [],
            "false_ref_answers": [],
            "data": [],
        }

    def _add_to_batch(self, batch, data):
        batch["watermarked_texts"].append(data["watermarked_text"])
        batch["unwatermarked_texts"].append(data["unwatermarked_text"])
        batch["true_ref_answers"].append(data["correct_answers"])
        batch["false_ref_answers"].append(data["incorrect_answers"])
        batch["data"].append(data)


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
    """
    BLEURT based truthfulness scorer.

    For every example there is a list of true and false references.
    For each list we compute the BLEURT score of the watermarked and
    unwatermarked text. The truthfulness score is the difference between
    the highest BLEURT score of the watermarked text and the highest BLEURT
    score of the unwatermarked text.
    """

    def __init__(self, batch_size: int = 64):
        super().__init__(batch_size)
        self.config = BleurtConfig.from_pretrained("lucadiliello/BLEURT-20-D12")
        self.model = BleurtForSequenceClassification.from_pretrained(
            "lucadiliello/BLEURT-20-D12"
        )
        self.tokenizer = BleurtTokenizer.from_pretrained("lucadiliello/BLEURT-20-D12")
        self.model.eval()
        logger.info("Initialized BLEURT scorer")

    def get_truthfulness_score(
        self,
        batch_texts: list[str],
        batch_true_ref_answers: list[list[str]],
        batch_false_ref_answers: list[list[str]],
    ) -> list[float]:
        scores = []
        for idx, (true_ref_answers, false_ref_answers) in enumerate(
            zip(batch_true_ref_answers, batch_false_ref_answers)
        ):
            text = batch_texts[idx]
            candidates_pos = [text] * len(true_ref_answers)
            candidates_neg = [text] * len(false_ref_answers)
            scores_pos = self.get_bleurt_score(true_ref_answers, candidates_pos)
            scores_neg = self.get_bleurt_score(false_ref_answers, candidates_neg)
            scores.append(max(scores_pos) - max(scores_neg))
        return scores

    def get_bleurt_score(
        self, references: list[str], candidates: list[str]
    ) -> list[float]:
        with torch.no_grad():
            inputs = self.tokenizer(
                references, candidates, padding="longest", return_tensors="pt"
            )
            res = self.model(**inputs).logits.flatten().tolist()
        return res
