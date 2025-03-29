import json
import logging
import os
from abc import abstractmethod
from collections import OrderedDict

import llm_blender
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from cleanup_utils import cleanup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROLE_TAGS = ["\n\nHuman:", "\n\nAssistant:"]
REMOVE_TOKENS = ["&quot;", "&quot", "\n\n.\n\n"]


class ModelBasedScorerBase:
    def __init__(self, text_field: str, file_name_suffix: str, score_field_name: str):
        self.text_field = text_field
        self.file_name_suffix = file_name_suffix
        self.score_field_name = score_field_name

    @abstractmethod
    def get_score(self, prompt: str, texts: list[str]) -> list[float]:
        raise NotImplementedError("Subclasses must implement this method")

    def compute_scores(self, input_path: str, output_path: str):
        """Compute scores for texts in input file and write results to output file.

        Args:
            input_path: Path to input JSONL file containing texts to score
            output_path: Path to write output JSONL file with scores
        """
        logger.info(f"Computing scores for {input_path} and writing to {output_path}")

        if self._output_file_is_complete(input_path, output_path):
            logger.info(f"Output file {output_path} is complete, skipping")
            return

        with open(input_path, "r") as input_fp, open(output_path, "w") as output_fp:
            for line in tqdm(input_fp):
                data = self._process_single_line(line)
                json.dump(data, output_fp)
                output_fp.write("\n")
                output_fp.flush()

    def _output_file_is_complete(self, input_path: str, output_path: str) -> bool:
        """Check if output file exists and has same number of lines as input."""
        if not os.path.exists(output_path):
            return False
        return sum(1 for _ in open(output_path)) == sum(1 for _ in open(input_path))

    def _process_single_line(self, line: str) -> dict:
        """Process a single line from input file and compute reward scores."""
        data = json.loads(line)
        data = cleanup(data, ROLE_TAGS, REMOVE_TOKENS, self.text_field)

        # Collect all texts that need scoring
        texts_to_score_map = self._collect_texts_to_score(data)

        # Get scores for all texts at once
        numerical_scores = self.get_score(
            data[self.text_field], list(texts_to_score_map.keys())
        )

        # Map scores back to texts
        for idx, text in enumerate(texts_to_score_map.keys()):
            texts_to_score_map[text] = numerical_scores[idx]

        # Add scores back to data dictionary
        data = self._score_single_texts(
            data,
            ["watermarked_text", "unwatermarked_text", "chosen", "rejected"],
            texts_to_score_map,
        )
        data = self._score_text_lists(
            data, ["watermarked_texts", "unwatermarked_texts"], texts_to_score_map
        )

        return data

    def _collect_texts_to_score(self, data: dict) -> OrderedDict:
        """Collect all texts that need scoring into an ordered dictionary."""
        texts_to_score_map = OrderedDict()

        # Single texts
        for key in ["watermarked_text", "unwatermarked_text", "chosen", "rejected"]:
            if key in data:
                texts_to_score_map[data[key]] = -1

        # Lists of texts
        for key in ["watermarked_texts", "unwatermarked_texts"]:
            if key in data:
                for text in data[key]:
                    texts_to_score_map[text] = -1

        return texts_to_score_map

    def _score_single_texts(
        self, data: dict, keys_to_score: list[str], texts_to_score_map: OrderedDict
    ) -> dict:
        """Add scores for individual text fields to data dictionary."""
        for key in keys_to_score:
            if key in data:
                data[f"{key}_{self.score_field_name}"] = float(
                    texts_to_score_map[data[key]]
                )
        return data

    def _score_text_lists(
        self, data: dict, keys_to_score: list[str], texts_to_score_map: OrderedDict
    ) -> dict:
        """Add scores for lists of texts to data dictionary."""
        for key in keys_to_score:
            if key in data:
                data[f"{key}_{self.score_field_name}"] = [
                    float(texts_to_score_map[text]) for text in data[key]
                ]
        return data


class ModelBasedScorerRegistry:
    _scorers: dict[str, type[ModelBasedScorerBase]] = {}

    @classmethod
    def register(cls, name):
        def decorator(scorer_class):
            cls._scorers[name] = scorer_class
            return scorer_class

        return decorator

    @classmethod
    def get(cls, name):
        return cls._scorers.get(name)


@ModelBasedScorerRegistry.register("llm-blender/PairRM")
class BlenderRewardScorer(ModelBasedScorerBase):
    def __init__(
        self,
        text_field: str,
        file_name_suffix: str = "_rewards",
        score_field_name: str = "reward_score",
        device: str = "cpu",
        gpu_ids: list[int] = [],
    ):
        super().__init__(text_field, file_name_suffix, score_field_name)
        reward_model = "llm-blender/PairRM"
        logger.info(f"Loading reward model: {reward_model}")
        self.blender = llm_blender.Blender()

        if gpu_ids:
            assert device == "cuda"
            gpu_id = gpu_ids[0]
            device = f"cuda:{gpu_id}"
        self.blender.loadranker(reward_model, device=device)

    def get_score(self, prompt: str, texts: list[str]) -> list[float]:
        # This is list of lists, because blender did not work with plain strings
        # or list of strings of size 1.
        return self.blender.rank([prompt], [texts], return_scores=True)[0]


@ModelBasedScorerRegistry.register("RLHFlow/ArmoRM-Llama3-8B-v0.1")
class ArmoRewardScorer(ModelBasedScorerBase):
    def __init__(
        self,
        text_field: str,
        file_name_suffix: str = "_rewards",
        score_field_name: str = "reward_score",
        device: str = "cpu",
        gpu_ids: list[int] = [],
    ):
        super().__init__(text_field, file_name_suffix, score_field_name)
        model_id = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
        if gpu_ids:
            assert device == "cuda", len(gpu_ids) == 1
            gpu_id = gpu_ids[0]
            device = f"cuda:{gpu_id}"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            max_length=2048,
            attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )

    def get_score(self, prompt: str, texts: list[str]) -> list[float]:
        res = []
        for text in texts:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": text},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.model.device)
            with torch.no_grad():
                output = self.model(input_ids)
                score = output.score.float().item()
            res.append(score)
        return res


@ModelBasedScorerRegistry.register("meta-llama/Llama-3.1-8B")
class PPLScorer(ModelBasedScorerBase):
    def __init__(
        self,
        text_field: str,
        file_name_suffix: str = "_ppl",
        score_field_name: str = "ppl_score",
        device: str = "cpu",
        gpu_ids: list[int] = [],
    ):
        super().__init__(text_field, file_name_suffix, score_field_name)

        # Initialize a language model for perplexity calculation
        model_name = "meta-llama/Llama-3.1-8B"
        if gpu_ids:
            assert device == "cuda"
            gpu_id = gpu_ids[0]
            device = f"cuda:{gpu_id}"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
            max_length=2048,
            attn_implementation="flash_attention_2",
        )

    def get_score(self, prompt: str, texts: list[str]) -> list[float]:
        """
        Compute perplexity scores for the given texts.
        Lower perplexity indicates better/more natural text.

        Args:
            prompt: The input prompt (not used for PPL calculation)
            texts: List of texts to score

        Returns:
            list[float]: Perplexity scores for each text
        """
        scores = []

        for text in texts:
            # Tokenize text
            encodings = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=2048
            ).to(self.device)

            # Calculate perplexity
            with torch.no_grad():
                outputs = self.model(**encodings, labels=encodings["input_ids"])
                loss = outputs.loss
                # Perplexity is e^loss
                ppl = torch.exp(loss).item()

                # Convert to negative since higher scores should be better
                # and lower perplexity means better text
                score = -ppl

            scores.append(score)

        return scores
