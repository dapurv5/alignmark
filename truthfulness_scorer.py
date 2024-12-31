import json
import logging
import os
from abc import abstractmethod

import nltk
import torch
from bleurt_pytorch import (
    BleurtConfig,
    BleurtForSequenceClassification,
    BleurtTokenizer,
)
from openai import OpenAI
from tqdm import tqdm

# from cleanup_utils import remove_prompt_from_response
from cleanup_utils import cleanup

# nltk.download("punkt")
# nltk.download("punkt_tab")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROLE_TAGS = ["\n\nHuman:", "\n\nAssistant:"]
REMOVE_TOKENS = [
    "&quot;",
    "&quot",
    "\n\n.\n\n",
    "# Answer\n",
    "### Response:\n\n",
    "# AI \n",
]


# def cleanup(data: dict):
#     # Remove the question from the generated text
#     data = remove_prompt_from_response(data, "question", "watermarked_text")
#     data = remove_prompt_from_response(data, "question", "unwatermarked_text")

#     def get_first_sentence(text):
#         ans = ""
#         if "\n" in text:
#             text_first_line = text.split("\n")[0]
#             if len(text) < 2:
#                 ans = text
#             else:
#                 ans = text_first_line
#         else:
#             sentences = nltk.sent_tokenize(text)
#             ans = sentences[0] if sentences else text
#         return ans

#     data["watermarked_text"] = get_first_sentence(data["watermarked_text"])
#     data["unwatermarked_text"] = get_first_sentence(data["unwatermarked_text"])


class TruthfulnessScorerBase:
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size

    @abstractmethod
    def get_truthfulness_score(
        self,
        questions: list[str],
        texts: list[str],
        true_ref_answers: list[list[str]],
        false_ref_answers: list[list[str]],
    ) -> list[float]:
        raise NotImplementedError("Subclasses must implement this method")

    def compute_truthfulness_scores(self, input_path: str, output_path: str):
        nr_lines_input, nr_lines_output = 0, 0
        logger.info(
            f"Computing truthfulness scores for {input_path} and writing to {output_path}"
        )
        # If the output file exists and has the same number of lines as the input file, skip
        if os.path.exists(output_path):
            nr_lines_input = sum(1 for _ in open(input_path))
            nr_lines_output = sum(1 for _ in open(output_path))
            if nr_lines_output == nr_lines_input:
                logger.info(
                    f"Output file {output_path} already exists and has the same number of lines as the input file, skipping"
                )
                return
        with open(input_path, "r") as input_fp, open(
            output_path, "w" if nr_lines_output == 0 else "a"
        ) as output_fp:
            batch = self._initialize_batch()
            for line_num, line in tqdm(enumerate(input_fp)):
                if line_num < nr_lines_output:
                    continue
                data = json.loads(line)
                cleanup(data, ROLE_TAGS, REMOVE_TOKENS, "question")
                self._add_to_batch(batch, data)
                if len(batch["watermarked_text_batch"]) == self.batch_size:
                    self._process_batch(batch, output_fp)
                    batch = self._initialize_batch()
            # Handle the last batch
            if batch["watermarked_text_batch"]:
                self._process_batch(batch, output_fp)

    def _process_batch(self, batch, output_fp):
        # Process each field type (single texts and text lists)
        self._process_text_fields(batch)

        # Write results to output file
        self._write_batch_results(batch, output_fp)

    def _process_text_fields(self, batch):
        fields = [
            "watermarked_text",
            "unwatermarked_text",
            "watermarked_texts",
            "unwatermarked_texts",
        ]

        for field in fields:
            if not (f"{field}_batch" in batch and batch[f"{field}_batch"]):
                continue

            if isinstance(batch[f"{field}_batch"][0], list):
                self._process_text_list(batch, field)
            else:
                self._process_single_text(batch, field)

    def _process_text_list(self, batch, field):
        for idx, field_batch in enumerate(batch[f"{field}_batch"]):
            scores = self.get_truthfulness_score(
                [batch["question_batch"][idx]] * len(field_batch),
                field_batch,
                [batch["correct_answers_batch"][idx]] * len(field_batch),
                [batch["incorrect_answers_batch"][idx]] * len(field_batch),
            )
            batch["data_batch"][idx][f"{field}_truthfulness_score"] = scores

    def _process_single_text(self, batch, field):
        scores = self.get_truthfulness_score(
            batch["question_batch"],
            batch[f"{field}_batch"],
            batch["correct_answers_batch"],
            batch["incorrect_answers_batch"],
        )
        for data, score in zip(batch["data_batch"], scores):
            data[f"{field}_truthfulness_score"] = float(score)

    def _write_batch_results(self, batch, output_fp):
        for data in batch["data_batch"]:
            json.dump(data, output_fp)
            output_fp.write("\n")
        output_fp.flush()

    def _initialize_batch(self):
        return {
            "question_batch": [],
            "watermarked_text_batch": [],
            "unwatermarked_text_batch": [],
            "correct_answers_batch": [],
            "incorrect_answers_batch": [],
            "data_batch": [],
            "watermarked_texts_batch": [],
            "unwatermarked_texts_batch": [],
        }

    def _add_to_batch(self, batch, data):
        fields = [
            "question",
            "watermarked_text",
            "unwatermarked_text",
            "correct_answers",
            "incorrect_answers",
            "watermarked_texts",
            "unwatermarked_texts",
        ]
        for field in fields:
            if field in data:
                batch[f"{field}_batch"].append(data[field])
        batch["data_batch"].append(data)


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

    BLEURT_MODEL = "lucadiliello/BLEURT-20-D12"

    def __init__(self, batch_size: int = 64):
        super().__init__(batch_size)
        self.config = BleurtConfig.from_pretrained(self.BLEURT_MODEL)
        self.model = BleurtForSequenceClassification.from_pretrained(self.BLEURT_MODEL)
        self.tokenizer = BleurtTokenizer.from_pretrained(self.BLEURT_MODEL)
        self.model.eval()
        logger.info("Initialized BLEURT scorer")

    def get_truthfulness_score(
        self,
        questions: list[str],
        texts: list[str],
        true_ref_answers: list[list[str]],
        false_ref_answers: list[list[str]],
    ) -> list[float]:
        scores = []
        for idx, (true_refs, false_refs) in enumerate(
            zip(true_ref_answers, false_ref_answers)
        ):
            text = texts[idx]
            candidates_pos = [text] * len(true_refs)
            candidates_neg = [text] * len(false_refs)
            scores_pos = self.get_bleurt_score(true_refs, candidates_pos)
            scores_neg = self.get_bleurt_score(false_refs, candidates_neg)
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


@TruthfulnessScorerRegistry.register("openai")
class OpenaiTruthfulnessScorer(TruthfulnessScorerBase):
    def __init__(self, model_name: str = "gpt-4o-mini", batch_size: int = 64):
        super().__init__(batch_size)
        self.model = OpenAI()
        self.model_name = model_name
        logger.info(f"Initialized OpenAI scorer with model {model_name}")

    def get_truthfulness_score(
        self,
        questions: list[str],
        texts: list[str],
        true_ref_answers: list[list[str]],
        false_ref_answers: list[list[str]],
    ) -> list[float]:
        scores = []
        # Load the template
        with open("prompt_templates/truthfulqa_eval.txt", "r") as file:
            template = file.read()
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import tenacity

        @tenacity.retry(stop=tenacity.stop_after_attempt(2))
        def process_single_evaluation(args):
            question, model_answer, true_refs, false_refs = args
            correct_answers_formatted = "\n".join(
                [f"- {answer}" for answer in true_refs]
            )
            incorrect_answers_formatted = "\n".join(
                [f"- {answer}" for answer in false_refs]
            )
            prompt = template.replace("{question}", question)
            prompt = prompt.replace("{model_answer}", model_answer)
            prompt = prompt.replace("{correct_answers}", correct_answers_formatted)
            prompt = prompt.replace("{incorrect_answers}", incorrect_answers_formatted)
            response = self.model.chat.completions.create(
                model=self.model_name,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = response.choices[0].message.content
            if "}" not in content:
                content += "}"
            try:
                gpt4_evaluation = json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode response: {content}")
                gpt4_evaluation = {}
            return gpt4_evaluation.get("truthfulness_score", 0)

        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = {
                executor.submit(process_single_evaluation, args): idx
                for idx, args in enumerate(
                    zip(questions, texts, true_ref_answers, false_ref_answers)
                )
            }
            results = [None] * len(questions)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
            scores.extend(results)
        return scores
