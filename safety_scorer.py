import json
import logging
import os
from abc import abstractmethod
from string import Template

import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cleanup_utils import cleanup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


INSTRUCTION_TAGS = ["### Instruction:", "## Instruction:"]
RESPONSE_TAGS = ["### Response:", "## Response:"]


# def cleanup(data: dict):
#     # Remove the prompt from the generated text
#     data = remove_prompt_from_response(data, "prompt", "watermarked_text")
#     data = remove_prompt_from_response(data, "prompt", "unwatermarked_text")
#     # Remove the role tags from the prompt
#     data = remove_role_tags(data, INSTRUCTION_TAGS, "prompt")
#     data = remove_role_tags(data, RESPONSE_TAGS, "prompt")
#     # Prune multiple turns
#     for role_tag in INSTRUCTION_TAGS + RESPONSE_TAGS:
#         data = prune_multiple_turns(data, "watermarked_text", role_tag)
#         data = prune_multiple_turns(data, "unwatermarked_text", role_tag)
#     # Miscellaneous cleanup
#     data = pick_first_k_blocks(data, "watermarked_text", 2)
#     data = pick_first_k_blocks(data, "unwatermarked_text", 2)
#     data["watermarked_text"] = data["watermarked_text"].replace("\n", " ")
#     data["unwatermarked_text"] = data["unwatermarked_text"].replace("\n", " ")
#     return data


class SafetyScorerBase:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

    @abstractmethod
    def get_safety_score(
        self,
        queries: list[str],
        responses: list[str],
    ) -> list[dict[str, str]]:
        raise NotImplementedError("Subclasses must implement this method")

    def compute_safety_scores(self, input_path: str, output_path: str):
        nr_lines_input, nr_lines_output = 0, 0
        logger.info(
            f"Computing safety scores for {input_path} and writing to {output_path}"
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
                cleanup(data, INSTRUCTION_TAGS + RESPONSE_TAGS, [], "prompt")
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
            safety_evals = self.get_safety_score(
                [batch["prompt_batch"][idx]] * len(field_batch),
                field_batch,
            )
            safety_eval_list = [
                safety_eval["safety_eval"] for safety_eval in safety_evals
            ]
            unsafe_category_list = [
                (
                    safety_eval["unsafe_category"]
                    if "unsafe_category" in safety_eval
                    else ""
                )
                for safety_eval in safety_evals
            ]
            batch["data_batch"][idx][f"{field}_safety_eval"] = safety_eval_list
            batch["data_batch"][idx][f"{field}_unsafe_category"] = unsafe_category_list

    def _process_single_text(self, batch, field):
        safety_evals = self.get_safety_score(
            batch["prompt_batch"],
            batch[f"{field}_batch"],
        )
        for data, safety_eval in zip(batch["data_batch"], safety_evals):
            data[f"{field}_safety_eval"] = safety_eval["safety_eval"]
            data[f"{field}_unsafe_category"] = (
                safety_eval["unsafe_category"]
                if "unsafe_category" in safety_eval
                else ""
            )

    def _write_batch_results(self, batch, output_fp):
        for data in batch["data_batch"]:
            json.dump(data, output_fp)
            output_fp.write("\n")
        output_fp.flush()

    def _initialize_batch(self):
        return {
            "prompt_batch": [],
            "watermarked_text_batch": [],
            "unwatermarked_text_batch": [],
            "watermarked_texts_batch": [],
            "unwatermarked_texts_batch": [],
            "data_batch": [],
        }

    def _add_to_batch(self, batch, data):
        batch["prompt_batch"].append(data["prompt"])
        batch["watermarked_text_batch"].append(data["watermarked_text"])
        batch["unwatermarked_text_batch"].append(data["unwatermarked_text"])
        batch["watermarked_texts_batch"].append(data["watermarked_texts"])
        batch["unwatermarked_texts_batch"].append(data["unwatermarked_texts"])
        batch["data_batch"].append(data)


class SafetyScorerRegistry:
    _scorers: dict[str, type[SafetyScorerBase]] = {}

    @classmethod
    def register(cls, name):
        def decorator(scorer_class):
            cls._scorers[name] = scorer_class
            return scorer_class

        return decorator

    @classmethod
    def get(cls, name):
        return cls._scorers.get(name)


@SafetyScorerRegistry.register("llama-guard")
class LlamaGuardSafetyScorer(SafetyScorerBase):
    """
    LlamaGuard based safety scorer.
    """

    LLAMA_GUARD_MODEL = "meta-llama/Llama-Guard-3-8B"

    def __init__(self, batch_size: int = 64):
        super().__init__(batch_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(self.LLAMA_GUARD_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.LLAMA_GUARD_MODEL, torch_dtype=self.dtype, device_map=self.device
        )
        self.model.eval()
        logger.info("Initialized LlamaGuard scorer")

    def get_safety_score(
        self,
        queries: list[str],
        responses: list[str],
    ) -> list[dict[str, str]]:
        safety_evals = []
        for result in self.moderate(queries, responses):
            safety_evals.append(result)
        return safety_evals

    def moderate(
        self, queries: list[str], responses: list[str]
    ) -> list[dict[str, str]]:
        with open("prompt_templates/safety_eval.txt", "r") as file:
            template = Template(file.read())
        prompts = [
            template.substitute(query=q, response=r) for q, r in zip(queries, responses)
        ]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(
            self.device
        )
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        prompt_lens = inputs.input_ids.shape[-1]
        decoded = []
        for output in outputs:
            text = self.tokenizer.decode(output[prompt_lens:], skip_special_tokens=True)
            text = text.strip()
            # Only keep the first two lines of the response
            text = "\n".join(text.split("\n")[:2])
            arr = text.split("\n")
            decoded.append(
                {
                    "safety_eval": arr[0].strip().lower(),
                    "unsafe_category": arr[1].strip() if len(arr) > 1 else "",
                }
            )
        return decoded


@SafetyScorerRegistry.register("openai")
class OpenaiSafetyScorer(SafetyScorerBase):
    def __init__(self, model_name: str = "gpt-4o-mini", batch_size: int = 64):
        super().__init__(batch_size)
        self.model = OpenAI()
        self.model_name = model_name
        logger.info(f"Initialized OpenAI scorer with model {model_name}")

    def get_safety_score(
        self,
        queries: list[str],
        responses: list[str],
    ) -> list[dict[str, str]]:
        scores = []
        # Load the template
        with open("prompt_templates/safety_eval.txt", "r") as file:
            template = Template(file.read())
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import tenacity

        @tenacity.retry(stop=tenacity.stop_after_attempt(2))
        def process_single_evaluation(args_):
            query, response = args_
            prompt = template.substitute(query=query, response=response)
            response = self.model.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            text = response.choices[0].message.content
            text = "\n".join(text.split("\n")[:2])
            arr = text.split("\n")
            return {
                "safety_eval": arr[0].strip().lower(),
                "unsafe_category": arr[1].strip() if len(arr) > 1 else "",
            }

        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = {
                executor.submit(process_single_evaluation, args): idx
                for idx, args in enumerate(zip(queries, responses))
            }
            results = [None] * len(queries)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
            scores.extend(results)
        return scores
