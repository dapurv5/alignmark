import json
import logging
import os
from abc import abstractmethod
from string import Template

import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cleanup_utils import (
    pick_first_k_blocks,
    prune_multiple_turns,
    remove_prompt_from_response,
    remove_role_tags,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


INSTRUCTION_TAGS = ["### Instruction:", "## Instruction:"]
RESPONSE_TAGS = ["### Response:", "## Response:"]


def cleanup(data: dict):
    # Remove the prompt from the generated text
    data = remove_prompt_from_response(data, "prompt", "watermarked_text")
    data = remove_prompt_from_response(data, "prompt", "unwatermarked_text")
    # Remove the role tags from the prompt
    data = remove_role_tags(data, INSTRUCTION_TAGS, "prompt")
    data = remove_role_tags(data, RESPONSE_TAGS, "prompt")
    # Prune multiple turns
    for role_tag in INSTRUCTION_TAGS + RESPONSE_TAGS:
        data = prune_multiple_turns(data, "watermarked_text", role_tag)
        data = prune_multiple_turns(data, "unwatermarked_text", role_tag)
    # Miscellaneous cleanup
    data = pick_first_k_blocks(data, "watermarked_text", 2)
    data = pick_first_k_blocks(data, "unwatermarked_text", 2)
    data["watermarked_text"] = data["watermarked_text"].replace("\n", " ")
    data["unwatermarked_text"] = data["unwatermarked_text"].replace("\n", " ")
    return data


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
        logger.info(
            f"Computing safety scores for {input_path} and writing to {output_path}"
        )
        # If the output file exists and has the same number of lines as the input file, skip
        if os.path.exists(output_path) and sum(1 for _ in open(output_path)) == sum(
            1 for _ in open(input_path)
        ):
            logger.info(
                f"Output file {output_path} already exists and has the same number of lines as the input file, skipping"
            )
            return
        with open(input_path, "r") as input_fp, open(output_path, "w") as output_fp:
            batch = self._initialize_batch()
            for idx, line in tqdm(enumerate(input_fp)):
                data = json.loads(line)
                cleanup(data)
                self._add_to_batch(batch, data)
                if len(batch["watermarked_texts"]) == self.batch_size:
                    self._process_batch(batch, output_fp)
                    batch = self._initialize_batch()
            # Handle the last batch
            if batch["watermarked_texts"]:
                self._process_batch(batch, output_fp)

    def _process_batch(self, batch, output_fp):
        wm_safety_evals = self.get_safety_score(
            batch["prompts"],
            batch["watermarked_texts"],
        )
        uwm_safety_evals = self.get_safety_score(
            batch["prompts"],
            batch["unwatermarked_texts"],
        )
        for data, wm_eval, uwm_eval in zip(
            batch["data"], wm_safety_evals, uwm_safety_evals
        ):
            data["watermarked_safety_eval"] = wm_eval["safety_eval"]
            data["unwatermarked_safety_eval"] = uwm_eval["safety_eval"]
            data["watermarked_unsafe_category"] = (
                wm_eval["unsafe_category"] if "unsafe_category" in wm_eval else ""
            )
            data["unwatermarked_unsafe_category"] = (
                uwm_eval["unsafe_category"] if "unsafe_category" in uwm_eval else ""
            )
            json.dump(data, output_fp)
            output_fp.write("\n")
        output_fp.flush()

    def _initialize_batch(self):
        return {
            "prompts": [],
            "watermarked_texts": [],
            "unwatermarked_texts": [],
            "data": [],
        }

    def _add_to_batch(self, batch, data):
        batch["prompts"].append(data["prompt"])
        batch["watermarked_texts"].append(data["watermarked_text"])
        batch["unwatermarked_texts"].append(data["unwatermarked_text"])
        batch["data"].append(data)


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
