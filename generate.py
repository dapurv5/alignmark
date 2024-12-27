import json
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class BaseWatermarkGenerator(ABC):
    def __init__(
        self,
        model_name: str,
        watermark_name: str,
        output_path: str,
        threshold: float = 0.05,
        batch_size: int = 16,
        text_field: str = "prompt",
        format_prompt_as_instructions: bool = False,
        num_wm_generations_per_prompt: int = 1,
        num_unwm_generations_per_prompt: int = 1,
        beam_size: int = 1,
        **kwargs,
    ):
        self.model_name = model_name
        self.watermark_name = watermark_name  # openai, maryland, unwatermarked
        self.output_path = output_path
        self.threshold = threshold
        self.batch_size = batch_size
        self.text_field = text_field
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Set pad_token if it's not defined
        self.llm = self._initialize_llm()
        if self.tokenizer.pad_token is None:
            print("Setting pad_token to eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm.config.pad_token_id = self.llm.config.eos_token_id
        self.vocab_size = self._infer_vocab_size(self.llm, self.tokenizer)
        self.num_wm_generations_per_prompt = num_wm_generations_per_prompt
        self.num_unwm_generations_per_prompt = num_unwm_generations_per_prompt
        self.beam_size = beam_size
        self.beamed_generation = (
            self.num_wm_generations_per_prompt > 1
            or self.num_unwm_generations_per_prompt > 1
        )
        # Initializations should be done after beamed_generation is set
        self.generator = self._initialize_wm_component(
            "generator", "unwatermarked", **kwargs
        )
        self.wm_generator = self._initialize_wm_component(
            "generator", watermark_name, **kwargs
        )
        self.wm_detector = self._initialize_wm_component(
            "detector", watermark_name, **kwargs
        )
        self.format_prompt_as_instructions = format_prompt_as_instructions
        self.kwargs = kwargs

    def _infer_vocab_size(self, model, tokenizer):
        text = "Hello, how are you?"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model(**inputs)
        return outputs.logits.shape[-1]

    def _initialize_llm(self):
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )

    def _initialize_wm_generator(self, component_name, **kwargs):
        return self._initialize_wm_component("generator", component_name, **kwargs)

    def _initialize_wm_detector(self, component_name, **kwargs):
        return self._initialize_wm_component("detector", component_name, **kwargs)

    def _initialize_wm_component(self, component_type, component_name, **kwargs):
        wm_kwargs = {
            k: kwargs[k]
            for k in [
                "ngram",
                "seed",
                "seeding",
                "salt_key",
                "payload",
                "delta",
                "gamma",
            ]
            if k in kwargs
        }
        if component_name == "unwatermarked":
            wm_kwargs.pop("delta", 2.0)
            wm_kwargs.pop("gamma", 0.5)

        if self.beamed_generation:
            generator = {
                "unwatermarked": "WmGeneratorBeam",
                "openai": "OpenaiGeneratorBeam",
                "maryland": "MarylandGeneratorBeam",
                "vllm-unwatermarked": "VLLMGeneratorBeam",
            }
        else:
            generator = {
                "unwatermarked": "WmGenerator",
                "openai": "OpenaiGenerator",
                "maryland": "MarylandGenerator",
                "vllm-unwatermarked": "VLLMGenerator",
            }

        component_classes = {
            "generator": generator,
            "detector": {
                "unwatermarked": "WmDetector",
                "openai": "OpenaiDetectorZ",
                "maryland": "MarylandDetectorZ",
                "vllm-unwatermarked": "WmDetector",
            },
        }

        if component_name not in component_classes[component_type]:
            raise ValueError(f"Invalid watermark name: {component_name}")
        module = None
        if component_type == "generator":
            module = __import__(
                "wm_generators_beam" if self.beamed_generation else "wm_generators"
            )
        elif component_type == "detector":
            module = __import__("wm_detectors")

        ComponentClass = getattr(
            module, component_classes[component_type][component_name]
        )

        common_args = {"model": self.llm, "tokenizer": self.tokenizer, **wm_kwargs}

        if component_type == "detector":
            common_args.pop("model")
            common_args["vocab_size"] = self.vocab_size

        return ComponentClass(**common_args)

    def format_prompt(self, prompt: str):
        if self.format_prompt_as_instructions:
            if ("### Instruction:" in prompt and "### Response:" in prompt) or (
                "Human:" in prompt and "Assistant:" in prompt
            ):
                return prompt
            else:
                # Add the instruction and response tags
                return f"### Instruction:\n{prompt}\n### Response:\n"
        else:
            return prompt

    def generate(self, examples: Any, **gen_kwargs_override):
        gen_kwargs = {
            k: self.kwargs[k]
            for k in ["max_gen_len", "top_p", "temperature"]
            if k in self.kwargs
        }
        gen_kwargs.update(gen_kwargs_override)

        with open(self.output_path, "w") as results_fp:
            for i in tqdm(range(0, len(examples), self.batch_size)):
                batch = []
                # This is needed because slicing with `examples[i : i + self.batch_size]`
                # will result in a dict object if examples is of type
                # datasets.arrow_dataset.Dataset
                for j in range(i, min(i + self.batch_size, len(examples))):
                    batch.append(examples[j])
                # Format the prompts to take care of model-specific formatting

                if self.format_prompt_as_instructions:
                    prompts = [
                        self.format_prompt(example[self.text_field])
                        for example in batch
                    ]
                    for idx, prompt in enumerate(prompts):
                        batch[idx][self.text_field] = prompt
                else:
                    prompts = [example[self.text_field] for example in batch]

                results = self.prepare_output_rows(prompts, **gen_kwargs)
                for idx, row in enumerate(results):
                    row = row | batch[idx]
                    results_fp.write(
                        json.dumps(row, default=self._json_serializer) + "\n"
                    )
                results_fp.flush()

    def _json_serializer(self, obj):
        """Custom JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def detect(self, text: str):
        scores_no_aggreg = self.wm_detector.get_scores_by_t([text])
        pvalues = self.wm_detector.get_pvalues(scores_no_aggreg)
        scores = self.wm_detector.aggregate_scores(scores_no_aggreg)
        # Assuming we're interested in the first payload (index 0)
        if len(pvalues) == 0 or len(pvalues[0]) == 0:
            print(f"No pvalues found for text: {text} because it's too short")
            return {
                "is_watermarked": False,
                "score": 0.0,
                "pvalue": 1.0,
            }
        pvalue = pvalues[0][0]
        score = scores[0][0]
        is_watermarked = pvalue < self.threshold
        return {
            "is_watermarked": is_watermarked,
            "score": score,
            "pvalue": pvalue,
        }

    @abstractmethod
    def prepare_output_rows(self, prompts: list[str], **gen_kwargs) -> list[dict]:
        raise NotImplementedError


class WatermarkTextPairsGenerator(BaseWatermarkGenerator):
    """
    Generate watermarked and unwatermarked text pairs for a given set of prompts.

    """

    def clean_text(self, text: str):
        return self.tokenizer.decode(
            self.tokenizer.encode(text), skip_special_tokens=True
        )

    def prepare_output_rows(self, prompts: list[str], **gen_kwargs) -> list[dict]:
        # Generate texts using beam search or greedy decoding
        if self.beamed_generation:
            unwatermarked_texts: list[list[str]] = self.generator.generate(
                prompts,
                **gen_kwargs,
                num_beams=self.beam_size,
                num_return_sequences=self.num_unwm_generations_per_prompt,
            )
            watermarked_texts: list[list[str]] = self.wm_generator.generate(
                prompts,
                **gen_kwargs,
                num_beams=self.beam_size,
                num_return_sequences=self.num_wm_generations_per_prompt,
            )
            return self._prepare_beamed_outputs(
                prompts, watermarked_texts, unwatermarked_texts
            )
        else:
            watermarked_texts: list[str] = self.wm_generator.generate(
                prompts, **gen_kwargs
            )
            unwatermarked_texts: list[str] = self.generator.generate(
                prompts, **gen_kwargs
            )
            return self._prepare_greedy_outputs(
                prompts, watermarked_texts, unwatermarked_texts
            )

    def _prepare_beamed_outputs(
        self,
        prompts: list[str],
        watermarked_texts: list[list[str]],
        unwatermarked_texts: list[list[str]],
    ):
        # Clean texts and get watermark detection results
        watermarked_texts = [
            [self.clean_text(text) for text in completions]
            for completions in watermarked_texts
        ]
        unwatermarked_texts = [
            [self.clean_text(text) for text in completions]
            for completions in unwatermarked_texts
        ]

        watermarked_outs = [
            [self.detect(text) for text in completions]
            for completions in watermarked_texts
        ]
        unwatermarked_outs = [
            [self.detect(text) for text in completions]
            for completions in unwatermarked_texts
        ]

        results = []
        for i in range(len(prompts)):
            # Store all generated texts and their metrics
            row = {
                "watermarked_texts": watermarked_texts[i],
                "unwatermarked_texts": unwatermarked_texts[i],
                "watermarked_texts.is_watermarked": [
                    bool(out["is_watermarked"]) for out in watermarked_outs[i]
                ],
                "unwatermarked_texts.is_watermarked": [
                    bool(out["is_watermarked"]) for out in unwatermarked_outs[i]
                ],
                "watermarked_texts.score": [
                    float(out["score"]) for out in watermarked_outs[i]
                ],
                "unwatermarked_texts.score": [
                    float(out["score"]) for out in unwatermarked_outs[i]
                ],
                "watermarked_texts.pvalue": [
                    float(out["pvalue"]) for out in watermarked_outs[i]
                ],
                "unwatermarked_texts.pvalue": [
                    float(out["pvalue"]) for out in unwatermarked_outs[i]
                ],
            }

            # Randomly select one text from each beam
            rand_idx_wm = random.randint(0, self.num_wm_generations_per_prompt - 1)
            rand_idx_unwm = random.randint(0, self.num_unwm_generations_per_prompt - 1)

            # Add selected texts and their metrics
            row["watermarked_text"] = watermarked_texts[i][rand_idx_wm]
            row["unwatermarked_text"] = unwatermarked_texts[i][rand_idx_unwm]
            row["watermarked_text.is_watermarked"] = row[
                "watermarked_texts.is_watermarked"
            ][rand_idx_wm]
            row["unwatermarked_text.is_watermarked"] = row[
                "unwatermarked_texts.is_watermarked"
            ][rand_idx_unwm]
            row["watermarked_text.score"] = row["watermarked_texts.score"][rand_idx_wm]
            row["unwatermarked_text.score"] = row["unwatermarked_texts.score"][
                rand_idx_unwm
            ]
            row["watermarked_text.pvalue"] = row["watermarked_texts.pvalue"][
                rand_idx_wm
            ]
            row["unwatermarked_text.pvalue"] = row["unwatermarked_texts.pvalue"][
                rand_idx_unwm
            ]

            results.append(row)

        # Flatten the watermarked_outs and unwatermarked_outs to compute batch metrics
        watermarked_outs_ = []
        unwatermarked_outs_ = []
        for out in watermarked_outs:
            watermarked_outs_.extend(out)
        for out in unwatermarked_outs:
            unwatermarked_outs_.extend(out)
        self._print_batch_metrics(watermarked_outs_, unwatermarked_outs_)
        return results

    def _prepare_greedy_outputs(
        self,
        prompts: list[str],
        watermarked_texts: list[str],
        unwatermarked_texts: list[str],
    ):
        # Clean texts
        watermarked_texts = [self.clean_text(text) for text in watermarked_texts]
        unwatermarked_texts = [self.clean_text(text) for text in unwatermarked_texts]

        # Get watermark detection results
        watermarked_outs = [self.detect(text) for text in watermarked_texts]
        unwatermarked_outs = [self.detect(text) for text in unwatermarked_texts]

        results = []
        for i in range(len(prompts)):
            results.append(
                {
                    "watermarked_text": watermarked_texts[i],
                    "unwatermarked_text": unwatermarked_texts[i],
                    "watermarked_text.is_watermarked": bool(
                        watermarked_outs[i]["is_watermarked"]
                    ),
                    "unwatermarked_text.is_watermarked": bool(
                        unwatermarked_outs[i]["is_watermarked"]
                    ),
                    "watermarked_score": float(watermarked_outs[i]["score"]),
                    "unwatermarked_score": float(unwatermarked_outs[i]["score"]),
                    "watermarked_pvalue": float(watermarked_outs[i]["pvalue"]),
                    "unwatermarked_pvalue": float(unwatermarked_outs[i]["pvalue"]),
                }
            )

        self._print_batch_metrics(watermarked_outs, unwatermarked_outs)
        return results

    def _print_batch_metrics(self, watermarked_outs, unwatermarked_outs):
        wm_correct = sum(1 for out in watermarked_outs if out["is_watermarked"])
        unwm_correct = sum(1 for out in unwatermarked_outs if not out["is_watermarked"])

        print(
            f"Batch FNR: {(len(watermarked_outs) - wm_correct) / len(watermarked_outs):.4f}, "
            f"Batch FPR: {(len(unwatermarked_outs) - unwm_correct) / len(unwatermarked_outs):.4f}"
        )


class WatermarkTextGenerator(BaseWatermarkGenerator):
    """
    Generate watermarked text for a given set of prompts.
    """

    def get_text_key(self):
        if (
            self.watermark_name == "vllm-unwatermarked"
            or self.watermark_name == "unwatermarked"
        ):
            return "unwatermarked_text"
        elif self.watermark_name == "openai" or self.watermark_name == "maryland":
            return "watermarked_text"
        else:
            raise ValueError(f"Invalid watermark name: {self.watermark_name}")

    def prepare_output_rows(self, prompts: list[str], **gen_kwargs) -> list[dict]:
        text_key = self.get_text_key()

        results = []
        for _ in range(self.num_generations_per_prompt):
            completions = self.wm_generator.generate(prompts, **gen_kwargs)
            for prompt_idx, completion in enumerate(completions):
                if len(results) <= prompt_idx:
                    results.append({text_key: []})
                results[prompt_idx][text_key].append(completion)
        return results
