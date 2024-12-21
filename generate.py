import json
import random
from abc import ABC, abstractmethod
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
        **kwargs,
    ):
        self.model_name = model_name
        self.watermark_name = watermark_name  # openai, maryland, no_watermark
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
        self.generator = self._initialize_wm_component(
            "generator", "no_watermark", **kwargs
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
        if component_name == "no_watermark":
            wm_kwargs.pop("delta", 2.0)
            wm_kwargs.pop("gamma", 0.5)

        component_classes = {
            "generator": {
                "no_watermark": "WmGenerator",
                "openai": "OpenaiGenerator",
                "maryland": "MarylandGenerator",
            },
            "detector": {
                "no_watermark": "WmDetector",
                "openai": "OpenaiDetectorZ",
                "maryland": "MarylandDetectorZ",
            },
        }

        if component_name not in component_classes[component_type]:
            raise ValueError(f"Invalid watermark name: {component_name}")

        module = __import__(
            "wm_generators" if component_type == "generator" else "wm_detectors"
        )
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
        # shuffle examples
        if isinstance(examples, list):
            random.shuffle(examples)
        elif hasattr(examples, "shuffle"):
            # For Dataset objects that have a shuffle method
            examples = examples.shuffle(seed=self.kwargs.get("seed", 42))
        else:
            print(
                "Warning: Unable to shuffle examples. Proceeding with original order."
            )
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
        scores = self.wm_detector.get_scores_by_t([text])
        pvalues = self.wm_detector.get_pvalues(scores)
        # Assuming we're interested in the first payload (index 0)
        pvalue = pvalues[0][0]
        is_watermarked = pvalue < self.threshold
        return {
            "is_watermarked": is_watermarked,
            "score": scores[0][0][0],
            "pvalue": pvalue,
        }

    @abstractmethod
    def prepare_output_rows(self, prompts: list[str], **gen_kwargs) -> list[dict]:
        raise NotImplementedError


class WatermarkTextPairsGenerator(BaseWatermarkGenerator):
    """
    Generate watermarked and unwatermarked text pairs for a given set of prompts.

    """

    def prepare_output_rows(self, prompts: list[str], **gen_kwargs) -> list[dict]:
        # Generate watermarked and unwatermarked texts
        watermarked_texts = self.wm_generator.generate(prompts, **gen_kwargs)
        unwatermarked_texts = self.generator.generate(prompts, **gen_kwargs)

        # Clean up special tokens
        watermarked_texts = [
            self.tokenizer.decode(self.tokenizer.encode(text), skip_special_tokens=True)
            for text in watermarked_texts
        ]
        unwatermarked_texts = [
            self.tokenizer.decode(self.tokenizer.encode(text), skip_special_tokens=True)
            for text in unwatermarked_texts
        ]

        # Run detection on both sets of texts
        watermarked_outs = [self.detect(text) for text in watermarked_texts]
        unwatermarked_outs = [self.detect(text) for text in unwatermarked_texts]

        # Build results list
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

        # Print batch metrics
        self._print_batch_metrics(prompts, watermarked_outs, unwatermarked_outs)
        return results

    def _print_batch_metrics(self, prompts, watermarked_outs, unwatermarked_outs):
        n = len(prompts)
        wm_correct = sum(1 for out in watermarked_outs if out["is_watermarked"])
        unwm_correct = sum(1 for out in unwatermarked_outs if not out["is_watermarked"])

        print(
            f"Batch FNR: {(n - wm_correct) / n:.4f}, "
            f"Batch FPR: {(n - unwm_correct) / n:.4f}"
        )


class WatermarkTextGenerator(BaseWatermarkGenerator):
    """
    Generate watermarked text for a given set of prompts.
    """

    def prepare_output_rows(self, prompts: list[str], **gen_kwargs) -> list[dict]:
        return []
