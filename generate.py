import json
import os
from typing import Any

import torch
import torch.distributed as dist

# from liger_kernel.transformers import AutoLigerKernelForCausalLM
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from utils.transformers_config import TransformersConfig
from vllm import LLM
from vllm_utils import patch_watermark
from watermark.auto_watermark import AutoWatermark


class WatermarkTextPairsGenerator:
    def __init__(
        self,
        model_name: str,
        watermark_name: str,
        output_path: str,
        watermark_algorithm_config: str,
        **generate_kwargs,
    ):
        self.model_name = model_name
        self.watermark_name = watermark_name
        self.output_path = output_path
        self.watermark_algorithm_config = watermark_algorithm_config

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.num_gpus = torch.cuda.device_count()
        self.llm = self._initialize_llm()
        self.watermark = self._initialize_watermark(**generate_kwargs)

    def _initialize_llm(self):
        if self.watermark_name in ["KGW"]:
            return LLM(
                model=self.model_name,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                gpu_memory_utilization=0.9,
                tensor_parallel_size=self.num_gpus,
            )
        else:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            return AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto",
            )

    def _initialize_watermark(self, **generate_kwargs):
        vocab = list(self.tokenizer.get_vocab().values())
        transformers_config = TransformersConfig(
            model=self.llm,
            tokenizer=self.tokenizer,
            vocab_size=len(vocab),
            device="cpu",
            max_new_tokens=generate_kwargs.get("max_tokens", 200),
            min_length=generate_kwargs.get("min_length", 200),
            do_sample=generate_kwargs.get("do_sample", True),
            temperature=generate_kwargs.get("temperature", 0.7),
            top_p=generate_kwargs.get("top_p", 0.95),
            no_repeat_ngram_size=4,
        )
        watermark = AutoWatermark.load(
            self.watermark_name,
            algorithm_config=self.watermark_algorithm_config,
            transformers_config=transformers_config,
        )
        if self.watermark_name in ["KGW"]:
            return patch_watermark(watermark, self.llm)
        else:
            watermark.config.device = "cuda" if torch.cuda.is_available() else "cpu"
            return watermark

    def generate(self, examples: Any):
        with open(os.path.join(self.output_path), "w") as results_fp:
            for example in tqdm(examples):
                prompt = example["prompt"]
                watermarked_text = self.watermark.generate_watermarked_text(prompt)
                unwatermarked_text = self.watermark.generate_unwatermarked_text(prompt)

                watermarked_out = self.watermark.detect_watermark(watermarked_text)
                unwatermarked_out = self.watermark.detect_watermark(unwatermarked_text)

                result = {
                    "watermarked_text.is_watermarked": bool(
                        watermarked_out["is_watermarked"]
                    ),
                    "unwatermarked_text.is_watermarked": bool(
                        unwatermarked_out["is_watermarked"]
                    ),
                    "watermarked_z_score": watermarked_out["score"],
                    "unwatermarked_z_score": unwatermarked_out["score"],
                    "watermarked_text": watermarked_text,
                    "unwatermarked_text": unwatermarked_text,
                } | example
                results_fp.write(json.dumps(result) + "\n")
                results_fp.flush()

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        del self.llm
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
