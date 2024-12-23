import json
import os
import random
from typing import Any, List

import fire
import numpy as np
import torch

from generate import WatermarkTextPairsGenerator
from utils import get_run_name, prepare_dataset, prune_dataset, shuffle_dataset


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_generator(
    model_name: str,
    watermark_name: str,
    examples: List[Any],
    output_path: str,
    threshold: float = 0.05,
    batch_size: int = 16,
    format_prompt_as_instructions: bool = False,
    num_wm_generations_per_prompt: int = 1,
    num_unwm_generations_per_prompt: int = 1,
    beam_size: int = 1,
    **kwargs,
):
    generator = WatermarkTextPairsGenerator(
        model_name,
        watermark_name,
        output_path,
        threshold=threshold,
        batch_size=batch_size,
        format_prompt_as_instructions=format_prompt_as_instructions,
        num_wm_generations_per_prompt=num_wm_generations_per_prompt,
        num_unwm_generations_per_prompt=num_unwm_generations_per_prompt,
        beam_size=beam_size,
        **kwargs,
    )
    generator.generate(examples)


def main(
    exp_dir: str,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    dataset_name: str = "",
    dataset_subset_name: str = "",
    dataset_split: str = "test",
    watermark_name: str = "openai",  # openai, maryland, unwatermarked, vllm-unwatermarked
    limit_dataset_size: int = -1,
    seed: int = 42,
    dataset_path: str = None,
    format_prompt_as_instructions: bool = False,
    num_wm_generations_per_prompt: int = 1,
    num_unwm_generations_per_prompt: int = 1,
    select_random_subset_from_dataset: bool = False,
    dataset_start_row: int = 0,
    dataset_end_row: int = -1,
    beam_size: int = 1,
    **kwargs,
):
    seed_everything(seed)
    dataset, dataset_name = prepare_dataset(
        dataset_name, dataset_subset_name, dataset_split, dataset_path
    )
    # shuffle examples (this should be done before pruning)
    if select_random_subset_from_dataset:
        shuffle_dataset(dataset, seed)
    total_dataset_size = len(dataset)
    dataset = prune_dataset(
        dataset, limit_dataset_size, dataset_start_row, dataset_end_row
    )
    os.makedirs(exp_dir, exist_ok=True)

    run_name = get_run_name(
        model_name,
        watermark_name,
        seed,
        dataset_name,
        dataset_start_row,
        dataset_end_row,
        total_dataset_size,
        **kwargs,
    )
    run_name += ".jsonl"
    output_path = os.path.join(exp_dir, run_name)
    kwargs["seed"] = seed  # This is so the wm generator and detector can use this.
    # this should be set after get_run_name

    # Run the generator
    run_generator(
        model_name=model_name,
        watermark_name=watermark_name,
        examples=dataset,
        output_path=output_path,
        format_prompt_as_instructions=format_prompt_as_instructions,
        num_wm_generations_per_prompt=num_wm_generations_per_prompt,
        num_unwm_generations_per_prompt=num_unwm_generations_per_prompt,
        beam_size=beam_size,
        **kwargs,
    )


if __name__ == "__main__":
    fire.Fire(main)
