import json
import os
import random
from typing import Any, List

import fire
import numpy as np
import torch
from datasets import load_dataset

from generate import WatermarkTextPairsGenerator


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
    **kwargs,
):
    generator = WatermarkTextPairsGenerator(
        model_name,
        watermark_name,
        output_path,
        threshold=threshold,
        batch_size=batch_size,
        **kwargs,
    )
    generator.generate(examples)


def main(
    exp_dir: str,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    dataset_name: str = "",
    dataset_subset_name: str = "",
    dataset_split: str = "test",
    watermark_name: str = "openai",  # openai, maryland, no_watermark
    limit_dataset_size: int = -1,
    seed: int = 42,
    dataset_path: str = None,
    **kwargs,
):
    seed_everything(seed)
    if dataset_name is not None and dataset_name != "":
        if dataset_subset_name:
            dataset = load_dataset(
                dataset_name, dataset_subset_name, trust_remote_code=True
            )
        else:
            dataset = load_dataset(dataset_name, trust_remote_code=True)
    elif dataset_path is not None:
        dataset = load_dataset("json", data_files=dataset_path)
        dataset_split = "train"
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    else:
        raise ValueError("Either dataset_name or dataset_path must be provided")

    if dataset_split in dataset:
        dataset = dataset[dataset_split]
    else:
        raise ValueError(f"Dataset split {dataset_split} not found")

    if limit_dataset_size > 0 and hasattr(dataset, "select"):
        dataset = dataset.select(range(limit_dataset_size))
    os.makedirs(exp_dir, exist_ok=True)

    def simple_name(name):
        return name.replace("_", "").split("/")[-1]  # name cannot contain underscores

    # Generate the output file name including temperature
    run_name = (
        "out_"
        f"{simple_name(dataset_name)}_"
        f"{simple_name(model_name)}_"
        f"{watermark_name}_"
        f"{seed}"
    )
    if "temperature" in kwargs:
        run_name += f"_temperature_{kwargs['temperature']}"
    for param_name in ["delta", "gamma", "ngram"]:
        if param_name in kwargs:
            run_name += f"_{param_name}_{kwargs[param_name]}"
    run_name += ".jsonl"
    output_path = os.path.join(exp_dir, run_name)

    # Run the generator
    run_generator(
        model_name=model_name,
        watermark_name=watermark_name,
        examples=dataset,
        output_path=output_path,
        **kwargs,
    )


if __name__ == "__main__":
    fire.Fire(main)
