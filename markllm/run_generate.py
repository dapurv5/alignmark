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
    watermark_algorithm_config: str,
    prompt_key: str = "prompt",
    **generate_kwargs,
):
    generator = WatermarkTextPairsGenerator(
        model_name,
        watermark_name,
        output_path,
        watermark_algorithm_config,
        prompt_key,
        **generate_kwargs,
    )
    generator.generate(examples)
    generator.cleanup()


def main(
    exp_dir: str,
    watermark_algorithm_default_config_path: str,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    dataset_name: str = "Dahoas/full-hh-rlhf",
    watermark_name: str = "KGW",
    watermark_algorithm_config_key: str = None,  # e.g. delta, can also be temp., top_p, etc.
    watermark_algorithm_config_val: float = None,  # e.g. 2.0
    limit_dataset_size: int = -1,
    seed: int = 42,
    prompt_key: str = "prompt",
    **additional_generate_kwargs,
):
    seed_everything(seed)
    dataset = load_dataset(dataset_name, trust_remote_code=True)
    dataset = dataset["test"] if "test" in dataset else dataset
    if limit_dataset_size > 0:
        dataset = dataset.select(range(limit_dataset_size))
    os.makedirs(exp_dir, exist_ok=True)

    # Extract temperature from additional_generate_kwargs, default to 0.2 if not provided
    temperature = additional_generate_kwargs.get("temperature", 0.2)

    generate_kwargs = {
        "temperature": temperature,
        **additional_generate_kwargs,
    }
    # If top_p, max_tokens are not provided
    if "top_p" not in generate_kwargs:
        generate_kwargs["top_p"] = 0.95
    if "max_tokens" not in generate_kwargs:
        generate_kwargs["max_tokens"] = 200

    write_exp_config(exp_dir, dataset_name, model_name, watermark_name)

    def simple_name(name):
        return name.split("/")[-1]

    # Only modify and write the config if both key and value are provided
    if (
        watermark_algorithm_config_key is not None
        and watermark_algorithm_config_val is not None
    ):
        with open(
            watermark_algorithm_default_config_path, "r"
        ) as algorithm_default_config_fp:
            algorithm_config_blob = json.load(algorithm_default_config_fp)
        algorithm_config_blob[watermark_algorithm_config_key] = (
            watermark_algorithm_config_val
        )
        algorithm_config_path = os.path.join(
            exp_dir,
            f"{watermark_name}_{watermark_algorithm_config_key.replace('_', '')}_{algorithm_config_blob[watermark_algorithm_config_key]}.json",
        )
        with open(algorithm_config_path, "w") as algorithm_config_fp:
            json.dump(algorithm_config_blob, algorithm_config_fp)
    else:
        algorithm_config_path = watermark_algorithm_default_config_path

    # Generate the output file name including temperature
    run_name = (
        "out_"
        f"{simple_name(dataset_name)}_"
        f"{simple_name(model_name)}_"
        f"{watermark_name}"
    )
    if "temperature" in additional_generate_kwargs:
        run_name += f"_temp_{additional_generate_kwargs['temperature']}"
    if (
        watermark_algorithm_config_key is not None
        and watermark_algorithm_config_val is not None
    ):
        run_name += f"_{watermark_algorithm_config_key.replace('_', '')}"
        run_name += f"_{watermark_algorithm_config_val}"
    run_name += ".jsonl"
    output_path = os.path.join(exp_dir, run_name)

    # Run the generator
    run_generator(
        model_name=model_name,
        watermark_name=watermark_name,
        examples=dataset,
        output_path=output_path,
        watermark_algorithm_config=algorithm_config_path,
        prompt_key=prompt_key,
        **generate_kwargs,
    )


def write_exp_config(exp_dir, dataset_name, model_name, watermark_name):
    exp_config = {}
    exp_config_path = os.path.join(exp_dir, "my_exp_config.json")
    exp_config["dataset_name"] = dataset_name
    exp_config["model_name"] = model_name
    exp_config["watermark_name"] = watermark_name
    # Write exp_config to my_exp_config_path
    # This is for experiment tracking ...
    with open(exp_config_path, "w") as exp_config_fp:
        json.dump(exp_config, exp_config_fp)


if __name__ == "__main__":
    fire.Fire(main)
