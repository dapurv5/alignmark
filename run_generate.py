import json
import os
from typing import Any, List

import fire
from datasets import load_dataset
from generate import WatermarkTextPairsGenerator


def run_generator(
    model_name: str,
    watermark_name: str,
    examples: List[Any],
    output_path: str,
    watermark_algorithm_config: str,
    **generate_kwargs,
):
    generator = WatermarkTextPairsGenerator(
        model_name,
        watermark_name,
        output_path,
        watermark_algorithm_config,
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
    watermark_algorithm_config_key: str = "delta",
    watermark_algorithm_config_val: float = 2.0,
    limit_dataset_size: int = -1,
):
    dataset = load_dataset(dataset_name, trust_remote_code=True)
    dataset = dataset["test"] if "test" in dataset else dataset
    if limit_dataset_size > 0:
        dataset = dataset.select(range(limit_dataset_size))
    os.makedirs(exp_dir, exist_ok=True)

    generate_kwargs = {
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 200,
    }

    write_exp_config(exp_dir, dataset_name, model_name, watermark_name, generate_kwargs)

    def simple_name(name):
        return name.split("/")[-1]

    # Open the default config file, update the value of the config key,
    # and write to a new config file
    with open(
        watermark_algorithm_default_config_path, "r"
    ) as algorithm_default_config_fp:
        algorithm_config_blob = json.load(algorithm_default_config_fp)
    algorithm_config_blob[
        watermark_algorithm_config_key
    ] = watermark_algorithm_config_val
    algorithm_config_path = os.path.join(
        exp_dir,
        f"{watermark_name}_{watermark_algorithm_config_key.replace('_', '')}_{algorithm_config_blob[watermark_algorithm_config_key]}.json",
    )
    with open(algorithm_config_path, "w") as algorithm_config_fp:
        json.dump(algorithm_config_blob, algorithm_config_fp)

    # Generate the output file name
    run_name = (
        "out_"
        f"{simple_name(dataset_name)}_"
        f"{simple_name(model_name)}_"
        f"{watermark_name}_{watermark_algorithm_config_key.replace('_', '')}_"
        f"{watermark_algorithm_config_val}.jsonl"
    )
    output_path = os.path.join(exp_dir, run_name)
    # Run the generator
    run_generator(
        model_name=model_name,
        watermark_name=watermark_name,
        examples=dataset,
        output_path=output_path,
        watermark_algorithm_config=algorithm_config_path,
        **generate_kwargs,
    )


def write_exp_config(
    exp_dir, dataset_name, model_name, watermark_name, generate_kwargs
):
    exp_config = {}
    exp_config_path = os.path.join(exp_dir, "my_exp_config.json")
    exp_config["generate_kwargs"] = generate_kwargs
    exp_config["dataset_name"] = dataset_name
    exp_config["model_name"] = model_name
    exp_config["watermark_name"] = watermark_name
    # Write exp_config to my_exp_config_path
    # This is for experiment tracking ...
    with open(exp_config_path, "w") as exp_config_fp:
        json.dump(exp_config, exp_config_fp)


if __name__ == "__main__":
    fire.Fire(main)
