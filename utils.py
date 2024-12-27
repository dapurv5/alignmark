import os
import random

from datasets import load_dataset


def prune_dataset(dataset, limit_dataset_size, dataset_start_row, dataset_end_row):
    if limit_dataset_size > 0 and hasattr(dataset, "select"):
        dataset = dataset.select(range(limit_dataset_size))
    if dataset_start_row >= 0 and dataset_end_row >= 0:
        dataset = dataset.select(range(dataset_start_row, dataset_end_row + 1))
    return dataset


def prepare_dataset(dataset_name, dataset_subset_name, dataset_split, dataset_path):
    dataset_subset_name = dataset_subset_name.strip()
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
    return dataset, dataset_name


def shuffle_dataset(dataset, seed):
    if isinstance(dataset, list):
        random.shuffle(dataset)
    elif hasattr(dataset, "shuffle"):
        # For Dataset objects that have a shuffle method
        dataset = dataset.shuffle(seed=seed)
    else:
        print("Warning: Unable to shuffle examples. Proceeding with original order.")


def get_base_run_name(dataset_name, model_name, watermark_name, seed):
    def simple_name(name):
        return name.replace("_", "").split("/")[-1]  # name cannot contain underscores

    run_name = (
        "out_"
        f"{simple_name(dataset_name)}_"
        f"{simple_name(model_name)}_"
        f"{watermark_name}_"
        f"{seed}"
    )
    return run_name


def get_run_name(
    model_name,
    watermark_name,
    seed,
    dataset_name,
    dataset_start_row,
    dataset_end_row,
    total_dataset_size,
    **kwargs,
):

    # Generate the output file name including temperature
    run_name = get_base_run_name(dataset_name, model_name, watermark_name, seed)
    if dataset_start_row >= 0 and dataset_end_row >= 0:
        # format the dataset start and end row as a string with leading zeros
        # Have same number of digits
        num_digits = len(str(total_dataset_size))
        dataset_start_row_str = f"{dataset_start_row:0{num_digits}d}"
        dataset_end_row_str = f"{dataset_end_row:0{num_digits}d}"
        run_name = f"part_{dataset_start_row_str}-{dataset_end_row_str}_" + run_name
    if "temperature" in kwargs:
        run_name += f"_temperature_{kwargs['temperature']}"
    for param_name in ["delta", "gamma", "ngram"]:
        if param_name in kwargs:
            run_name += f"_{param_name}_{kwargs[param_name]}"
    return run_name


def get_device_to_use(
    num_processes: int,
    num_gpus_per_process: int,
) -> str:
    if num_gpus_per_process == 0 and num_processes == 1:
        return "cpu"
    elif num_gpus_per_process == 0 and num_processes > 1:
        raise ValueError(
            "num_gpus_per_process must be greater than 0 when num_processes is greater than 1"
        )
    elif num_gpus_per_process > 0 and num_processes == 1:
        return "cuda"
    elif num_gpus_per_process > 0 and num_processes > 1:
        return "cuda"
    else:
        raise ValueError("Invalid input parameters")
