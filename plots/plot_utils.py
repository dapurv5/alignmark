import json
import os


def read_jsonl(file_path: str) -> list[dict]:
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def parse_filename(filename: str, rm_bon_suffix_from_wm_type: bool = False) -> dict:
    filename = filename.replace(".jsonl", "")
    filename_parts = filename.split("_")
    dataset_name = filename_parts[1]
    model_name = filename_parts[2]
    watermark_type = filename_parts[3]
    seed = filename_parts[4]
    start_idx = 5
    if seed in ["delta", "gamma", "ngram", "temperature"]:
        # The filename is old filename format and there is no seed
        seed = -1
        start_idx = 4
    params = {}
    idx = start_idx - 1
    for part in filename_parts[start_idx:]:
        idx += 1
        if part.startswith(("delta", "gamma", "ngram", "temperature")):
            key, value = part, filename_parts[idx + 1]
            params[key] = float(value) if key != "ngram" else int(value)
    if rm_bon_suffix_from_wm_type:
        assert "BoN" in watermark_type
        params["BoN"] = int(watermark_type.split("-BoN-")[1])
        watermark_type = watermark_type.split("-BoN-")[0]
    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "watermark_type": watermark_type,
        "seed": seed,
    } | params


def _get_scores(blobs: list[dict], prefix: str, score_name: str) -> list:
    """
    Get the scores from the blobs for the given prefix and score_name.

    Example:
    blobs = read_jsonl(file_path)
    watermarked_sc = get_scores(blobs, "watermarked", "rewards")
    unwatermarked_sc = get_scores(blobs, "unwatermarked", "rewards")
    """
    return [
        (
            blob[f"{prefix}_{score_name}_score"]
            if f"{prefix}_{score_name}_score" in blob
            else blob[f"{prefix}_text_{score_name}_score"]
        )
        for blob in blobs
    ]


def _sort_data_by_params_and_keys(data: dict) -> dict:
    """Sort the data dictionary by parameter values and keys.

    Args:
        data: Dictionary with (watermark_type, dataset_name) as keys and scores as values

    Returns:
        Sorted dictionary with sorted parameter values
    """
    # Sort the lists for each key, seed by param_name
    for key in data:
        for seed in data[key]:
            data[key][seed] = sorted(data[key][seed], key=lambda x: x[0])
    # Also sort the keys in the dictionary and make it ordered
    return dict(sorted(data.items()))


def _is_scores_file(filename: str, score_name: str) -> bool:
    # Needed for reward and rewards discrepancy
    # TODO: This is brittle, we should use regex to match the score_name
    return (
        filename.endswith(f"_{score_name}.jsonl")
        or filename.endswith(f"_{score_name}s.jsonl")
        or filename.endswith(f"_{score_name}_ppl.jsonl")
        or filename.endswith(f"_{score_name}s_ppl.jsonl")
    )


def group_files_by_model(files):
    files_dict = {}
    for file in files:
        filename = file.name
        parsed_info = parse_filename(filename)
        model_name = parsed_info["model_name"]
        if model_name not in files_dict:
            files_dict[model_name] = []
        files_dict[model_name].append(file)
    return files_dict.values()


def process_files(
    input_dir: str,
    model_name_to_plot: str,
    param_name_to_plot: str,
    score_name: str,
) -> dict[tuple[str, str], dict[str, list[tuple[float, list[float], list[float]]]]]:
    """
    Process all the files in the input directory and return a dictionary
    with (watermark_type, dataset_name) as keys and a list of tuples containing
    the param_name, watermarked scores, and unwatermarked scores.

    Args:
        input_dir (str): The directory containing the score files.
        model_name_to_plot (str): The name of the model to plot.
        param_name_to_plot (str): The name of the parameter which varies.
    Returns:
        dict[tuple[str, str], dict[str, list[tuple[float, list[float], list[float]]]]: A dictionary
        with (watermark_type, dataset_name) as keys and a list of tuples containing
        the param_name, watermarked scores, and unwatermarked scores.
    """
    data: dict[
        tuple[str, str], dict[str, list[tuple[float, list[float], list[float]]]]
    ] = {}
    rm_bon_suffix_from_wm_type: bool = param_name_to_plot == "BoN"
    for filename in os.listdir(input_dir):
        filename_ = filename
        # no _ allowed in dataset name
        filename = filename.replace("truthful_qa", "truthfulqa")
        if _is_scores_file(filename, score_name):
            print(f"Processing {filename}")
            file_path = os.path.join(input_dir, filename_)
            parsed_info = parse_filename(filename, rm_bon_suffix_from_wm_type)
            if (
                param_name_to_plot in parsed_info
                and parsed_info["model_name"] == model_name_to_plot
            ):
                param_value = parsed_info[param_name_to_plot]
                dataset_name = parsed_info["dataset_name"]
                watermark_type = parsed_info["watermark_type"]
                seed = parsed_info["seed"]

                blobs = read_jsonl(file_path)
                watermarked_sc: list = _get_scores(blobs, "watermarked", score_name)
                unwatermarked_sc: list = _get_scores(blobs, "unwatermarked", score_name)
                key = (watermark_type, dataset_name)
                if key not in data:
                    data[key] = {}
                if seed not in data[key]:
                    data[key][seed] = []
                data[key][seed].append((param_value, watermarked_sc, unwatermarked_sc))
    data = _sort_data_by_params_and_keys(data)
    return data


def get_short_model_name(model_name: str) -> str:
    return {
        "Mistral-7B-Instruct-v0.3": "Mistral-7B-Inst",
        "Meta-Llama-3.1-8B-Instruct": "LLaMA-8B-Inst",
        "gemma-2-9b-it": "Gemma-9B-Inst",
        "Phi-3-mini-4k-instruct": "Phi-3-Mini-Inst",
        "Llama-3.1-8B-Instruct": "LLaMA-8B-Inst",
        "Llama-3.2-1B-Instruct": "LLaMA-1B-Inst",
        "Llama-3.2-3B-Instruct": "LLaMA-3B-Inst",
        "Llama-3.1-70B-Instruct": "LLaMA-70B-Inst",
        "Qwen2.5-7B-Instruct": "Qwen2.5-7B-Inst",
        "Qwen2.5-3B-Instruct": "Qwen2.5-3B-Inst",
        "Qwen2-7B-Instruct": "Qwen2-7B-Inst",
        "Qwen2-3B-Instruct": "Qwen2-3B-Inst",
        "Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Inst",
        "Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B-Inst",
    }.get(model_name, model_name)


def get_short_watermark_name(watermark_type: str) -> str:
    return {
        "unwatermarked": "Unwatermarked",
        "Unwatermarked": "Unwatermarked",
        "maryland": "KGW (Distort)",
        "openai": "Gumbel (Dist-Free)",
        "maryland-BoN-2": "KGW (BoN-2)",
        "openai-BoN-2": "Gumbel (BoN-2)",
        "maryland-BoN-3": "KGW (BoN-3)",
        "openai-BoN-3": "Gumbel (BoN-3)",
        "maryland-BoN-4": "KGW (BoN-4)",
        "openai-BoN-4": "Gumbel (BoN-4)",
        "KGW": "KGW (Distort)",
        "Gumbel": "Gumbel (Dist-Free)",
        "KGW (Distort)": "KGW (Distort)",
        "Gumbel (Dist-Free)": "Gumbel (Dist-Free)",
        "KGW (BoN-2)": "KGW (BoN-2)",
        "Gumbel (BoN-2)": "Gumbel (BoN-2)",
        "KGW (BoN-3)": "KGW (BoN-3)",
        "Gumbel (BoN-3)": "Gumbel (BoN-3)",
        "KGW (BoN-4)": "KGW (BoN-4)",
        "Gumbel (BoN-4)": "Gumbel (BoN-4)",
        "openai-theoretical": "Gumbel (theory)",
        "maryland-theoretical": "KGW (theory)",
    }.get(watermark_type, watermark_type)


def get_color(watermark_type: str) -> str:
    # Maryland family - Orange shades (warmer, more saturated)
    if "maryland" in watermark_type:
        return "#E66101"  # Deep orange
    # OpenAI family - Green shades (more blue-tinted for better distinction)
    elif "openai" in watermark_type:
        return "#1B7837"  # Deep green
    elif "KGW" in watermark_type:
        return "#E66101"  # Deep orange
    elif "Gumbel" in watermark_type:
        return "#1B7837"  # Deep green
    else:
        return "#1f77b4"  # Blue


def get_pattern(watermark_type: str) -> str:
    # Maryland family - Different patterns
    if "BoN-2" in watermark_type:
        return "///"
    elif "BoN-3" in watermark_type:
        return "++"
    elif "BoN-4" in watermark_type:
        return "...."
    else:
        return ""  # No pattern


def get_short_param_name_to_plot(param_name_to_plot: str) -> str:
    return {
        "temperature": "Temperature",
        "BoN": "Best-of-N",
        "ngram": "N-gram",
        "gamma": "Gamma",
        "delta": "Delta",
    }.get(param_name_to_plot, param_name_to_plot.capitalize())
