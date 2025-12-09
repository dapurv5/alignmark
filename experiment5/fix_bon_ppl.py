import glob
import json
import os

from fire import Fire


def get_list_of_jsonl_files(folder_path: str):
    pattern = os.path.join(folder_path, "*.jsonl")
    return sorted(glob.glob(pattern))


def read_jsonl_as_list(file_path: str):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def write_list_as_jsonl(data: list, file_path: str):
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def get_n_from_filename(filename: str) -> int | None:
    """
    Extracts the N value from the filename.
    For e.g. filenames can be like
    - out_truthfulqa_Llama-3.1-8B-Instruct_maryland-BoN-2_42_temperature_1.0_delta_2.0_gamma_0.25_ngram_4_rewards_truthfulness_ppl.jsonl
    - out_truthfulqa_Phi-3-mini-4k-instruct_openai-BoN-2_42_temperature_1.0_ngram_4_rewards_truthfulness_rewards.jsonl
    - out_safety-data_Llama-3.1-8B-Instruct_maryland-BoN-2_42_temperature_1.0_delta_2.0_gamma_0.25_ngram_4_rewards_safety_scores_ppl.jsonl

    Args:
        filename (str): The filename to extract N from.

    Returns:
        int | None: The extracted N value, or None if pattern not found.
    """
    # Try both patterns: "-BoN-" and "_BoN-"
    for pattern in ["-BoN-", "_BoN-"]:
        if pattern in filename:
            arr = filename.split(pattern)[-1].split("_")
            try:
                return int(arr[0])
            except ValueError:
                continue
    return None


# Mapping from plural field names to singular field names
FIELD_MAPPINGS = {
    "watermarked_texts_ppl_score": "watermarked_text_ppl_score",
    "watermarked_texts_truthfulness_score": "watermarked_text_truthfulness_score",
    "watermarked_texts_reward_score": "watermarked_text_reward_score",
    "watermarked_texts": "watermarked_text",
    "watermarked_texts.pvalue": "watermarked_text.pvalue",
    "watermarked_texts.score": "watermarked_text.score",
    "watermarked_texts.is_watermarked": "watermarked_text.is_watermarked",
    "watermarked_texts_safety_eval": "watermarked_text_safety_eval",
    "watermarked_texts_unsafe_category": "watermarked_text_unsafe_category",
    "watermarked_texts_refusal_eval": "watermarked_text_refusal_eval",
}


def main(
    input_folder: str,
    output_folder: str,
):
    """
    This function takes an input folder with a list of jsonl files
    From the filename it determines the N to use for BoN.

    It looks at each json blob's `watermarked_texts_ppl_score` field
    This field should be a list of floats with length >= N

    It picks the maximum value from the first N values in the list
    and writes it to the `watermarked_text_ppl_score` field
    It also remembers the idx of the chosen value and uses it to fix other
    fields.
    The other fields that are fixed (if present) are:
        - watermarked_texts_truthfulness_score
        - watermarked_texts_reward_score
        - watermarked_texts.pvalue
        - watermarked_texts.score
        - watermarked_texts.is_watermarked
        - watermarked_texts
        - watermarked_texts_safety_eval
        - watermarked_texts_unsafe_category
        - watermarked_texts_refusal_eval
    For each of the above fields, it picks the value at the chosen idx
    and writes it to the corresponding singular field.
    The corresponding singular fields are:
        - watermarked_text_truthfulness_score
        - watermarked_text_reward_score
        - watermarked_text.pvalue
        - watermarked_text.score
        - watermarked_text.is_watermarked
        - watermarked_text
        - watermarked_text_safety_eval
        - watermarked_text_unsafe_category
        - watermarked_text_refusal_eval

    Finally it writes the modified json blobs to the output folder

    Args:
        input_folder (str): _description_
        output_folder (str): _description_
    """
    os.makedirs(output_folder, exist_ok=True)
    input_files = get_list_of_jsonl_files(input_folder)
    for input_file in input_files:
        filename = os.path.basename(input_file)
        n = get_n_from_filename(filename)
        if n is None:
            print(f"Skipping file (no BoN pattern found): {filename}")
            continue
        data = read_jsonl_as_list(input_file)
        for item in data:
            ppl_scores = item["watermarked_texts_ppl_score"][:n]
            max_ppl = max(ppl_scores)
            max_idx = ppl_scores.index(max_ppl)

            # Apply field mappings for all fields that exist in the item
            for plural_field, singular_field in FIELD_MAPPINGS.items():
                if plural_field in item:
                    if plural_field == "watermarked_texts_ppl_score":
                        # For the ppl_score field, use the max value
                        item[singular_field] = max_ppl
                    else:
                        # For all other fields, use the value at max_idx
                        item[singular_field] = item[plural_field][max_idx]

        # Remove the _ppl suffix for the plotting to work correctly
        output_file = os.path.join(output_folder, filename.replace("_ppl", ""))
        write_list_as_jsonl(data, output_file)


if __name__ == "__main__":
    Fire(main)
