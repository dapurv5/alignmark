import json

from fire import Fire


def filter_data(input_path: str, output_path: str):
    """Filter data to only include examples where the watermarked text is
    detected as watermarked and the unwatermarked text is not detected as
    watermarked.

    Args:
        input_path (str): Path to the input data.
        output_path (str): Path to the output data.
    """
    with open(input_path, "r") as input_fp, open(output_path, "w") as output_fp:
        for example_idx, line in enumerate(input_fp):
            data = json.loads(line)
            if (
                data["watermarked_text.is_watermarked"]
                and not data["unwatermarked_text.is_watermarked"]
            ):
                data["example_idx"] = example_idx
                output_fp.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    Fire(filter_data)
