import json

from datasets import load_dataset
from fire import Fire


def process_or_bench(split="train"):
    """
    Process the or-bench dataset and convert it to the required format.

    Args:
        split (str): Dataset split to process ('train' or 'test')

    Returns:
        list: List of processed examples
    """
    # Load the dataset
    dataset = load_dataset("bench-llm/or-bench", "or-bench-hard-1k")
    data = dataset[split]

    processed_examples = []
    for example in data:
        processed_examples.append(example)

    return processed_examples


def main(output_path: str):
    """
    Main function to process the dataset and save it to a JSONL file.
    """
    # Process the dataset
    examples = process_or_bench(split="train")  # or "test" depending on your needs

    # Save to JSONL file
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"Processed {len(examples)} examples")


if __name__ == "__main__":
    Fire(main)
