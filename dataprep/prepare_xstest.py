import csv
import json

from fire import Fire


def main(input_path: str, output_path: str):
    with open(input_path, "r") as input_fp, open(output_path, "w") as output_fp:
        reader = csv.DictReader(input_fp)
        for row in reader:
            id_v1 = row["id_v1"].strip()
            if id_v1:
                output_fp.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    Fire(main)
