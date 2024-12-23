"""
Merge the parts files into a single file

    python merge_parts.py \
        --exp_dir "$EXP_DIR_PREFIX/$EXP_NAME/parts" \
        --output_dir "$EXP_DIR_PREFIX/$EXP_NAME"
"""

import glob
import os

import fire


def main(
    exp_dir,
    output_dir,
    dataset_size,
):
    # Get all the part files in the exp_dir
    part_files = glob.glob(os.path.join(exp_dir, "part_*"))
    print(f"Found {len(part_files)} part files")
    # Sort the part files by the part number
    part_files.sort()

    filenames = []
    for part_file in part_files:
        filename = os.path.basename(part_file)
        filenames.append("_".join(filename.split("_")[2:]))
    assert len(set(filenames)) == 1, "All part files should have the same filename"

    total_lines_written = 0
    # Merge the part files into a single file
    with open(os.path.join(output_dir, filenames[0]), "w") as f:
        for part_file in part_files:
            with open(part_file, "r") as part_f:
                f.write(part_f.read())
            with open(part_file, "r") as part_f:
                total_lines_written += sum(1 for _ in part_f)
    assert (
        total_lines_written == dataset_size
    ), f"Total lines written {total_lines_written} should match dataset size {dataset_size}"


if __name__ == "__main__":
    fire.Fire(main)
