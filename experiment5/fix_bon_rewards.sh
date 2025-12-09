#!/bin/bash

# This script is a wrapper script over the Python script `fix_bon_ppl.py`.
#
# It runs on a preselected list of folders containing JSONL files.
# And writes the fixed JSONL files to corresponding output folders


# Input the base directory, this hosts all the folders mentioned above
base_dir=${1:-"$HOME/mint/wm-output40GB/outputs/watermarking-v3"}

# Loop over the folders
FOLDERS=(
    "exp_002_sweep_temp_truthfulqa_beam_rewards_armo_subset_truthfulness_BoN_rewards_skywork"
    "exp_003_safety_beam_rewards_armo_subset_safety_BoN_v1_rewards_skywork"
    "exp_004_overrefusal_beam_rewards_armo_refusal_BoN_rewards_skywork"
)

for folder in "${FOLDERS[@]}"; do
    input_folder="$base_dir/$folder"
    output_folder="$base_dir/${folder}_fixed"

    echo "Processing folder: $input_folder"
    echo "Output will be saved to: $output_folder"

    # Create output folder if it doesn't exist
    mkdir -p "$output_folder"

    # Run the Python script to fix the JSONL files
    python3 experiment5/fix_bon_rewards.py --input_folder "$input_folder" --output_folder "$output_folder"

    echo "Finished processing folder: $input_folder"
done
echo "All folders processed."
