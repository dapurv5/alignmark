#! /bin/bash

set -o errexit -o xtrace -o nounset

EXP_ROOT_DIR=$HOME/mint/watermarking-feasability-exps
EXP_DIR_NAMES=("EXP_001_maryland_viability_sweep" "EXP_001_openai_viability_sweep")
MODEL_NAME="Llama-3.1-8B-Instruct"

for exp_dir_name in "${EXP_DIR_NAMES[@]}"; do
    echo "Generating table for $exp_dir_name (Unfiltered)"
    python plots/table_threshold_diff.py \
        --input_dir $EXP_ROOT_DIR/$exp_dir_name \
        --score_name reward \
        --model_name_to_plot $MODEL_NAME

    # Filter data
    mkdir -p $EXP_ROOT_DIR/${exp_dir_name}_filtered
    # Iterate over all the files in the directory
    for file in $EXP_ROOT_DIR/$exp_dir_name/*; do
        # skip subdirectories
        if [ -d "$file" ]; then
            continue
        fi
        # extract the filename without the extension
        filename=$(basename -- "$file")
        python plots/filter_data.py \
            --input_path $file \
            --output_path $EXP_ROOT_DIR/${exp_dir_name}_filtered/${filename}
    done

    echo "Generating table for $exp_dir_name (Filtered)"
    python plots/table_threshold_diff.py \
        --input_dir $EXP_ROOT_DIR/${exp_dir_name}_filtered \
        --score_name reward \
        --model_name_to_plot $MODEL_NAME
done
