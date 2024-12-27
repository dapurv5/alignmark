#! /bin/bash

set -o errexit -o xtrace -o nounset

# Models to choose from:
# - meta-llama/Llama-3.1-8B-Instruct
# - microsoft/Phi-3-mini-4k-instruct
# - mistralai/Mistral-7B-Instruct-v0.3
# - Qwen/Qwen2-7B-Instruct
# - google/gemma-2-9b-it (not working)

# Values to be set by user
###################
CLUSTER="AWS"  # "AWS" or "WULVER"
MODEL_NAME=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
EXP_NAME=${2:-"exp_002_sweep_temp_truthfulqa_beam"}
# Choose number of GPUs to be exactly divisible by DATASET_SIZE
NUM_GPUS_AVAILABLE=4  # Don't set > 4 for now, because it hangs after a while for unknown reasons
DATASET_SIZE=800  # The actual dataset size is 817
BATCH_SIZE=8  # Use batch size 8 for 40GB GPU and also for 80GB GPU to keep it full utilized
SEED=42
CLEAN_MODEL_AFTER_RUN=${CLEAN_MODEL_AFTER_RUN:-"false"}
# Choose these values based on the GPU memory available
# --num_wm_generations_per_prompt 4 \  # 4 for 40GB, 8 for 80GB
# --num_unwm_generations_per_prompt 2 \  # 2 for 40GB, 4 for 80GB
# --beam_size 4 \  # 4 for 40GB, 8 for 80GB

###################
# Before running the script give a prompt to the user to enter y for the question
# "Are the model already downloaded and cached and symlinks created?"
# read -p "Are the model already downloaded and cached and symlinks created? (y/n): " answer
# if [ "$answer" != "y" ]; then
#     echo "Please download the model and create symlinks before running this script."
#     exit 1
# fi

python utils/download_model.py $MODEL_NAME

if [ "$CLUSTER" == "WULVER" ]; then
    EXP_DIR_PREFIX="/project/phan/av787/projs/watermarking-v1/outputs"
else
    EXP_DIR_PREFIX="/home/ec2-user/SageMaker/outputs/watermarking-v1"
fi

# Generally higher temperatures because outputs are diverse
for temperature in $(seq 0.2 0.2 1.0); do
    echo "Running temperature = $temperature ..."
    sleep 2

    # Calculate rows per GPU
    ROWS_PER_GPU=$((DATASET_SIZE / NUM_GPUS_AVAILABLE))

    for gpu in $(seq 0 $((NUM_GPUS_AVAILABLE - 1))); do
        # Calculate start and end rows for this GPU
        START_ROW=$((gpu * ROWS_PER_GPU))
        END_ROW=$(((gpu + 1) * ROWS_PER_GPU - 1))

        # For the last GPU, make sure we process any remaining rows
        if [ $gpu -eq $((NUM_GPUS_AVAILABLE - 1)) ]; then
            END_ROW=$((DATASET_SIZE - 1))
        fi

        # Calculate number of rows for this GPU and fix batch size if necessary
        NUM_ROWS_FOR_GPU=$((END_ROW - START_ROW + 1))
        # Batch size is minimum of 32 and number of rows per GPU
        BATCH_SIZE=$((NUM_ROWS_FOR_GPU < BATCH_SIZE ? NUM_ROWS_FOR_GPU : BATCH_SIZE))

        # Launch process for each GPU in background
        (
            CUDA_VISIBLE_DEVICES=$gpu python run_generate.py \
                --exp_dir "$EXP_DIR_PREFIX/$EXP_NAME/parts" \
                --model_name $MODEL_NAME \
                --dataset_name "truthfulqa/truthful_qa" \
                --dataset_subset_name "generation" \
                --dataset_split "validation" \
                --text_field "question" \
                --watermark_name "maryland" \
                --delta 2.0 \
                --gamma 0.25 \
                --ngram 4 \
                --threshold 0.05 \
                --seed $SEED \
                --temperature $temperature \
                --max_gen_len 50 \
                --top_p 0.95 \
                --batch_size $BATCH_SIZE \
                --dataset_start_row $START_ROW \
                --dataset_end_row $END_ROW \
                --num_wm_generations_per_prompt 4 \
                --num_unwm_generations_per_prompt 2 \
                --beam_size 4 \
                #--select_random_subset_from_dataset  # (keep this off otherwise the dataset will change)
        ) &
        sleep 10
    done
    # Wait for all background processes to complete before moving to next temperature
    wait

    # Now merge the part files into a single file from the experiment directory
    python merge_parts.py \
        --exp_dir "$EXP_DIR_PREFIX/$EXP_NAME/parts" \
        --output_dir "$EXP_DIR_PREFIX/$EXP_NAME" \
        --dataset_size $DATASET_SIZE

    # Delete the parts directory
    rm -rf "$EXP_DIR_PREFIX/$EXP_NAME/parts"
done

if [ "$CLEAN_MODEL_AFTER_RUN" == "true" ]; then
    python utils/clean_model.py $MODEL_NAME
fi
