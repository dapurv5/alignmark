#! /bin/bash

set -o errexit -o xtrace -o nounset

# Models to choose from:
# - meta-llama/Llama-3.1-8B-Instruct
# - microsoft/Phi-3-mini-4k-instruct

DATASET_NAME="Dahoas/full-hh-rlhf"
DATASET_SIZE=1024
NUM_GPUS_AVAILABLE=8

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

    # Launch process for each GPU in background
    (
        for temperature in $(seq 0.1 0.1 0.4); do
            echo "Running on GPU $gpu, temperature = $temperature, rows $START_ROW to $END_ROW"
            sleep 2
            CUDA_VISIBLE_DEVICES=$gpu python run_generate.py \
                --exp_dir "/home/ec2-user/SageMaker/outputs/wm_bulk_gen/openai_v0" \
                --model_name "meta-llama/Llama-3.1-8B-Instruct" \
                --dataset_name $DATASET_NAME \
                --text_field "prompt" \
                --watermark_name "openai" \
                --threshold 0.05 \
                --seed 42 \
                --temperature $temperature \
                --max_gen_len 250 \
                --top_p 0.95 \
                --batch_size 32 \
                --num_generations_per_prompt 10 \
                --turn_shuffle_off \
                --dataset_start_row $START_ROW \
                --dataset_end_row $END_ROW
        done
    ) &
done

# Wait for all background processes to complete
wait

echo "All processes completed"

