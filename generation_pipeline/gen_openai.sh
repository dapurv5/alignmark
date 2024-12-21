#! /bin/bash

set -o errexit -o xtrace -o nounset

# Models to choose from:
# - meta-llama/Llama-3.1-8B-Instruct
# - microsoft/Phi-3-mini-4k-instruct

gpu=$1

for temperature in $(seq 0.1 0.1 0.4); do
    echo "Running temperature = $temperature"
    sleep 2
    CUDA_VISIBLE_DEVICES=$gpu python run_generate.py \
        --exp_dir "/home/ec2-user/SageMaker/outputs/wm_bulk_gen/openai_v0" \
        --model_name "meta-llama/Llama-3.1-8B-Instruct" \
        --dataset_name "Dahoas/full-hh-rlhf" \
        --text_field "prompt" \
        --watermark_name "openai" \
        --threshold 0.05 \
        --seed 42 \
        --temperature $temperature \
        --max_gen_len 250 \
        --top_p 0.95 \
        --batch_size 128 \
        --limit_dataset_size 256 \
        --pairs_generator  # comment this out to run the non-pairs generator
        --num_generations_per_prompt 10  # only used for non-pairs generator (no-op for pairs generator)
done
