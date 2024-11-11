#! /bin/bash

set -o errexit -o xtrace -o nounset


for temperature in $(seq 0.2 0.2 1.0); do
    echo "Running temperature = $temperature ..."
    sleep 2
    python run_generate.py \
        --exp_dir "/project/phan/av787/projs/watermarking/outputs/exp_014_sweep_temperature_hh-rlhf_1k" \
        --model_name "microsoft/Phi-3-mini-4k-instruct" \
        --dataset_name "Dahoas/full-hh-rlhf" \
        --text_field "prompt" \
        --watermark_name "openai" \
        --threshold 0.05 \
        --seed 42 \
        --temperature $temperature \
        --max_gen_len 250 \
        --top_p 0.95 \
        --batch_size 64 \
        --limit_dataset_size 1024
done

# for temperature in $(seq 0.2 0.2 1.0); do
#     echo "Running temperature = $temperature ..."
#     sleep 2
#     python run_generate.py \
#         --exp_dir "/project/phan/av787/projs/watermarking/outputs/exp_014_sweep_temperature_hh-rlhf_1k" \
#         --model_name "microsoft/Phi-3-mini-4k-instruct" \
#         --dataset_name "Dahoas/full-hh-rlhf" \
#         --dataset_split "test" \
#         --text_field "prompt" \
#         --watermark_name "maryland" \
#         --delta 2.0 \
#         --gamma 0.5 \
#         --threshold 0.05 \
#         --seed 42 \
#         --temperature $temperature \
#         --max_gen_len 250 \
#         --top_p 0.95 \
#         --batch_size 64 \
#         --limit_dataset_size 1024
# done
