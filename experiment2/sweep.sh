#! /bin/bash

set -o errexit -o xtrace -o nounset


# for temperature in $(seq 0.2 0.2 1.0); do
#     echo "Running temperature = $temperature ..."
#     sleep 2
#     python run_generate.py \
#         --exp_dir "/project/phan/av787/projs/watermarking/outputs/exp_005_sweep_temperature_hh-rlhf_1k" \
#         --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
#         --dataset_name "Dahoas/full-hh-rlhf" \
#         --text_field "prompt" \
#         --watermark_name "openai" \
#         --threshold 0.05 \
#         --seed 42 \
#         --temperature $temperature \
#         --max_gen_len 250 \
#         --top_p 0.95 \
#         --batch_size 64 \
#         --limit_dataset_size 1024
# done

for temperature in $(seq 0.2 0.2 1.0); do
    echo "Running temperature = $temperature ..."
    sleep 2
    python run_generate.py \
        --exp_dir "/project/phan/av787/projs/watermarking/outputs/exp_006_sweep_temperature_truthful_qa" \
        --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --dataset_name "truthfulqa/truthful_qa" \
        --dataset_subset_name "generation" \
        --dataset_split "validation" \
        --text_field "question" \
        --watermark_name "maryland" \
        --delta 2.0 \
        --gamma 0.5 \
        --threshold 0.05 \
        --seed 42 \
        --temperature $temperature \
        --max_gen_len 50 \
        --top_p 0.95 \
        --batch_size 256
done


for temperature in $(seq 0.2 0.2 1.0); do
    echo "Running temperature = $temperature ..."
    sleep 2
    python run_generate.py \
        --exp_dir "/project/phan/av787/projs/watermarking/outputs/exp_006_sweep_temperature_truthful_qa" \
        --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --dataset_name "truthfulqa/truthful_qa" \
        --dataset_subset_name "generation" \
        --dataset_split "validation" \
        --text_field "question" \
        --watermark_name "openai" \
        --threshold 0.05 \
        --seed 42 \
        --temperature $temperature \
        --max_gen_len 50 \
        --top_p 0.95 \
        --batch_size 256
done


for temperature in $(seq 0.2 0.2 1.0); do
    echo "Running temperature = $temperature ..."
    sleep 2
    python run_generate.py \
        --exp_dir "/project/phan/av787/projs/watermarking/outputs/exp_006_sweep_temperature_truthful_qa" \
        --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
        --dataset_name "truthfulqa/truthful_qa" \
        --dataset_subset_name "generation" \
        --dataset_split "validation" \
        --text_field "question" \
        --watermark_name "openai" \
        --threshold 0.05 \
        --seed 42 \
        --temperature $temperature \
        --max_gen_len 50 \
        --top_p 0.95 \
        --batch_size 256
done
