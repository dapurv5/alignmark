#! /bin/bash

set -o errexit -o xtrace -o nounset

export PYTHONPATH=$HOME/anahata-python3/reference_projects/2024/MarkLLM:$PYTHONPATH

# for delta in $(seq 2.0 0.5 8.0); do
#     echo "Running delta = $delta ..."
#     sleep 5
#     python run_generate.py \
#         --exp_dir "/project/phan/av787/projs/watermarking/outputs/exp_001_sweep_delta_hh-rlhf_1k" \
#         --watermark_algorithm_default_config_path \
#         "/home/av787/anahata-python3/reference_projects/2024/MarkLLM/config/KGW.json" \
#         --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
#         --watermark_name "KGW" \
#         --dataset_name "Dahoas/full-hh-rlhf" \
#         --watermark_algorithm_config_key "delta" \
#         --watermark_algorithm_config_val $delta \
#         --limit_dataset_size 1000
# done


# for delta in $(seq 7.0 0.5 8.0); do
#     echo "Running delta = $delta ..."
#     sleep 5
#     python run_generate.py \
#         --exp_dir "/project/phan/av787/projs/watermarking/outputs/exp_001_sweep_delta_hh-rlhf_1k" \
#         --watermark_algorithm_default_config_path \
#         "/home/av787/anahata-python3/reference_projects/2024/MarkLLM/config/KGW.json" \
#         --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
#         --watermark_name "KGW" \
#         --dataset_name "Dahoas/full-hh-rlhf" \
#         --watermark_algorithm_config_key "delta" \
#         --watermark_algorithm_config_val $delta \
#         --limit_dataset_size 1000
# done


for prefix_length in $(seq 2 1 10); do
    echo "Running prefix_length = $prefix_length ..."
    sleep 5
    python run_generate.py \
        --exp_dir "/project/phan/av787/projs/watermarking/outputs/exp_002_sweep_prefix_length_hh-rlhf_1k" \
        --watermark_algorithm_default_config_path \
        "/home/av787/anahata-python3/reference_projects/2024/MarkLLM/config/EXP.json" \
        --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --watermark_name "EXP" \
        --dataset_name "Dahoas/full-hh-rlhf" \
        --watermark_algorithm_config_key "prefix_length" \
        --watermark_algorithm_config_val $prefix_length \
        --limit_dataset_size 250
done
