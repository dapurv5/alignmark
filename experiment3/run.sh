#! /bin/bash

set -o errexit -o xtrace -o nounset


# for seed in $(seq 42 1 42); do
#     sleep 2
#     python run_generate.py \
#         --exp_dir "/project/phan/av787/projs/watermarking/outputs/exp_010_safety" \
#         --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
#         --dataset_path "/home/av787/safety-data.jsonl" \
#         --text_field "prompt" \
#         --watermark_name "maryland" \
#         --delta 2.0 \
#         --gamma 0.5 \
#         --threshold 0.05 \
#         --seed $seed \
#         --max_gen_len 500 \
#         --top_p 0.95 \
#         --batch_size 128
# done


for seed in $(seq 42 1 42); do
    sleep 2
    python run_generate.py \
        --exp_dir "/project/phan/av787/projs/watermarking/outputs/exp_010_safety" \
        --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --dataset_path "/home/av787/safety-data.jsonl" \
        --text_field "prompt" \
        --watermark_name "openai" \
        --threshold 0.05 \
        --seed $seed \
        --max_gen_len 500 \
        --top_p 0.95 \
        --batch_size 128
done


# for seed in $(seq 42 1 42); do
#     sleep 2
#     python run_generate.py \
#         --exp_dir "/project/phan/av787/projs/watermarking/outputs/exp_010_safety" \
#         --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
#         --dataset_path "/home/av787/safety-data.jsonl" \
#         --text_field "prompt" \
#         --watermark_name "maryland" \
#         --delta 2.0 \
#         --gamma 0.5 \
#         --threshold 0.05 \
#         --seed $seed \
#         --max_gen_len 500 \
#         --top_p 0.95 \
#         --batch_size 128
# done


for seed in $(seq 42 1 42); do
    sleep 2
    python run_generate.py \
        --exp_dir "/project/phan/av787/projs/watermarking/outputs/exp_010_safety" \
        --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
        --dataset_path "/home/av787/safety-data.jsonl" \
        --text_field "prompt" \
        --watermark_name "openai" \
        --threshold 0.05 \
        --seed $seed \
        --max_gen_len 500 \
        --top_p 0.95 \
        --batch_size 128
done
