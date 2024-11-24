#! /bin/bash

set -o errexit -o xtrace -o nounset


MODEL_NAMES=(
    "microsoft/Phi-3-mini-4k-instruct"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "Qwen/Qwen2-7B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
)
for model in "${MODEL_NAMES[@]}"; do
    for seed in $(seq 42 1 42); do
        sleep 2
        python run_generate.py \
            --exp_dir "/project/phan/av787/projs/watermarking/outputs/EXP_004_refusal" \
            --model_name $model \
            --dataset_path "/home/av787/refusal-data.jsonl" \
            --text_field "prompt" \
            --watermark_name "maryland" \
            --delta 2.0 \
            --gamma 0.5 \
            --threshold 0.05 \
            --seed $seed \
            --max_gen_len 200 \
            --top_p 0.95 \
            --batch_size 256 \
            --temperature 1.0 \
            --format_prompt_as_instructions
    done

    for seed in $(seq 42 1 42); do
    sleep 2
    python run_generate.py \
            --exp_dir "/project/phan/av787/projs/watermarking/outputs/EXP_004_refusal" \
            --model_name $model \
            --dataset_path "/home/av787/refusal-data.jsonl" \
            --text_field "prompt" \
            --watermark_name "openai" \
            --threshold 0.05 \
            --seed $seed \
            --max_gen_len 200 \
            --top_p 0.95 \
            --batch_size 256 \
            --temperature 1.0 \
            --format_prompt_as_instructions
    done
done
