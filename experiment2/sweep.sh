#! /bin/bash

set -o errexit -o xtrace -o nounset


# for seed in $(seq 43 1 44); do
#     for temperature in $(seq 0.2 0.2 1.0); do
#         echo "Running temperature = $temperature ..."
#         sleep 2
#         python run_generate.py \
#             --exp_dir "/project/phan/av787/projs/watermarking/outputs/exp_008_sweep_temperature_truthful_qa" \
#             --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
#             --dataset_name "truthfulqa/truthful_qa" \
#             --dataset_subset_name "generation" \
#             --dataset_split "validation" \
#             --text_field "question" \
#             --watermark_name "maryland" \
#             --delta 2.0 \
#             --gamma 0.5 \
#             --threshold 0.05 \
#             --seed $seed \
#             --temperature $temperature \
#             --max_gen_len 50 \
#             --top_p 0.95 \
#             --batch_size 256
#     done
# done


# for seed in $(seq 43 1 44); do
#     for temperature in $(seq 0.2 0.2 1.0); do
#         echo "Running temperature = $temperature ..."
#     sleep 2
#         python run_generate.py \
#             --exp_dir "/project/phan/av787/projs/watermarking/outputs/exp_008_sweep_temperature_truthful_qa" \
#             --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
#             --dataset_name "truthfulqa/truthful_qa" \
#             --dataset_subset_name "generation" \
#             --dataset_split "validation" \
#             --text_field "question" \
#             --watermark_name "openai" \
#             --threshold 0.05 \
#             --seed $seed \
#             --temperature $temperature \
#             --max_gen_len 50 \
#             --top_p 0.95 \
#             --batch_size 256
#     done
# done


# for seed in $(seq 43 1 44); do
#     for temperature in $(seq 0.2 0.2 1.0); do
#         echo "Running temperature = $temperature ..."
#         sleep 2
#         python run_generate.py \
#             --exp_dir "/project/phan/av787/projs/watermarking/outputs/exp_008_sweep_temperature_truthful_qa" \
#             --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
#             --dataset_name "truthfulqa/truthful_qa" \
#             --dataset_subset_name "generation" \
#             --dataset_split "validation" \
#             --text_field "question" \
#             --watermark_name "openai" \
#             --threshold 0.05 \
#             --seed $seed \
#             --temperature $temperature \
#             --max_gen_len 50 \
#             --top_p 0.95 \
#             --batch_size 256
#     done
# done


# for seed in $(seq 43 1 44); do
#     for temperature in $(seq 0.2 0.2 1.0); do
#         echo "Running temperature = $temperature ..."
#         sleep 2
#         python run_generate.py \
#             --exp_dir "/project/phan/av787/projs/watermarking/outputs/exp_008_sweep_temperature_truthful_qa" \
#             --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
#             --dataset_name "truthfulqa/truthful_qa" \
#             --dataset_subset_name "generation" \
#             --dataset_split "validation" \
#             --text_field "question" \
#             --watermark_name "maryland" \
#             --delta 2.0 \
#             --gamma 0.5 \
#             --threshold 0.05 \
#             --seed $seed \
#             --temperature $temperature \
#             --max_gen_len 50 \
#             --top_p 0.95 \
#             --batch_size 256
#     done
# done


for seed in $(seq 43 1 43); do
    for temperature in $(seq 0.2 0.2 1.0); do
        echo "Running temperature = $temperature ..."
        sleep 2
        python run_generate.py \
            --exp_dir "/project/phan/av787/projs/watermarking/outputs/exp_009_sweep_temperature_truthful_qa_gpt4omini" \
            --model_name "google/gemma-2-9b-it" \
            --dataset_name "truthfulqa/truthful_qa" \
            --dataset_subset_name "generation" \
            --dataset_split "validation" \
            --text_field "question" \
            --watermark_name "openai" \
            --threshold 0.05 \
            --seed $seed \
            --temperature $temperature \
            --max_gen_len 50 \
            --top_p 0.95 \
            --batch_size 256
    done
done


for seed in $(seq 43 1 43); do
    for temperature in $(seq 0.2 0.2 1.0); do
        echo "Running temperature = $temperature ..."
        sleep 2
        python run_generate.py \
            --exp_dir "/project/phan/av787/projs/watermarking/outputs/exp_009_sweep_temperature_truthful_qa_gpt4omini" \
            --model_name "google/gemma-2-9b-it" \
            --dataset_name "truthfulqa/truthful_qa" \
            --dataset_subset_name "generation" \
            --dataset_split "validation" \
            --text_field "question" \
            --watermark_name "maryland" \
            --delta 2.0 \
            --gamma 0.5 \
            --threshold 0.05 \
            --seed $seed \
            --temperature $temperature \
            --max_gen_len 50 \
            --top_p 0.95 \
            --batch_size 256
    done
done