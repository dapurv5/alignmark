#! /bin/bash

set -o errexit -o xtrace -o nounset

# Models to choose from:
# - meta-llama/Llama-3.1-8B-Instruct
# - microsoft/Phi-3-mini-4k-instruct

for temperature in $(seq 0.1 0.1 0.4); do
    for seed in $(seq 42 1 62); do
        echo "Running temperature = $temperature, seed = $seed ..."
        sleep 2
        python run_generate.py \
            --exp_dir "/project/phan/av787/projs/watermarking/outputs/EXP_001_maryland_viability_sweep" \
            --model_name "meta-llama/Llama-3.1-8B-Instruct" \
            --dataset_name "Dahoas/full-hh-rlhf" \
            --dataset_split "test" \
            --text_field "prompt" \
            --watermark_name "maryland" \
            --delta 2.0 \
            --gamma 0.5 \
            --threshold 0.05 \
            --seed $seed \
            --temperature $temperature \
            --max_gen_len 250 \
            --top_p 0.95 \
            --batch_size 64 \
            --limit_dataset_size 256
    done
done
