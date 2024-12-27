#! /bin/bash

set -o errexit -o xtrace -o nounset

# - meta-llama/Llama-3.1-8B-Instruct
# - microsoft/Phi-3-mini-4k-instruct
# - mistralai/Mistral-7B-Instruct-v0.3
# - Qwen/Qwen2-7B-Instruct


list_of_models=("meta-llama/Llama-3.1-8B-Instruct" \
"microsoft/Phi-3-mini-4k-instruct" \
"mistralai/Mistral-7B-Instruct-v0.3" \
"Qwen/Qwen2-7B-Instruct")

for model in "${list_of_models[@]}"; do
    echo "Running Maryland WM for model: $model"
    bash experiment4/run_maryland.sh $model
done

for model in "${list_of_models[@]}"; do
    echo "Running OpenAI WM for model: $model"
    bash experiment4/run_openai.sh $model
done
