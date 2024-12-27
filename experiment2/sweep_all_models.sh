#! /bin/bash

set -o errexit -o xtrace -o nounset

# - meta-llama/Llama-3.1-8B-Instruct
# - microsoft/Phi-3-mini-4k-instruct
# - mistralai/Mistral-7B-Instruct-v0.3
# - Qwen/Qwen2.5-7B-Instruct
# - Qwen/Qwen2.5-1.5B-Instruct
# - Qwen/Qwen2.5-3B-Instruct
# - meta-llama/Llama-3.2-1B-Instruct
# - meta-llama/Llama-3.2-3B-Instruct
# - Qwen/Qwen2.5-14B-Instruct



list_of_models=("meta-llama/Llama-3.1-8B-Instruct" \
"microsoft/Phi-3-mini-4k-instruct" \
"mistralai/Mistral-7B-Instruct-v0.3" \
"Qwen/Qwen2-7B-Instruct" \
"Qwen/Qwen2-1.5B-Instruct" \
"Qwen/Qwen2-3B-Instruct" \
"meta-llama/Llama-3.2-1B-Instruct" \
"meta-llama/Llama-3.2-3B-Instruct" \
"Qwen/Qwen2-14B-Instruct" )


for model in "${list_of_models[@]}"; do
    echo "Downloading model: $model"
    bash utils/download_model.py $model
done

for model in "${list_of_models[@]}"; do
    echo "Running Maryland WM for model: $model"
    export CLEAN_MODEL_AFTER_RUN="false"
    bash experiment2/sweep_maryland_beam.sh $model
    echo "Running OpenAI WM for model: $model"
    export CLEAN_MODEL_AFTER_RUN="true"
    bash experiment2/sweep_openai_beam.sh $model
done

for model in "${list_of_models[@]}"; do
    echo "Cleaning model: $model"
    bash utils/clean_model.py $model
done

