#! /bin/bash

set -o errexit -o xtrace -o nounset

# list_of_models=("meta-llama/Llama-3.1-8B-Instruct" \
# "microsoft/Phi-3-mini-4k-instruct" \
# "mistralai/Mistral-7B-Instruct-v0.3" \
# "Qwen/Qwen2.5-7B-Instruct" \
# "Qwen/Qwen2.5-1.5B-Instruct" \
# "Qwen/Qwen2.5-3B-Instruct" \
# "meta-llama/Llama-3.2-1B-Instruct" \
# "meta-llama/Llama-3.2-3B-Instruct" \
# "Qwen/Qwen2.5-0.5B-Instruct" \
# "Qwen/Qwen2.5-14B-Instruct" \
# "mistralai/Mistral-Small-3.1-24B-Instruct-2503" )

list_of_models=("Qwen/Qwen2.5-14B-Instruct" \
	"google/gemma-2-27b-it" \
	"mistralai/Mistral-Small-3.1-24B-Instruct-2503" )


for model in "${list_of_models[@]}"; do
    echo "Running Maryland WM for model: $model"
    export CLEAN_MODEL_AFTER_RUN="false"
    bash experiment2/run_maryland.sh $model
    echo "Running OpenAI WM for model: $model"
    export CLEAN_MODEL_AFTER_RUN="true"
    bash experiment2/run_openai.sh $model
done
