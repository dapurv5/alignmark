#! /bin/bash

set -o errexit -o xtrace -o nounset

# Models to choose from:
# - meta-llama/Llama-3.1-8B-Instruct
# - microsoft/Phi-3-mini-4k-instruct
# - mistralai/Mistral-7B-Instruct-v0.3
# - Qwen/Qwen2-7B-Instruct

# Values to be set by user
###################
CLUSTER="AWS"  # "AWS" or "WULVER"
MODEL_NAME=${1:-"meta-llama/Llama-3.2-1B-Instruct"}
EXP_NAME=${2:-"exp_001_sweep_temp_hhrlhf_beam"}
DATASET_SIZE=1024
SEED=42
CLEAN_MODEL_AFTER_RUN=${CLEAN_MODEL_AFTER_RUN:-"false"}
###################
## python utils/download_model.py $MODEL_NAME

if [ "$CLUSTER" == "WULVER" ]; then
    EXP_DIR_PREFIX="/project/phan/av787/projs/outputs/watermarking-v2"
    VLLM_WATERMARK_DIR="$HOME/vLLM-Watermark"
else
    EXP_DIR_PREFIX="/home/ec2-user/SageMaker/outputs/watermarking-v2"
    VLLM_WATERMARK_DIR="/home/ec2-user/SageMaker/vLLM-Watermark"
fi

SIMPLE_MODEL_NAME="${MODEL_NAME##*/}"  # meta-llama/Llama-3.2-1B-Instruct -> Llama-3.2-1B-Instruct

# Generally higher temperatures because outputs are diverse
for temperature in $(seq 0.2 0.2 1.0); do
    echo "Running temperature = $temperature ..."
    sleep 2

    OUTPUT_FILENAME="out_hhrlhf_${SIMPLE_MODEL_NAME}_openai_${SEED}_temperature_${temperature}_ngram_4.jsonl"

    CUDA_VISIBLE_DEVICES=0 python $VLLM_WATERMARK_DIR/scripts/generate_wm_and_unwm.py \
      --input_path "Dahoas/full-hh-rlhf" \
      --hf_split "test" \
      --input_key "prompt" \
      --output_path "$EXP_DIR_PREFIX/$EXP_NAME/$OUTPUT_FILENAME" \
      --watermarking_algorithm "OPENAI_DR" \
      --model_name $MODEL_NAME \
      --seed $SEED \
      --ngram 4 \
      --detection_threshold 0.10 \
      --temperature $temperature \
      --max_tokens 250 \
      --top_p 0.95 \
      --num_wm_generations_per_prompt 8 \
      --num_unwm_generations_per_prompt 4 \
      --dataset_start_row 0 \
      --dataset_end_row $DATASET_SIZE
done

if [ "$CLEAN_MODEL_AFTER_RUN" == "true" ]; then
    python utils/clean_model.py $MODEL_NAME
fi
