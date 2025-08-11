#! /bin/bash

set -o errexit -o xtrace -o nounset

# Values to be set by user
###################
CLUSTER="AWS"  # "AWS" or "WULVER"
MODEL_NAME=${1:-"meta-llama/Llama-3.2-1B-Instruct"}
EXP_NAME=${2:-"exp_002_truthfulqa_beam"}
SEED=42
CLEAN_MODEL_AFTER_RUN=${CLEAN_MODEL_AFTER_RUN:-"false"}

python utils/download_model.py $MODEL_NAME

if [ "$CLUSTER" == "WULVER" ]; then
    EXP_DIR_PREFIX="/project/phan/av787/projs/outputs/watermarking-v2"
    VLLM_WATERMARK_DIR="$HOME/vLLM-Watermark"
else
    EXP_DIR_PREFIX="/home/ec2-user/SageMaker/outputs/watermarking-v2"
    VLLM_WATERMARK_DIR="/home/ec2-user/SageMaker/vLLM-Watermark"
fi

SIMPLE_MODEL_NAME="${MODEL_NAME##*/}"  # meta-llama/Llama-3.2-1B-Instruct -> Llama-3.2-1B-Instruct
OUTPUT_FILENAME="out_truthfulqa_${SIMPLE_MODEL_NAME}_openai_${SEED}_temperature_1.0_ngram_4.jsonl"

python $VLLM_WATERMARK_DIR/scripts/generate_wm_and_unwm.py \
  --input_path "truthfulqa/truthful_qa" \
  --hf_name "generation" \
  --hf_split "validation" \
  --input_key "question" \
  --output_path "$EXP_DIR_PREFIX/$EXP_NAME/$OUTPUT_FILENAME" \
  --watermarking_algorithm "OPENAI_DR" \
  --model_name $MODEL_NAME \
  --seed $SEED \
  --ngram 4 \
  --detection_threshold 0.05 \
  --temperature 1.0 \
  --max_tokens 128 \
  --top_p 0.95 \
  --num_wm_generations_per_prompt 4 \
  --num_unwm_generations_per_prompt 2 \
  --dataset_start_row 0 \
  --dataset_end_row 800

if [ "$CLEAN_MODEL_AFTER_RUN" == "true" ]; then
    python utils/clean_model.py $MODEL_NAME
fi
