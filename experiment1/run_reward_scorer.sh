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
EXP_NAME=${1:-"exp_001_sweep_temp_hhrlhf_beam"}
NUM_GPUS_PER_PROCESS=2
NUM_PROCESSES=2
REWARD_MODEL="armo"  # llm-blender/PairRM | armo
REWARD_MODEL_SHORTFORM="armo"  # blender | armo
###################

if [ "$CLUSTER" == "WULVER" ]; then
    EXP_DIR_PREFIX="/project/phan/av787/projs/watermarking-v1/outputs"
    GPU_START_ID=0
else
    EXP_DIR_PREFIX="/home/ec2-user/SageMaker/outputs/watermarking-v1"
    GPU_START_ID=4
fi

OUTPUT_DIR=$EXP_DIR_PREFIX/"${EXP_NAME}_rewards_${REWARD_MODEL_SHORTFORM}"

python run_reward_scorer.py \
    --input_path $EXP_DIR_PREFIX/$EXP_NAME \
    --output_path $OUTPUT_DIR \
    --reward_model $REWARD_MODEL \
    --num_processes $NUM_PROCESSES \
    --num_gpus_per_process $NUM_GPUS_PER_PROCESS \
    --gpu_start_id $GPU_START_ID
    #--debug_mode False
