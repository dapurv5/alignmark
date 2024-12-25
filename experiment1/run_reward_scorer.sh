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
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
EXP_NAME="exp_005_sweep_temperature_hh-rlhf_1k_beam"
NUM_GPUS_PER_PROCESS=1
NUM_PROCESSES=4
REWARD_MODEL="llm-blender/PairRM"
REWARD_MODEL_SHORTFORM="blender"

###################

if [ "$CLUSTER" == "WULVER" ]; then
    EXP_DIR_PREFIX="/project/phan/av787/projs/watermarking-v1/outputs"
else
    EXP_DIR_PREFIX="/home/ec2-user/SageMaker/outputs/watermarking-v1"
fi

OUTPUT_DIR=$EXP_DIR_PREFIX/"${EXP_NAME}_rewards_${REWARD_MODEL_SHORTFORM}"

python run_reward_scorer.py \
    --input_path $EXP_DIR_PREFIX/$EXP_NAME \
    --output_path $OUTPUT_DIR \
    --reward_model $REWARD_MODEL \
    --num_processes $NUM_PROCESSES \
    --num_gpus_per_process $NUM_GPUS_PER_PROCESS
