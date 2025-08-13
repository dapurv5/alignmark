#! /bin/bash

set -o errexit -o xtrace -o nounset

# Values to be set by user
###################
CLUSTER="AWS"  # "AWS" or "WULVER"
EXP_NAME=${1:-"exp_001_sweep_temp_hhrlhf_beam"}
NUM_GPUS_PER_PROCESS=1
NUM_PROCESSES=8
REWARD_MODEL="RLHFlow/ArmoRM-Llama3-8B-v0.1"  # llm-blender/PairRM | RLHFlow/ArmoRM-Llama3-8B-v0.1
REWARD_MODEL_SHORTFORM="armo"  # blender | armo
TEXT_FIELD="prompt"
FILE_NAME_SUFFIX="_rewards"
SCORE_FIELD_NAME="reward_score"
###################

if [ "$CLUSTER" == "WULVER" ]; then
    EXP_DIR_PREFIX="/project/phan/av787/projs/outputs/watermarking-v2"
    GPU_START_ID=0
else
    EXP_DIR_PREFIX="/home/ec2-user/SageMaker/outputs/watermarking-v2"
    GPU_START_ID=0
fi
python utils/download_model.py $REWARD_MODEL

OUTPUT_DIR=$EXP_DIR_PREFIX/"${EXP_NAME}_rewards_${REWARD_MODEL_SHORTFORM}"

python run_model_based_scorer.py \
    --input_path $EXP_DIR_PREFIX/$EXP_NAME \
    --output_path $OUTPUT_DIR \
    --scorer_model $REWARD_MODEL \
    --num_processes $NUM_PROCESSES \
    --num_gpus_per_process $NUM_GPUS_PER_PROCESS \
    --gpu_start_id $GPU_START_ID \
    --text_field $TEXT_FIELD \
    --file_name_suffix $FILE_NAME_SUFFIX \
    --score_field_name $SCORE_FIELD_NAME
    #--debug_mode False
