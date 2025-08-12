#! /bin/bash

set -o errexit -o xtrace -o nounset


# Values to be set by user
###################
CLUSTER="LOCAL"  # "AWS" or "WULVER"
EXP_NAME=${1:-"exp_003_safety_beam_rewards_armo"}
TGT_DIR_NAME=${2:-"exp_003_safety_beam_rewards_armo_safety"}
###################

if [ "$CLUSTER" = "WULVER" ]; then
    EXP_DIR_PREFIX="/project/phan/av787/projs/outputs/watermarking-v2"
elif [ "$CLUSTER" = "AWS" ]; then
    EXP_DIR_PREFIX="/home/ec2-user/SageMaker/outputs/watermarking-v2"
else
    EXP_DIR_PREFIX="$HOME/mint/wm-output40GB/outputs/watermarking-v2"
fi

python run_safety_scorer.py \
    --input_path $EXP_DIR_PREFIX/$EXP_NAME \
    --output_path $EXP_DIR_PREFIX/$TGT_DIR_NAME \
    --scorer_name openai \
    --batch_size 8
