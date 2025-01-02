#! /bin/bash

set -o errexit -o xtrace -o nounset


# Values to be set by user
###################
CLUSTER="LOCAL"  # "AWS" or "WULVER"
EXP_NAME=${1:-"exp_004_overrefusal_beam_rewards_armo"}
TGT_DIR_NAME=${2:-"exp_004_overrefusal_beam_rewards_armo_refusal"}
###################

if [ "$CLUSTER" = "WULVER" ]; then
    EXP_DIR_PREFIX="/project/phan/av787/projs/watermarking-v1/outputs"
elif [ "$CLUSTER" = "AWS" ]; then
    EXP_DIR_PREFIX="/home/ec2-user/SageMaker/outputs/watermarking-v1"
else
    EXP_DIR_PREFIX="/Users/verapurv/mint/wm-output40GB/outputs/watermarking-v1"
fi

python run_refusal_scorer.py \
    --input_path $EXP_DIR_PREFIX/$EXP_NAME \
    --output_path $EXP_DIR_PREFIX/$TGT_DIR_NAME \
    --scorer_name exact-match \
    --batch_size 1
