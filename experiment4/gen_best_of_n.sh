#! /bin/bash

set -o errexit -o xtrace -o nounset

CLUSTER="LOCAL"  # "AWS" or "WULVER"
N=${1:-2}
EXP_NAME=${2:-"exp_004_overrefusal_beam_rewards_armo_refusal"}
TGT_DIR_NAME=${3:-"exp_004_overrefusal_beam_rewards_armo_refusal_BoN"}
if [ "$CLUSTER" = "WULVER" ]; then
    EXP_DIR_PREFIX="/project/phan/av787/projs/watermarking-v1/outputs"
elif [ "$CLUSTER" = "AWS" ]; then
    EXP_DIR_PREFIX="/home/ec2-user/SageMaker/outputs/watermarking-v1"
else
    EXP_DIR_PREFIX="/Users/verapurv/mint/wm-output40GB/outputs/watermarking-v1"
fi

# In words, this is saying find the index of the top score in idx = watermarked_texts_reward_score[:n]
# and then overwrite the tgt_score_field with this value
# Also, overwrite tgt_associated_fields with the top score and associated fields
python generate_best_of_n.py \
    --input_dir $EXP_DIR_PREFIX/$EXP_NAME \
    --output_dir $EXP_DIR_PREFIX/$TGT_DIR_NAME \
    --score_field "watermarked_texts_refusal_eval" \
    --src_fields "watermarked_texts,watermarked_texts_refusal_eval,watermarked_texts.is_watermarked,watermarked_texts.score,watermarked_texts.pvalue" \
    --tgt_score_field "watermarked_text_refusal_eval" \
    --tgt_fields "watermarked_text,watermarked_text_refusal_eval,watermarked_text.is_watermarked,watermarked_text.score,watermarked_text.pvalue" \
    --n $N \
    --filename_pattern "*_refusal_scores.jsonl" \
    --preferred_categorical_label 0
