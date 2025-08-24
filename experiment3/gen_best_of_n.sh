#! /bin/bash

set -o errexit -o xtrace -o nounset

CLUSTER="LOCAL"  # "AWS" or "WULVER"
N=${1:-2}
EXP_NAME=${2:-"exp_003_safety_beam_rewards_armo_safety"}
TGT_DIR_NAME=${3:-"exp_003_safety_beam_rewards_armo_safety_BoN"}
if [ "$CLUSTER" = "WULVER" ]; then
    EXP_DIR_PREFIX="/project/phan/av787/projs/outputs/watermarking-v2"
elif [ "$CLUSTER" = "AWS" ]; then
    EXP_DIR_PREFIX="/home/ec2-user/SageMaker/outputs/watermarking-v2"
else
    EXP_DIR_PREFIX="/Users/verapurv/mint/wm-output40GB/outputs/watermarking-v2"
fi

# In words, this is saying find the index of the top score in idx = watermarked_texts_reward_score[:n]
# and then overwrite the tgt_score_field with this value
# Also, overwrite tgt_associated_fields with the top score and associated fields
python generate_best_of_n.py \
    --input_dir $EXP_DIR_PREFIX/$EXP_NAME \
    --output_dir $EXP_DIR_PREFIX/$TGT_DIR_NAME \
    --score_field "watermarked_texts_safety_eval" \
    --src_fields "watermarked_texts,watermarked_texts_safety_eval,watermarked_texts_unsafe_category,watermarked_texts.is_watermarked,watermarked_texts.score,watermarked_texts.pvalue" \
    --tgt_score_field "watermarked_text_safety_eval" \
    --tgt_fields "watermarked_text,watermarked_text_safety_eval,watermarked_text_unsafe_category,watermarked_text.is_watermarked,watermarked_text.score,watermarked_text.pvalue" \
    --n $N \
    --filename_pattern "*_safety_scores.jsonl" \
    --preferred_categorical_label "safe"
