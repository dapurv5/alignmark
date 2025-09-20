#! /bin/bash

set -o errexit -o xtrace -o nounset

CLUSTER="LOCAL"  # "AWS" or "WULVER"
N=${1:-2}
EXP_NAME=${2:-"exp_001_sweep_temp_hhrlhf_beam_rewards_armo_ppl"}
TGT_DIR_NAME=${3:-"exp_001_sweep_temp_hhrlhf_beam_rewards_armo_ppl_BoN"}
if [ "$CLUSTER" = "WULVER" ]; then
    EXP_DIR_PREFIX="/project/phan/av787/projs/watermarking-v1/outputs"
elif [ "$CLUSTER" = "AWS" ]; then
    EXP_DIR_PREFIX="/home/ec2-user/SageMaker/outputs/watermarking-v1"
else
    EXP_DIR_PREFIX="/Users/verapurv/mint/wm-output40GB/outputs/20250920/outputs/watermarking-v2"
fi

# In words, this is saying find the index of the top score in idx = watermarked_texts_reward_score[:n]
# and then overwrite the tgt_score_field with this value
# Also, overwrite tgt_associated_fields with the top score and associated fields
python generate_best_of_n.py \
    --input_dir $EXP_DIR_PREFIX/$EXP_NAME \
    --output_dir $EXP_DIR_PREFIX/$TGT_DIR_NAME \
    --score_field "watermarked_texts_ppl_score" \
    --src_fields "watermarked_texts,watermarked_texts_ppl_score,watermarked_texts_reward_score,watermarked_texts.is_watermarked,watermarked_texts.score,watermarked_texts.pvalue" \
    --tgt_score_field "watermarked_text_ppl_score" \
    --tgt_fields "watermarked_text,watermarked_text_ppl_score,watermarked_text_reward_score,watermarked_text.is_watermarked,watermarked_text.score,watermarked_text.pvalue" \
    --n $N \
    --filename_pattern "*_ppl.jsonl"

