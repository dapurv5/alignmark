#! /bin/bash

set -o errexit -o xtrace -o nounset

CLUSTER="LOCAL"  # "AWS" or "WULVER"
EXP_NAME=${1:-"exp_001_sweep_temp_hhrlhf_beam_rewards_armo_BoN"}

if [ "$CLUSTER" = "WULVER" ]; then
    EXP_DIR_PREFIX="/project/phan/av787/projs/watermarking-v1/outputs"
elif [ "$CLUSTER" = "AWS" ]; then
    EXP_DIR_PREFIX="/home/ec2-user/SageMaker/outputs/watermarking-v1"
else
    EXP_DIR_PREFIX="/Users/verapurv/mint/wm-output40GB/outputs/watermarking-v1"
fi

model_list=("Llama-3.1-8B-Instruct" "Phi-3-mini-4k-instruct")

for temperature in $(seq 0.2 0.2 1.0); do
    OUTPUT_DIR=$EXP_DIR_PREFIX/$EXP_NAME/BoNs-$temperature
    mkdir -p $OUTPUT_DIR
    python utils/mv_files.py \
        --src_dir $EXP_DIR_PREFIX/$EXP_NAME \
        --tgt_dir $OUTPUT_DIR \
        --pattern "*_temperature_${temperature}_*.jsonl"

    mkdir -p $OUTPUT_DIR/plots

    for model in ${model_list[@]}; do
        python $HOME/MyCode/_MyResearchProjects/alignmark/plots/plot_score_with_param.py \
            --input_dir $OUTPUT_DIR  \
            --model_name_to_plot $model --param_name_to_plot BoN \
            --output_file $OUTPUT_DIR/plots/rewards_${model}_bon.pdf \
            --score_name reward \
            --param_name_to_plot BoN \
            --plot_theoretical_sqrt_log
    done
done


