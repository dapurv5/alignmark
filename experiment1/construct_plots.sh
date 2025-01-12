#! /bin/bash

set -o errexit -o xtrace -o nounset


# python $HOME/MyCode/_MyResearchProjects/alignmark/plots/plot_score_with_param.py \
#   --input_dir /Users/verapurv/mint/watermarking-analysis/exp_005_sweep_temperature_hh-rlhf_1k  \
#   --model_name_to_plot Meta-Llama-3.1-8B-Instruct --param_name_to_plot temperature \
#   --output_file /Users/verapurv/mint/watermarking-analysis/exp_005_sweep_temperature_hh-rlhf_1k/plots/rewards_llama_temp.pdf \
#   --score_name reward

model_list=("Llama-3.1-8B-Instruct" "Phi-3-mini-4k-instruct")

for model in ${model_list[@]}; do
    python $HOME/MyCode/_MyResearchProjects/alignmark/plots/plot_score_with_param.py \
        --input_dir /Users/verapurv/mint/wm-output40GB/outputs/watermarking-v1/exp_001_sweep_temp_hhrlhf_beam_rewards_armo  \
        --model_name_to_plot $model --param_name_to_plot temperature \
        --output_file /Users/verapurv/mint/wm-output40GB/outputs/watermarking-v1/exp_001_sweep_temp_hhrlhf_beam_rewards_armo/plots/rewards_${model}_temp.pdf \
        --score_name reward

    python $HOME/MyCode/_MyResearchProjects/alignmark/plots/plot_score_with_param.py \
        --input_dir /Users/verapurv/mint/wm-output40GB/outputs/watermarking-v1/exp_001_sweep_temp_hhrlhf_beam_rewards_armo_BoN  \
        --model_name_to_plot $model --param_name_to_plot temperature \
        --output_file /Users/verapurv/mint/wm-output40GB/outputs/watermarking-v1/exp_001_sweep_temp_hhrlhf_beam_rewards_armo_BoN/plots/rewards_${model}_bon.pdf \
        --score_name reward
done
