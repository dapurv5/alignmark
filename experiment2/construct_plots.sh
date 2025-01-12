

list_of_models=("Llama-3.1-8B-Instruct" \
"Phi-3-mini-4k-instruct" \
"Mistral-7B-Instruct-v0.3" \
"Qwen2.5-7B-Instruct" \
"Qwen2.5-1.5B-Instruct" \
"Qwen2.5-3B-Instruct" \
"Llama-3.2-1B-Instruct" \
"Llama-3.2-3B-Instruct" \
"Qwen2.5-0.5B-Instruct" \
"Qwen2.5-14B-Instruct" )

# python $HOME/MyCode/_MyResearchProjects/alignmark/plots/plot_score_with_param.py \
#     --input_dir /Users/verapurv/mint/watermarking-analysis/exp_006_sweep_temperature_truthful_qa  \
#     --model_name_to_plot Meta-Llama-3.1-8B-Instruct \
#     --param_name_to_plot temperature \
#     --output_file /Users/verapurv/mint/watermarking-analysis/exp_006_sweep_temperature_truthful_qa/plots/truthfulness_llama8b_temp.pdf \
#     --score_name truthfulness


# for model in "${list_of_models[@]}"; do
#     # Plot based on reward scores instead of truthfulness scores
#     python $HOME/MyCode/_MyResearchProjects/alignmark/plots/plot_score_with_param.py \
#         --input_dir $HOME/mint/wm-output40GB/outputs/watermarking-v1/exp_002_sweep_temp_truthfulqa_beam_rewards_armo  \
#         --model_name_to_plot $model \
#         --param_name_to_plot temperature \
#         --output_file $HOME/mint/wm-output40GB/outputs/watermarking-v1/exp_002_sweep_temp_truthfulqa_beam_rewards_armo/plots/reward_${model}_temp.pdf \
#         --score_name reward
# done

# python $HOME/MyCode/_MyResearchProjects/alignmark/plots/plot_score_with_param.py \
#     --input_dir $HOME/mint/wm-output40GB/outputs/watermarking-v1/exp_002_sweep_temp_truthfulqa_beam_truthfulness  \
#     --model_name_to_plot Llama-3.1-8B-Instruct \
#     --param_name_to_plot temperature \
#     --output_file $HOME/mint/wm-output40GB/outputs/watermarking-v1/exp_002_sweep_temp_truthfulqa_beam_truthfulness/plots/truthfulness_llama8b_temp.pdf \
#     --score_name truthfulness

python $HOME/MyCode/_MyResearchProjects/alignmark/plots/plot_score_bars.py \
    --input_dir $HOME/mint/wm-output40GB/outputs/watermarking-v1/exp_002_sweep_temp_truthfulqa_beam_rewards_armo_subset_truthfulness  \
    --output_dir $HOME/mint/wm-output40GB/outputs/watermarking-v1/exp_002_sweep_temp_truthfulqa_beam_rewards_armo_subset_truthfulness/plots \
    --score_name truthfulness

python $HOME/MyCode/_MyResearchProjects/alignmark/plots/plot_score_bars.py \
    --input_dir $HOME/mint/wm-output40GB/outputs/watermarking-v1/exp_002_sweep_temp_truthfulqa_beam_rewards_armo_subset_truthfulness_BoN  \
    --output_dir $HOME/mint/wm-output40GB/outputs/watermarking-v1/exp_002_sweep_temp_truthfulqa_beam_rewards_armo_subset_truthfulness_BoN/plots \
    --score_name truthfulness