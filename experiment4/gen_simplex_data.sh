


python plots/gen_simplex_data.py \
--safety_dir $HOME/mint/wm-output40GB/outputs/watermarking-v1/exp_003_safety_beam_rewards_armo_subset_safety \
--refusal_dir $HOME/mint/wm-output40GB/outputs/watermarking-v1/exp_004_overrefusal_beam_rewards_armo_refusal \
--output_path $HOME/mint/wm-output40GB/outputs/watermarking-v1/simplex-data.tsv



python plots/gen_simplex_data.py \
--safety_dir $HOME/mint/wm-output40GB/outputs/watermarking-v1/exp_003_safety_beam_rewards_armo_subset_safety_BoN \
--refusal_dir $HOME/mint/wm-output40GB/outputs/watermarking-v1/exp_004_overrefusal_beam_rewards_armo_refusal_BoN \
--output_path $HOME/mint/wm-output40GB/outputs/watermarking-v1/simplex-data-BoN.tsv


python plots/gen_simplex_data.py \
--safety_dir $HOME/mint/wm-output40GB/outputs/watermarking-v1/exp_003_safety_beam_rewards_armo_subset_safety \
--refusal_dir $HOME/mint/wm-output40GB/outputs/watermarking-v1/exp_004_overrefusal_beam_rewards_armo_refusal \
--output_path $HOME/mint/wm-output40GB/outputs/watermarking-v1/simplex-data-normalized.tsv \
--normalize



python plots/gen_simplex_data.py \
--safety_dir $HOME/mint/wm-output40GB/outputs/watermarking-v1/exp_003_safety_beam_rewards_armo_subset_safety_BoN \
--refusal_dir $HOME/mint/wm-output40GB/outputs/watermarking-v1/exp_004_overrefusal_beam_rewards_armo_refusal_BoN \
--output_path $HOME/mint/wm-output40GB/outputs/watermarking-v1/simplex-data-BoN-normalized.tsv \
--normalize

