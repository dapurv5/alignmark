


### Cluster Startup Setup Commands
```
export PYTHONPATH=$HOME/anahata-python3
export PYTHONPATH=$HOME/anahata-python3/reference_projects/2024/MarkLLM:$PYTHONPATH

module load CUDA/12.4.0
conda deactivate
conda activate ml_dev311
module load git/2.33.1

pip install git+https://github.com/yuchenlin/LLM-Blender.git
```


### Commands Exp-1 (Reward Score Gap)
This experiment uses the MarkLLM implementation.

```
cd markllm
bash sweep.sh

# Run reward scorer
python run_reward_scorer.py --input_path /project/phan/av787/projs/watermarking/outputs/exp_003_sweep_temperature_hh-rlhf_1k --output_path /project/phan/av787/projs/watermarking/outputs/exp_003_sweep_temperature_hh-rlhf_1k

# Plotting
python $HOME/MyCode/anahata/anahata-python3/anahata/play/ml_research/watermark-analysis/plots/plot_reward_diff_with_wm_strength.py --input_dir $HOME/mint/watermarking-analysis/exp_001_sweep_delta_hh-rlhf_1k  --watermark_type_to_plot KGW --wm_strength_param_name_to_plot delta --output_file $HOME/mint/watermarking-analysis/exp_001_sweep_delta_hh-rlhf_1k/plots/rewards_gap_kgw_delta.pdf

```