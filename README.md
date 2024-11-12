


### Cluster Startup Setup Commands
```
module load CUDA/12.4.0
conda deactivate
conda activate ml_dev311
module load git/2.33.1

pip install git+https://github.com/yuchenlin/LLM-Blender.git
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
```


### Commands
```
### EXPERIMENT 1 (Reward Scores Gap)

bash experiment1/sweep.sh

# Run reward scorer
python run_reward_scorer.py --input_path /project/phan/av787/projs/watermarking/outputs/exp_005_sweep_temperature_hh-rlhf_1k --output_path /project/phan/av787/projs/watermarking/outputs/exp_005_sweep_temperature_hh-rlhf_1k

# Plotting
python $HOME/MyCode/_MyResearchProjects/alignmark/plots/plot_score_with_param.py --input_dir /Users/verapurv/mint/watermarking-analysis/exp_005_sweep_temperature_hh-rlhf_1k  --model_name_to_plot Meta-Llama-3.1-8B-Instruct --param_name_to_plot temperature --output_file /Users/verapurv/mint/watermarking-analysis/exp_005_sweep_temperature_hh-rlhf_1k/plots/rewards_llama_temp.pdf --score_name reward

### EXPERIMENT 2 (Truthfulness)

bash experiment2/sweep.sh

# Run truthfulness scorer
python run_truthfulness_scorer.py --input_path /project/phan/av787/projs/watermarking/outputs/exp_009_sweep_temperature_truthful_qa_gpt4omini --output_path /project/phan/av787/projs/watermarking/outputs/exp_009_sweep_temperature_truthful_qa_gpt4omini --scorer_name openai --batch_size 8

# Plotting
python plots/plot_score_with_param.py --input_dir /Users/verapurv/mint/watermarking-analysis/exp_006_sweep_temperature_truthful_qa  --model_name_to_plot Meta-Llama-3.1-8B-Instruct --param_name_to_plot temperature --output_file /Users/verapurv/mint/watermarking-analysis/exp_006_sweep_temperature_truthful_qa/plots/truthfulness_mistral_temp.pdf --score_name truthfulness


### EXPERIMENT 3 (Safety)

bash experiment3/run.sh

python run_safety_scorer.py --input_path /project/phan/av787/projs/watermarking/outputs/exp_017_safety_gpt4omini/ --output_path /project/phan/av787/projs/watermarking/outputs/exp_017_safety_gpt4omini/ --batch_size 8 --scorer_name openai

python plots/plot_bars.py --input_dir $HOME/mint/watermarking-analysis/exp_012_safety/ --output_dir $HOME/mint/watermarking-analysis/exp_016_safety_gpt4omini/plots
```
