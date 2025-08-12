# Watermarking Degrades Alignment in Language Models: Analysis and Mitigation
> Mitigating alignment degradation in watermarked Large Language Models (LLMs).

<div align="center">
	<img width="200" height="200" src="fig/alignmark.png" alt="Watermarking Degrades Alignment in LLMs">
<br>
</div>


<a href="https://openreview.net/pdf?id=SIBkIV48gF">
    <img src="https://dida.do/img/containers/assets/news/image-2.png/c0ece7988eacc28a0cdab115441d63e4.png" alt="arXiv" width="100px">
</a>


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

### EXPERIMENT 1 (Reward Scores Gap)

```
bash experiment1/sweep_maryland.sh
bash experiment1/sweep_openai.sh
bash experiment1/sweep_maryland_beam.sh
bash experiment1/sweep_openai_beam.sh

# Run reward scorer
bash experiment1/run_reward_scorer.sh
bash experiment1/run_ppl_scorer.sh

# Run BoN
bash experiment1/gen_best_of_n.sh 2
bash experiment1/gen_best_of_n.sh 3
bash experiment1/gen_best_of_n.sh 4

bash experiment1/gen_best_of_n_ppl.sh 2
bash experiment1/gen_best_of_n_ppl.sh 3
bash experiment1/gen_best_of_n_ppl.sh 4

# Plotting
bash experiment1/construct_plots.sh

bash experiment1/plot_best_of_n_asym.sh exp_001_sweep_temp_hhrlhf_beam_rewards_armo_ppl_BoN  # change legend to include Perplexity manually
```

### EXPERIMENT 2 (Truthfulness)
```
bash experiment2/run_all_models.sh
bash experiment2/run_reward_scorer.sh
bash experiment2/run_truthfulness_scorer.sh
bash experiment2/construct_plots.sh
```

### EXPERIMENT 3 (Safety)
```
bash experiment3/run_all_models.sh
bash experiment3/run_reward_scorer.sh
bash experiment3/run_safety_scorer.sh
bash experiment3/construct_plots.sh
```

### EXPERIMENT 4 (Over-refusal)
```
bash experiment4/run.sh
bash experiment4/run_reward_scorer.sh
bash experiment4/run_refusal_scorer.sh
bash experiment4/gen_simplex_data.sh
bash experiment4/construct_plots.sh

python plots/plot_simplex.py --input_path $HOME/mint/watermarking-analysis/SimplexData_v1.tsv  --model_name "Meta-Llama-3.1-8B-Instruct"

python plots/plot_stacked_bar_chart.py --input_path $HOME/mint/watermarking-analysis/SimplexData_v3.tsv --output_path $HOME/mint/watermarking-analysis/simplexdata_v3_stackedbar.pdf
```

### EXPERIMENT 5 (Feasability)
```
bash experiment1/sweep_maryland.sh
bash experiment1/sweep_openai.sh

# Run reward scorer
python run_reward_scorer.py --input_path /project/phan/av787/projs/watermarking/outputs/EXP_001_maryland_viability_sweep --output_path /project/phan/av787/projs/watermarking/outputs/EXP_001_maryland_viability_sweep

python run_reward_scorer.py --input_path /project/phan/av787/projs/watermarking/outputs/EXP_001_openai_viability_sweep --output_path /project/phan/av787/projs/watermarking/outputs/EXP_001_openai_viability_sweep

bash plots/table_threshold_diff.sh
```
