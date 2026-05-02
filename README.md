<div align="center">
  <img src="resources/logo.png" alt="AlignMark Logo" width="120"/>
  <h1>Watermarking Degrades Alignment in Language Models</h1>
  <p><strong>Watermarking breaks alignment. We show why, and how to undo the damage.</strong></p>

  [![Workshop](https://img.shields.io/badge/ICLR%20Workshop-2025-blue.svg)](https://openreview.net/forum?id=SIBkIV48gF)
  [![Paper](https://img.shields.io/badge/paper-OpenReview-b31b1b.svg)](https://openreview.net/forum?id=w2ATKQcfWx)
  [![Read More](https://img.shields.io/badge/read%20more-blog-8A2BE2?logo=readme&logoColor=white)](https://vermaapurv.com/2025-04-24-watermarking-alignment/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
</div>

---

## The Problem

Watermarking injects statistical signals into LLM outputs for later detection. But both watermarking and alignment training modify the same thing: the token probability distribution. They interfere. Watermarking undoes alignment.

Two things go wrong. In **guard attenuation**, the model answers harmful requests it should refuse. In **guard amplification**, it refuses harmless ones. Ask a watermarked Phi-3 how to kill a process in Python. It thinks you want to murder someone.

<div align="center">
  <img src="resources/WatermarkingDegrades_TMLR.png" alt="Watermarking degrades alignment across truthfulness, safety, and overrefusal" width="90%"/>
  <p><em>LLaMA-8B-Inst with KGW watermark (δ=2, γ=0.25). Top: unwatermarked. Middle: watermarked. Bottom: after Alignment Resampling.</em></p>
</div>

Look at the middle row. The watermarked model hallucinates about the moon landing, helps with phishing, and won't answer a factual question about Shinzo Abe. The unwatermarked model gets all three right.

## The Fix

Sample 2–4 watermarked outputs. Score them with a reward model. Return the best one. I call this Alignment Resampling. It works. Alignment returns to unwatermarked levels, sometimes exceeds them, and the watermark remains detectable.

<div align="center">
  <img src="resources/rewards_llama8b_bon.png" alt="Best-of-N recovery" width="65%"/>
  <p><em>Reward scores with Best-of-N sampling. Dashed lines are theoretical predictions.</em></p>
</div>

Why does this work? If rewards are Gaussian, alignment improves as √log(n):

$$\mathbb{E}_{\pi_w^{(n)}}[r] - \mathbb{E}_{\pi_{ref}}[r] \geq C\sqrt{\log n} - \varepsilon$$

where $\varepsilon$ is the degradation from watermarking. The bound is tight. In practice, two samples suffice.

### Why not just pick the lowest perplexity output?

I tried this. It fails. Perplexity and alignment correlate weakly. Selecting by perplexity is no better than random. Use a reward model.

## Safety Profiles

The simplex plots three response types: Safe, Unsafe, Overrefusal. Arrows show the shift from watermarking. On the left, models scatter away from safe. On the right, Best-of-4 pulls them back.

<table>
<tr>
<td align="center" width="50%">
<img src="resources/simplex_baseline.png" width="95%"/><br/>
<em>Watermarked</em>
</td>
<td align="center" width="50%">
<img src="resources/simplex_bon4.png" width="95%"/><br/>
<em>Best-of-4</em>
</td>
</tr>
</table>

Phi-3-Mini looks safer after watermarking. It isn't. It just refuses everything: 43.5% more overrefusals with Gumbel. Safety scores mean nothing if you ignore overrefusals.


---

## Running Experiments

### Setup

```bash
module load CUDA/12.4.0
conda activate ml_dev311
module load git/2.33.1

pip install git+https://github.com/yuchenlin/LLM-Blender.git
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
```

### Experiment 1: Reward Score Gap

```bash
# Watermark sweeps
bash experiment1/sweep_maryland.sh
bash experiment1/sweep_openai.sh
bash experiment1/sweep_maryland_beam.sh
bash experiment1/sweep_openai_beam.sh

# Score generation
bash experiment1/run_reward_scorer.sh
bash experiment1/run_ppl_scorer.sh

# Best-of-N sampling
bash experiment1/gen_best_of_n.sh 2
bash experiment1/gen_best_of_n.sh 3
bash experiment1/gen_best_of_n.sh 4

bash experiment1/gen_best_of_n_ppl.sh 2
bash experiment1/gen_best_of_n_ppl.sh 3
bash experiment1/gen_best_of_n_ppl.sh 4

# Plots
bash experiment1/construct_plots.sh
bash experiment1/plot_best_of_n_asym.sh exp_001_sweep_temp_hhrlhf_beam_rewards_armo_ppl_BoN
```

### Experiment 2: Truthfulness

```bash
bash experiment2/sweep.sh
bash experiment2/run_reward_scorer.sh
bash experiment2/run_truthfulness_scorer.sh
bash experiment2/construct_plots.sh
```

### Experiment 3: Safety

```bash
bash experiment3/run.sh
bash experiment3/run_reward_scorer.sh
bash experiment3/run_safety_scorer.sh
bash experiment3/construct_plots.sh
```

### Experiment 4: Overrefusal

```bash
bash experiment4/run.sh
bash experiment4/run_reward_scorer.sh
bash experiment4/run_refusal_scorer.sh
bash experiment4/gen_simplex_data.sh
bash experiment4/construct_plots.sh

# Simplex visualization
python plots/plot_simplex.py \
  --input_path $HOME/mint/watermarking-analysis/SimplexData_v1.tsv \
  --model_name "Meta-Llama-3.1-8B-Instruct"

python plots/plot_stacked_bar_chart.py \
  --input_path $HOME/mint/watermarking-analysis/SimplexData_v3.tsv \
  --output_path $HOME/mint/watermarking-analysis/simplexdata_v3_stackedbar.pdf
```

### Experiment 5: Feasibility

```bash
bash experiment1/sweep_maryland.sh
bash experiment1/sweep_openai.sh

python run_reward_scorer.py \
  --input_path /project/phan/av787/projs/watermarking/outputs/EXP_001_maryland_viability_sweep \
  --output_path /project/phan/av787/projs/watermarking/outputs/EXP_001_maryland_viability_sweep

python run_reward_scorer.py \
  --input_path /project/phan/av787/projs/watermarking/outputs/EXP_001_openai_viability_sweep \
  --output_path /project/phan/av787/projs/watermarking/outputs/EXP_001_openai_viability_sweep

bash plots/table_threshold_diff.sh
```


## Citation

```bibtex
@article{verma2026watermarking,
  title={Watermarking Degrades Alignment in Language Models: Analysis and Mitigation},
  author={Verma, Apurv and Phan, Hai and Trivedi, Shubhendu},
  journal={Transactions on Machine Learning Research},
  year={2026},
  publisher={OpenReview.net}
}

@article{verma2025watermarking,
  title={Watermarking Degrades Alignment in Language Models: Analysis and Mitigation},
  author={Verma, Apurv and Phan, NhatHai and Trivedi, Shubhendu},
  journal={arXiv preprint arXiv:2506.04462},
  year={2025}
}
```
