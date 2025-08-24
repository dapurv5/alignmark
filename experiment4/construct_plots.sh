

## OLD EXPERIMENTS

# python plots/plot_simplex.py \
# --input_path $HOME/mint/watermarking-analysis/SimplexData_v1.tsv  \
# --model_name "Meta-Llama-3.1-8B-Instruct"

# python plots/plot_stacked_bar_chart.py \
# --input_path $HOME/mint/watermarking-analysis/SimplexData_v3.tsv \
# --output_path $HOME/mint/watermarking-analysis/simplexdata_v3_stackedbar.pdf


## NEW EXPERIMENTS

# Simplex plot

# Without BoN
# python plots/plot_simplex.py \
# --input_path $HOME/mint/wm-output40GB/outputs/watermarking-v1/simplex-data-normalized.tsv \
# --model_name "Phi-3-mini-4k-instruct"

# With BoN
python plots/plot_simplex.py \
--input_path $HOME/mint/wm-output40GB/outputs/watermarking-v2/simplex-data-BoN-4-normalized.tsv \
--model_name "Phi-3-mini-4k-instruct"
# --model_name "Qwen2.5-7B-Instruct"



# Stacked bar chart

# Without BoN (but remember here you did multinomial sampling for Gumbel)

# python plots/plot_stacked_bar_chart.py \
# --input_path $HOME/mint/wm-output40GB/outputs/watermarking-v2/simplex-data.tsv \
# --output_path $HOME/mint/wm-output40GB/outputs/watermarking-v2/simplexdata_stackedbar.pdf

# With BoN
# python plots/plot_stacked_bar_chart.py \
# --input_path $HOME/mint/wm-output40GB/outputs/watermarking-v2/simplex-data-BoN.tsv \
# --output_path $HOME/mint/wm-output40GB/outputs/watermarking-v2/simplexdata_BoN_stackedbar.pdf
