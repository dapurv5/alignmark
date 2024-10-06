import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from wm_detectors import OpenaiDetector
from wm_generators import WmGenerator

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = "google/gemma-2-9b-it"
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# the code does not work with above models since they do not have pad_token set
# model_name = "NousResearch/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map="auto",
    attn_implementation="flash_attention_2",
).eval()

# Set pad_token if it's not defined
if tokenizer.pad_token is None:
    print("Setting pad_token to eos_token")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Create an OpenaiGenerator instance (assuming it takes similar parameters)
generator = WmGenerator(model, tokenizer, ngram=4)

# Sample prompt
prompt = "Explain the importance of renewable energy sources."

# Generate watermarked text (assuming the generate method exists)
watermarked_texts = generator.generate(
    [prompt, prompt],
    temperature=0.7,
    top_p=0.95,
    max_gen_len=200,
)

print("Generated watermarked text:")
print(watermarked_texts[0])
print()
print(watermarked_texts[1])

# Create an OpenaiDetector instance
detector = OpenaiDetector(tokenizer, ngram=4)


# Custom detect method
def detect(detector, text: str, threshold=0.05):
    scores = detector.get_scores_by_t([text])
    pvalues = detector.get_pvalues(scores)

    # Assuming we're interested in the first payload (index 0)
    pvalue = pvalues[0][0]

    is_watermarked = pvalue < threshold
    return {
        "is_watermarked": is_watermarked,
        "scores": scores,
        "pvalue": pvalue,
    }


# Detect watermark in the generated text
detection_result = detect(detector, watermarked_texts[0])

print("Watermark detection result:")
print(f"Is watermarked: {detection_result['is_watermarked']}")
print(f"P-value: {detection_result['pvalue']}")

print("-" * 100)
detection_result = detect(detector, watermarked_texts[1])

print("Watermark detection result:")
print(f"Is watermarked: {detection_result['is_watermarked']}")
print(f"P-value: {detection_result['pvalue']}")
