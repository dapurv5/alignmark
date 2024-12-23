import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from wm_detectors import MarylandDetectorZ
from wm_generators_beam import MarylandGeneratorBeam, WmGeneratorBeam

model_name = "meta-llama/Llama-3.1-8B-Instruct"
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


def infer_vocab_size(model, tokenizer):
    # Infer vocab size by passing a random text through the model
    text = "Hello, how are you?"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    vocab_size = outputs.logits.shape[-1]
    return vocab_size


# Set pad_token if it's not defined
if tokenizer.pad_token is None:
    print("Setting pad_token to eos_token")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Create an MarylandGeneratorBeam instance (assuming it takes similar parameters)
generator = MarylandGeneratorBeam(model, tokenizer, ngram=4, gamma=0.5, delta=0.2)
generator = WmGeneratorBeam(model, tokenizer)

# Sample prompt
prompt = "Explain the importance of renewable energy sources."
num_return_sequences = 5
batch_size = 16

# Generate watermarked text (assuming the generate method exists)
start_time = time.time()
watermarked_texts = generator.generate(
    [prompt] * batch_size,
    temperature=0.7,
    top_p=0.95,
    max_gen_len=200,
    num_beams=num_return_sequences,
    num_return_sequences=num_return_sequences,
)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print("Generated watermarked text:")
print(watermarked_texts)

vocab_size = infer_vocab_size(model, tokenizer)
# Create an OpenaiDetector instance
detector = MarylandDetectorZ(
    tokenizer, ngram=4, gamma=0.5, delta=0.2, vocab_size=vocab_size
)


# Custom detect method
def detect(detector, text: str, threshold=0.05):
    scores_no_aggreg = detector.get_scores_by_t([text])
    pvalues = detector.get_pvalues(scores_no_aggreg)
    scores = detector.aggregate_scores(scores_no_aggreg)

    # Assuming we're interested in the first payload (index 0)
    pvalue = pvalues[0][0]
    score = scores[0][0]

    is_watermarked = pvalue < threshold
    return {
        "is_watermarked": is_watermarked,
        "scores_no_aggreg": scores_no_aggreg,
        "pvalue": pvalue,
        "score": score,
    }


for batch_idx in range(batch_size):
    for seq_idx in range(num_return_sequences):
        detection_result = detect(detector, watermarked_texts[batch_idx][seq_idx])
        print(f"Watermark detection result for batch {batch_idx}, sequence {seq_idx}:")
        print(f"Is watermarked: {detection_result['is_watermarked']}")
        print(f"P-value: {detection_result['pvalue']}")
        print(f"Score: {detection_result['score']}")
        print("-" * 100)
print(f"Time taken: {end_time - start_time} seconds")
