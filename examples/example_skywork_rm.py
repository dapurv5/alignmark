import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
device = "cuda:0"
model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
rm = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2",
    num_labels=1,
)
rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits all her apples equally among herself and her 2 siblings. How many apples does each person get?"
response1 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples."
response2 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among her 2 siblings (2 people in total). 9 ÷ 2 = 4.5 apples each. Each person gets 4 apples."

conv1 = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": response1},
]
conv2 = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": response2},
]

# Format and tokenize the conversations
# If you use `tokenize=False` with `apply_chat_template` and `tokenizer()` to tokenize the conversation,
# remeber to remove the duplicated BOS token.
conv1_tokenized = rm_tokenizer.apply_chat_template(
    conv1, tokenize=True, return_tensors="pt"
).to(device)
conv2_tokenized = rm_tokenizer.apply_chat_template(
    conv2, tokenize=True, return_tensors="pt"
).to(device)

# Get the reward scores
with torch.no_grad():
    score1 = rm(conv1_tokenized).logits[0][0].item()
    score2 = rm(conv2_tokenized).logits[0][0].item()
print(f"Score for response 1: {score1}")
print(f"Score for response 2: {score2}")

# Output:
# 27B:
# Score for response 1: 0.5625
# Score for response 2: -8.5

# 8B:
# Score for response 1: 13.6875
# Score for response 2: -9.1875
