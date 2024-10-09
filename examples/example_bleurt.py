# python examples/example_bleurt.py

"""
Install bleurt:
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
"""

import torch
from bleurt_pytorch import (
    BleurtConfig,
    BleurtForSequenceClassification,
    BleurtTokenizer,
)

config = BleurtConfig.from_pretrained("lucadiliello/BLEURT-20-D12")
model = BleurtForSequenceClassification.from_pretrained("lucadiliello/BLEURT-20-D12")
tokenizer = BleurtTokenizer.from_pretrained("lucadiliello/BLEURT-20-D12")


model.eval()

references = [
    "a bird chirps by the window",
    "this is a random sentence",
]
candidates = [
    "a bird chirps by the window",
    "this looks like a random sentence",
]

with torch.no_grad():
    inputs = tokenizer(references, candidates, padding="longest", return_tensors="pt")
    res = model(**inputs).logits.flatten().tolist()
print(res)
