import random

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)
# Construct a random 1D tensor of size 10
logits = torch.rand(10)
temperature = 0.8

# Take softmax of the tensor
probs = torch.softmax(logits / temperature, dim=-1)
probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
top_indices = torch.multinomial(probs_sort, num_samples=3)
next_scores = torch.gather(probs_sort, -1, top_indices)
next_tokens = torch.gather(probs_idx, -1, top_indices)

print(next_scores)
print(next_tokens)


top_indices = torch.multinomial(probs_sort, num_samples=3)
next_scores = torch.gather(probs_sort, -1, top_indices)
next_tokens = torch.gather(probs_idx, -1, top_indices)

print(next_scores)
print(next_tokens)
