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
# Construct a random 2D tensor of size 10 where 2 is the batch size
logits = torch.rand(10)
batch_size = 2
logits = logits.repeat(batch_size, 1)
print("logits\n", logits)
temperature = 0.8
generator = torch.Generator()
generator.manual_seed(42)
# Take softmax of the tensor
probs = torch.softmax(logits / temperature, dim=-1)
print("probs\n", probs)
probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
print("probs_sort\n", probs_sort)
print("probs_idx\n", probs_idx)
top_indices = torch.multinomial(probs_sort, num_samples=3, generator=generator)
print("top_indices\n", top_indices)
next_scores = torch.gather(probs_sort, -1, top_indices)
next_tokens = torch.gather(probs_idx, -1, top_indices)

print("next_scores\n", next_scores)
print("next_tokens\n", next_tokens)


top_indices = torch.multinomial(probs_sort, num_samples=3)
next_scores = torch.gather(probs_sort, -1, top_indices)
next_tokens = torch.gather(probs_idx, -1, top_indices)

print("next_scores\n", next_scores)
print("next_tokens\n", next_tokens)
