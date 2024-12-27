from typing import List, Tuple

import torch


class WmGeneratorBeam:
    def __init__(
        self,
        model,
        tokenizer,
        ngram: int = 1,
        seed: int = 0,
        seeding: str = "hash",
        salt_key: int = 35317,
        payload: int = 0,
    ):
        # model config
        self.tokenizer = tokenizer
        self.model = model
        self.max_seq_len = 1024
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        # watermark config
        self.ngram = ngram
        self.salt_key = salt_key
        self.seed = seed
        self.hashtable = torch.randperm(1000003)
        self.seeding = seeding
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        self.payload = payload

        # Move hashtable to GPU if available
        self.device = self.model.device
        # Move RNG to GPU if CUDA is available
        if torch.cuda.is_available():
            self.rng = torch.Generator(device=self.device)
        else:
            self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        self.hashtable = torch.randperm(1000003).to(self.device)

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Optimized hashint using GPU."""
        return self.hashtable[integer_tensor % len(self.hashtable)]

    def get_seed_rng(self, input_ids: torch.LongTensor) -> int:
        """Seed RNG with hash of input_ids."""
        if self.seeding == "hash":
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.salt_key + i.item()) % (2**64 - 1)
        elif self.seeding == "additive":
            seed = self.salt_key * torch.sum(input_ids).item()
            seed = self.hashint(seed)
        elif self.seeding == "skip":
            seed = self.salt_key * input_ids[0].item()
            seed = self.hashint(seed)
        elif self.seeding == "min":
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed).item()
        return seed

    def sample_next(
        self,
        logits: torch.FloatTensor,  # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor,  # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8,  # temperature for sampling
        top_p: float = 0.95,  # top p for sampling
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Sample next tokens for beam search, returning both tokens and their scores.
        Returns:
            next_tokens: shape (bsz * num_beams)
            next_scores: shape (bsz * num_beams)
        """
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

            # Sample num_beams indices from multinomial distribution of next_scores
            # Note that when temperature > 0, the beam search is not monotonically decreasing
            top_indices = torch.multinomial(
                probs_sort, num_samples=1, generator=self.rng
            )
            next_scores = torch.gather(probs_sort, -1, top_indices)
            next_tokens = torch.gather(probs_idx, -1, top_indices)
        else:
            # For greedy search, just take the top num_beams tokens
            next_scores, next_tokens = torch.argmax(logits, dim=-1)

        return next_tokens, next_scores

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        num_beams: int = 1,
        num_return_sequences: int = 1,
    ) -> List[List[str]]:
        """
        Generate text from prompts using beam search.
        Returns num_return_sequences candidates for each prompt.
        """
        assert (
            num_return_sequences <= num_beams
        ), f"num_return_sequences ({num_return_sequences}) must be less than or equal to num_beams ({num_beams})"

        bsz = len(prompts)
        prompt_tokens = [
            self.tokenizer.encode(x, add_special_tokens=False) for x in prompts
        ]

        # Truncate prompts that are too long
        prompt_tokens = [t[: self.max_seq_len] for t in prompt_tokens]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        # Initialize tokens and scores
        tokens = torch.full(
            (bsz * num_beams, total_len), self.pad_id, device=self.device
        ).long()
        beam_scores = torch.zeros((bsz, num_beams), device=self.device)

        # Copy input prompts
        for k, t in enumerate(prompt_tokens):
            for beam in range(num_beams):
                beam_idx = k * num_beams + beam
                tokens[beam_idx, : len(t)] = torch.tensor(t, device=self.device).long()

        input_text_mask = tokens != self.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        outputs = None

        with torch.amp.autocast("cuda"):
            for cur_pos in range(start_pos, total_len):
                outputs = self.model.forward(
                    tokens[:, prev_pos:cur_pos],
                    use_cache=True,
                    past_key_values=outputs.past_key_values if prev_pos > 0 else None,
                )
                ngram_tokens = tokens[:, cur_pos - self.ngram : cur_pos]

                # Get next token probabilities and select top-k
                next_token_logits = outputs.logits[:, -1, :]
                next_tokens, next_scores = self.sample_next(
                    next_token_logits,
                    ngram_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                # Update beam scores with log probabilities
                beam_scores = beam_scores.view(-1) + torch.log(next_scores.view(-1))
                beam_scores = beam_scores.view(bsz, num_beams)

                # Change from (bsz * num_beams, 1) to (bsz * num_beams,)
                next_tokens = next_tokens.view(-1)
                # Only update tokens where we're not in the input text
                tokens[:, cur_pos] = torch.where(
                    input_text_mask[:, cur_pos],
                    tokens[:, cur_pos],  # keep original tokens
                    next_tokens,  # use predicted tokens
                )
                prev_pos = cur_pos

        # Process results
        decoded = []
        for i in range(bsz):
            beam_outputs = []
            beam_scores_i = beam_scores[i]
            ordered_beams = torch.argsort(beam_scores_i, descending=True)[
                :num_return_sequences
            ]

            for beam_idx in ordered_beams:
                t = tokens[i * num_beams + beam_idx]
                # Truncate to actual length
                t = t[: len(prompt_tokens[i]) + max_gen_len]
                # Find EOS if present
                eos_indices = (t == self.eos_id).nonzero()
                if len(eos_indices) > 0:
                    t = t[: eos_indices[0]]
                beam_outputs.append(self.tokenizer.decode(t.cpu().tolist()))
            decoded.append(beam_outputs)

        return decoded


class OpenaiGeneratorBeam(WmGeneratorBeam):
    """Generate text using LLaMA and Aaronson's watermarking method."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_next(
        self,
        logits: torch.FloatTensor,
        ngram_tokens: torch.LongTensor,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Beam search version of Aaronson's watermarking method.
        Returns top-k tokens and their scores for each sequence.
        """
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

            # Process each sequence in batch
            for ii in range(ngram_tokens.shape[0]):
                seed = self.get_seed_rng(ngram_tokens[ii])
                self.rng.manual_seed(seed)

                # Generate random values and apply payload shift
                vocab_size = logits.shape[-1]
                rs = torch.rand(
                    vocab_size, generator=self.rng, device=probs_sort.device
                )
                rs = rs.roll(-self.payload)
                rs = rs[probs_idx[ii]]

                # Modified watermarking score for beam search
                probs_sort[ii] = torch.pow(rs, 1 / probs_sort[ii])

            # Sample from multinomial ( r^(1/p) ) instead of argmax ( r^(1/p) )
            next_tokens = torch.multinomial(
                probs_sort, num_samples=1, generator=self.rng
            )
            next_scores = torch.gather(probs_sort, -1, next_tokens)
            next_tokens = torch.gather(probs_idx, -1, next_tokens)
        else:
            # For greedy search, get top token and score
            next_scores, next_tokens = torch.max(logits, dim=-1, keepdim=True)

        return next_tokens, next_scores


class MarylandGeneratorBeam(WmGeneratorBeam):
    """Generate text using LLaMA and Maryland's watermarking method."""

    def __init__(self, *args, gamma: float = 0.5, delta: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.delta = delta

    def logits_processor(self, logits, ngram_tokens):
        """Process logits to mask out words in greenlist."""
        bsz, vocab_size = logits.shape
        logits = logits.clone()
        for ii in range(ngram_tokens.shape[0]):  # batch of texts
            seed = self.get_seed_rng(ngram_tokens[ii])
            self.rng.manual_seed(seed)
            vocab_permutation = torch.randperm(
                vocab_size, generator=self.rng, device=self.device
            )
            greenlist = vocab_permutation[: int(self.gamma * vocab_size)]  # gamma * n
            bias = torch.zeros(vocab_size, device=self.device)  # n
            bias[greenlist] = self.delta
            bias = bias.roll(-self.payload)
            logits[ii] += bias  # add bias to greenlist words
        return logits

    def sample_next(
        self,
        logits: torch.FloatTensor,
        ngram_tokens: torch.LongTensor,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Beam search version of Maryland's watermarking method.
        Returns top-k tokens and their scores for each sequence.
        """
        # Apply Maryland's watermarking by adding bias to greenlist tokens
        logits = self.logits_processor(logits, ngram_tokens)

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

            # Sample num_beams indices from multinomial distribution of next_scores
            # Note that when temperature > 0, the beam search is not monotonically decreasing
            top_indices = torch.multinomial(
                probs_sort, num_samples=1, generator=self.rng
            )
            next_scores = torch.gather(probs_sort, -1, top_indices)
            next_tokens = torch.gather(probs_idx, -1, top_indices)
        else:
            next_scores, next_tokens = torch.argmax(logits, dim=-1)

        return next_tokens, next_scores
