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
            self.sampling_rng = torch.Generator(device=self.device)
        else:
            self.rng = torch.Generator()
            self.sampling_rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        self.sampling_rng.manual_seed(self.seed)
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

    def sample_next_beam(
        self,
        logits: torch.FloatTensor,  # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor,  # (bsz, ngram): tokens to consider when seeding
        num_beams: int,  # number of beams to consider
        temperature: float = 0.8,  # temperature for sampling
        top_p: float = 0.95,  # top p for sampling
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Sample next tokens for beam search, returning both tokens and their scores.
        Returns:
            next_tokens: shape (bsz, num_beams)
            next_scores: shape (bsz, num_beams)
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
                probs_sort, num_samples=num_beams, generator=self.sampling_rng
            )
            next_scores = torch.gather(probs_sort, -1, top_indices)
            next_tokens = torch.gather(probs_idx, -1, top_indices)
        else:
            # For greedy search, just take the top num_beams tokens
            next_scores, next_tokens = torch.topk(
                logits, num_beams, dim=-1, largest=True, sorted=True
            )

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
        ), "num_return_sequences must be <= num_beams"

        bsz = len(prompts)
        prompt_tokens = [
            self.tokenizer.encode(x, add_special_tokens=False) for x in prompts
        ]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        # Initialize beam candidates for each prompt
        # Shape: (bsz * num_beams, total_len)
        tokens = torch.full(
            (bsz * num_beams, total_len), self.pad_id, device=self.device
        ).long()

        # Copy prompt tokens for each beam
        for k, t in enumerate(prompt_tokens):
            tokens[k * num_beams : (k + 1) * num_beams, : min(len(t), total_len)] = (
                torch.tensor(t[:total_len], device=self.device)
                .long()
                .unsqueeze(0)
                .repeat(num_beams, 1)
            )

        # Track beam scores
        beam_scores = torch.zeros((bsz, num_beams), device=self.device)
        input_text_mask = tokens != self.pad_id

        # Get individual prompt lengths for each batch
        prompt_lengths = [
            (tokens[i * num_beams] != self.pad_id).sum() for i in range(bsz)
        ]

        # Only allow generation after each prompt's actual length
        for batch_idx in range(bsz):
            batch_start = batch_idx * num_beams
            batch_end = (batch_idx + 1) * num_beams
            prompt_length = prompt_lengths[batch_idx]
            input_text_mask[batch_start:batch_end, :prompt_length] = True
            input_text_mask[batch_start:batch_end, prompt_length:] = False

        start_pos = min(prompt_lengths).item()
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

                # Get next token candidates using watermarking method
                if num_beams > 1:
                    next_tokens, next_token_scores = self.sample_next_beam(
                        outputs.logits[:, -1, :],
                        ngram_tokens,
                        num_beams,
                        temperature,
                        top_p,
                    )
                else:
                    next_tokens = self.sample_next(
                        outputs.logits[:, -1, :], ngram_tokens, temperature, top_p
                    )
                    next_token_scores = torch.zeros_like(next_tokens, dtype=torch.float)

                # For each prompt, update beams
                for batch_idx in range(bsz):
                    batch_start = batch_idx * num_beams
                    batch_end = (batch_idx + 1) * num_beams

                    if num_beams > 1:
                        # Initialize beams at the end of each prompt
                        if cur_pos == prompt_lengths[batch_idx].item():
                            beam_scores[batch_idx] = next_token_scores[
                                batch_start : batch_start + num_beams, 0
                            ]
                            tokens[batch_start:batch_end, cur_pos] = next_tokens[
                                batch_start : batch_start + num_beams, 0
                            ]
                        # Only update after prompt length
                        elif cur_pos > prompt_lengths[batch_idx].item():
                            # Calculate scores for all possible next tokens
                            beam_scores_batch = (
                                beam_scores[batch_idx].unsqueeze(1)
                                + next_token_scores[batch_start:batch_end]
                            )
                            # Get top-k next tokens and their scores
                            beam_scores_flat = beam_scores_batch.view(-1)
                            top_k_scores, top_k_indices = torch.topk(
                                beam_scores_flat,
                                num_beams,
                                dim=0,
                                largest=True,
                                sorted=True,
                            )
                            beam_indices = top_k_indices // num_beams
                            token_indices = next_tokens[batch_start:batch_end].view(-1)[
                                top_k_indices % num_beams
                            ]
                            # Update beam scores
                            beam_scores[batch_idx] = top_k_scores
                            # Update tokens
                            for beam_idx in range(num_beams):
                                tokens[batch_start + beam_idx, :cur_pos] = tokens[
                                    batch_start + beam_indices[beam_idx], :cur_pos
                                ]
                                tokens[batch_start + beam_idx, cur_pos] = token_indices[
                                    beam_idx
                                ]
                        else:
                            continue
                    else:
                        # Only update tokens if not in input text
                        if not input_text_mask[batch_start, cur_pos]:
                            tokens[batch_start:batch_end, cur_pos] = next_tokens[
                                batch_start:batch_end
                            ]

                prev_pos = cur_pos

        # Prepare output sequences
        decoded_sequences = []
        for batch_idx in range(bsz):
            batch_sequences = []
            batch_start = batch_idx * num_beams

            # Get top num_return_sequences beams
            for beam_idx in range(num_return_sequences):
                t = tokens[batch_start + beam_idx].tolist()
                # Cut to max gen len
                t = t[: len(prompt_tokens[batch_idx]) + max_gen_len]
                # Cut to eos tok if any
                try:
                    t = t[: t.index(self.eos_id)]
                except ValueError:
                    pass
                batch_sequences.append(self.tokenizer.decode(t))

            decoded_sequences.append(batch_sequences)

        torch.cuda.empty_cache()
        return decoded_sequences


class OpenaiGeneratorBeam(WmGeneratorBeam):
    """Generate text using LLaMA and Aaronson's watermarking method."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_next_beam(
        self,
        logits: torch.FloatTensor,
        ngram_tokens: torch.LongTensor,
        num_beams: int,
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

            # Get top-k tokens and scores for each sequence
            next_scores, top_indices = torch.topk(
                probs_sort, num_beams, dim=-1, largest=True, sorted=True
            )
            next_tokens = torch.gather(probs_idx, -1, top_indices)
        else:
            # For greedy search, just take top-k tokens
            next_scores, next_tokens = torch.topk(
                logits, num_beams, dim=-1, largest=True, sorted=True
            )

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

    def sample_next_beam(
        self,
        logits: torch.FloatTensor,
        ngram_tokens: torch.LongTensor,
        num_beams: int,
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
                probs_sort, num_samples=num_beams, generator=self.rng
            )
            next_scores = torch.gather(probs_sort, -1, top_indices)
            next_tokens = torch.gather(probs_idx, -1, top_indices)
        else:
            # For greedy search, just take top-k tokens
            next_scores, next_tokens = torch.topk(
                logits, num_beams, dim=-1, largest=True, sorted=True
            )

        return next_tokens, next_scores
