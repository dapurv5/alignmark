from typing import List

import torch
from vllm import LLM, SamplingParams
from watermark.base import BaseWatermark


class VLLMLogitsProcessorWrapper:
    """
    A wrapper around a logits processor that converts input_ids and scores to
    tensors with shape (1, vocab_size) before calling the original logits
    processor.
    """

    def __init__(self, logits_processor):
        self.logits_processor = logits_processor

    def __call__(self, input_ids: List[int], scores: List[float]) -> List[float]:
        """
        Process the logits.
        Args:
            input_ids (List[int]): The input IDs.
            scores (List[float]): The scores.
        Returns:
            List[float]: The processed scores.
        """
        # Convert input_ids to a tensor with shape (1, len(input_ids))
        input_ids_tensor = torch.tensor(input_ids).unsqueeze(0)

        # Convert scores to a tensor with shape (1, vocab_size)
        scores_tensor = torch.tensor(scores).unsqueeze(0)

        # Call the original logits processor
        processed_scores = self.logits_processor(input_ids_tensor, scores_tensor)

        # Return the processed scores as a tensor
        return processed_scores.squeeze(0)


def generate(prompt, llm_, logits_processor=None, **kwargs):
    """
    Generate text using a VLLM model.
    Args:
        prompt (str): The prompt to generate text from.
        llm_ (vllm.LLM): The LLM model to use.
        logits_processor (callable, optional): A function that processes the
            logits. Defaults to None.
        kwargs (dict, optional): Additional keyword arguments. Defaults to {}.
    Returns:
        str: The generated text.
    """
    vllm_params = {}
    if logits_processor is not None:
        vllm_params["logits_processors"] = [
            VLLMLogitsProcessorWrapper(logits_processor)
        ]
    sampling_params = SamplingParams(
        temperature=kwargs.get("temperature", 0.7),
        top_p=kwargs.get("top_p", 0.95),
        max_tokens=kwargs.get("max_new_tokens", 500),
        **vllm_params,
    )
    outputs = llm_.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text


def patch_watermark(watermark: BaseWatermark, llm: LLM):
    """
    A wrapper around a watermark that uses VLLM to generate text.
    """
    # Not all watermarks have a logits_processor. e.g. KGW has it but EXP does not.
    logits_processor = getattr(watermark, "logits_processor", None)
    # In the processors, gen_kwargs are passed in the constructor.
    # So no need to pass them here.
    if logits_processor is not None:
        watermark.generate_watermarked_text = lambda prompt: generate(
            prompt, llm, logits_processor, **watermark.config.gen_kwargs
        )
    else:
        raise NotImplementedError(
            "This codepath does not work currently for EXP watermark"
        )
    watermark.generate_unwatermarked_text = lambda prompt: generate(
        prompt, llm, **watermark.config.gen_kwargs
    )
    return watermark
