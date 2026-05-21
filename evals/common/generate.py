"""
Standard generation utilities wrapping model.generate().

All evals use this module so that generation behavior is consistent.
"""

from __future__ import annotations

from typing import Optional

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


# Stop strings inserted after a model's answer to prevent base-model Q/A continuation
_BASE_STOP_STRINGS = ["\nQuestion:", "\n\nQuestion:"]


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    model_type: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    n: int = 1,
    do_sample: Optional[bool] = None,
) -> list[str]:
    """Generate completions for a single prompt.

    Args:
        model: Loaded model (base, instruct, or PeftModel).
        tokenizer: Matching tokenizer.
        prompt: Already-formatted prompt string (output of build_prompt).
        model_type: "base", "instruct", or "lora". Controls stop-string handling.
        max_new_tokens: Maximum tokens to generate per completion.
        temperature: Sampling temperature. 0.0 uses greedy decoding.
        n: Number of independent completions to produce.
        do_sample: Override for do_sample. Defaults to True when temperature > 0.

    Returns:
        List of n decoded completion strings (prompt prefix stripped).
    """
    if do_sample is None:
        do_sample = temperature > 1e-6

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs: dict = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    if n > 1:
        # Repeat the single prompt n times so model.generate returns n completions.
        # Do NOT set num_return_sequences — that would produce n² outputs on a batch.
        input_ids = inputs["input_ids"].repeat(n, 1)
        attention_mask = inputs["attention_mask"].repeat(n, 1)
    else:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    completions = []
    skip_special = model_type != "base"
    for i in range(output_ids.shape[0]):
        generated = output_ids[i][input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=skip_special)

        # Strip residual special tokens for base models
        if not skip_special:
            for tok in tokenizer.all_special_tokens:
                text = text.replace(tok, "")

        # Truncate at base-model stop strings to prevent Q/A runaway
        if model_type == "base":
            for stop in _BASE_STOP_STRINGS:
                idx = text.find(stop)
                if idx != -1:
                    text = text[:idx]

        completions.append(text.strip())

    return completions


def generate_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    model_type: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    batch_size: int = 1,
) -> list[str]:
    """Generate one completion per prompt in batches.

    Args:
        prompts: List of already-formatted prompt strings.
        batch_size: Number of prompts to process per GPU forward pass.

    Returns:
        List of completions in the same order as prompts.
    """
    completions = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(prompts), batch_size), total=total_batches, desc="Generating"):
        batch = prompts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        input_len = encoded["input_ids"].shape[1]

        do_sample = temperature > 1e-6
        gen_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                **gen_kwargs,
            )

        skip_special = model_type != "base"
        for j, out in enumerate(output_ids):
            generated = out[input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=skip_special)

            if not skip_special:
                for tok in tokenizer.all_special_tokens:
                    text = text.replace(tok, "")

            if model_type == "base":
                for stop in _BASE_STOP_STRINGS:
                    idx = text.find(stop)
                    if idx != -1:
                        text = text[:idx]

            completions.append(text.strip())

    return completions
