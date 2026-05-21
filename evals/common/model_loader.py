"""
Unified model and tokenizer loading for base, instruct, and LoRA models.

Usage:
    model, tokenizer, model_type = load_model_and_tokenizer(
        model_path="meta-llama/Llama-3-8B-Instruct",
        model_type="instruct",
    )

    # With LoRA:
    model, tokenizer, model_type = load_model_and_tokenizer(
        model_path="meta-llama/Llama-3-8B-Instruct",
        model_type="lora",
        lora_path="/path/to/lora/adapter",
    )
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def load_model_and_tokenizer(
    model_path: str,
    model_type: str,
    lora_path: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    trust_remote_code: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizer, str]:
    """Load a HuggingFace model with optional LoRA adapter.

    Args:
        model_path: HF model ID or local path. For LoRA, this is the BASE model.
        model_type: One of "base", "instruct", or "lora".
        lora_path: Path to PEFT adapter directory (required when model_type == "lora").
        dtype: torch dtype for the model weights.
        device_map: Passed to from_pretrained; "auto" distributes across GPUs.
        trust_remote_code: Passed to both model and tokenizer loading.

    Returns:
        (model, tokenizer, model_type) — model_type is echoed back for convenience.
    """
    if model_type not in ("base", "instruct", "lora"):
        raise ValueError(f"model_type must be 'base', 'instruct', or 'lora'; got {model_type!r}")

    if model_type == "lora" and not lora_path:
        raise ValueError("lora_path must be provided when model_type == 'lora'")

    # Load tokenizer: prefer the LoRA dir's tokenizer if it exists, else use base model
    tokenizer_source = model_path
    if lora_path and os.path.exists(os.path.join(lora_path, "tokenizer_config.json")):
        tokenizer_source = lora_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=trust_remote_code,
        padding_side="left",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2",
        )
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

    # Attach LoRA adapter
    if model_type == "lora":
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return model, tokenizer, model_type
