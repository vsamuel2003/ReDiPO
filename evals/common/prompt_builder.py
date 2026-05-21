"""
Builds the final prompt string for a given model type.

Instruct / LoRA models:
    Uses tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).
    No system prompt is injected by this function; the template's own default is preserved.

Base models (Qwen family):
    Returns the last user message content unchanged (single-turn) or
    all user messages concatenated with double newlines (multi-turn).

Base models (all others):
    Single-turn:  "Question: {q}\\nAnswer:"
    Multi-turn:   alternating Q/A blocks, ending with "Question: {qN}\\nAnswer:"

Usage:
    messages = [{"role": "user", "content": "What is 2+2?"}]
    prompt = build_prompt(tokenizer, messages, model_type="instruct", model_path="...")
"""

from __future__ import annotations

from transformers import PreTrainedTokenizer


def _is_qwen(model_path: str) -> bool:
    return "qwen" in model_path.lower()


def _build_base_prompt(messages: list[dict], model_path: str) -> str:
    """Build a prompt for a base model that has no chat template."""
    user_turns = [m["content"] for m in messages if m["role"] == "user"]
    assistant_turns = [m["content"] for m in messages if m["role"] == "assistant"]

    if _is_qwen(model_path):
        # For Qwen base models just use the raw instruction text
        return "\n\n".join(user_turns)

    # Alternating Q/A format for all other base models
    parts = []
    for i, q in enumerate(user_turns):
        parts.append(f"Question: {q}")
        if i < len(assistant_turns):
            parts.append(f"Answer: {assistant_turns[i]}")

    parts.append("Answer:")
    return "\n".join(parts)


def build_prompt(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict],
    model_type: str,
    model_path: str,
) -> str:
    """Build the prompt string for a given model type.

    Args:
        tokenizer: The loaded tokenizer.
        messages: List of {"role": "user"|"assistant", "content": "..."}.
                  The final entry must be a user turn.
        model_type: "base", "instruct", or "lora".
        model_path: Used for Qwen detection when model_type == "base".

    Returns:
        The formatted prompt string ready for tokenization.
    """
    if model_type in ("instruct", "lora"):
        extra_kwargs = {"enable_thinking": False} if _is_qwen(model_path) else {}
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **extra_kwargs,
        )

    # Base model
    return _build_base_prompt(messages, model_path)
