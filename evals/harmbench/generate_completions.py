"""
Generate HarmBench completions using the unified common layer.

Uses HF transformers + PEFT only (no vLLM). Supports mid-run resume: existing
completions.json entries are matched by (behavior_id, test_case) and skipped.
Progress is checkpointed every --checkpoint-every batches.

Usage:
    python generate_completions.py \\
        --model-path meta-llama/Llama-3-8B-Instruct \\
        --model-type instruct \\
        --test-cases-path /path/to/test_cases.json \\
        --save-path results/completions.json \\
        [--lora /path/to/adapter] \\
        [--behaviors-path data/behavior_datasets/harmbench_behaviors_text_test.csv] \\
        [--max-new-tokens 256] \\
        [--batch-size 8] \\
        [--checkpoint-every 64]
"""

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.model_loader import load_model_and_tokenizer
from common.prompt_builder import build_prompt
from common.generate import generate_batch

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_existing(save_path: str) -> dict:
    """Load existing completions dict; return {} on missing or corrupt file."""
    if not os.path.exists(save_path):
        return {}
    try:
        with open(save_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return {}


def _checkpoint(data: dict, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tmp = save_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=4)
    os.replace(tmp, save_path)


def main(args):
    with open(args.test_cases_path, "r") as f:
        test_cases_data = json.load(f)

    returned_data = _load_existing(args.save_path)

    # Build set of already-completed (behavior_id, test_case_str) pairs
    done: set[tuple[str, str]] = set()
    for bid, completions in returned_data.items():
        for entry in completions:
            if entry.get("generation", ""):
                done.add((bid, entry["test_case"]))

    # Flatten only the missing items
    flat_cases = []
    for behavior_id, cases in test_cases_data.items():
        for case in cases:
            if (behavior_id, case) not in done:
                flat_cases.append({"behavior_id": behavior_id, "test_case": case})

    if done:
        print(f"Resuming: {len(done)} test cases already done, {len(flat_cases)} remaining.")

    if not flat_cases:
        print("No test cases to generate completions for.")
        return

    print(f"Generating completions for {len(flat_cases)} test cases...")

    model, tokenizer, model_type = load_model_and_tokenizer(
        model_path=args.model_path,
        model_type=args.model_type,
        lora_path=args.lora,
        dtype=torch.bfloat16,
    )

    print("Model loaded. Building prompts...")
    prompts = []
    for item in flat_cases:
        messages = [{"role": "user", "content": item["test_case"]}]
        prompt = build_prompt(
            tokenizer=tokenizer,
            messages=messages,
            model_type=model_type,
            model_path=args.model_path,
        )
        prompts.append(prompt)

    # Process in checkpoint-sized chunks
    for chunk_start in tqdm(range(0, len(flat_cases), args.checkpoint_every),
                             desc="Checkpointing", unit="chunk"):
        chunk_items = flat_cases[chunk_start: chunk_start + args.checkpoint_every]
        chunk_prompts = prompts[chunk_start: chunk_start + args.checkpoint_every]

        generations = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=chunk_prompts,
            model_type=model_type,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            batch_size=args.batch_size,
        )

        for item, generation in zip(chunk_items, generations):
            bid = item["behavior_id"]
            returned_data.setdefault(bid, [])
            returned_data[bid].append({
                "test_case": item["test_case"],
                "generation": generation,
            })

        _checkpoint(returned_data, args.save_path)

    print(f"Completions saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--model-type", type=str, default="instruct",
                        choices=["base", "instruct", "lora"])
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--test-cases-path", type=str, required=True,
                        help="Path to test_cases JSON: {behavior_id: [case, ...]}")
    parser.add_argument("--save-path", type=str, required=True,
                        help="Output path for completions JSON")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=64,
                        help="Checkpoint completions.json every N test cases")
    # Back-compat alias (behaviour is now always incremental)
    parser.add_argument("--incremental-update", action="store_true",
                        help="(Deprecated: resume is now always enabled)")
    args = parser.parse_args()

    if args.model_type == "lora" and not args.lora:
        parser.error("--lora is required when --model-type is 'lora'")

    main(args)
