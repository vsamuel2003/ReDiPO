"""
Generate model outputs for AlpacaEval.

Loads the tatsu-lab/alpaca_eval dataset, runs HF inference with the
unified common layer (supports base, instruct, and LoRA models), and writes
outputs.json in the format expected by alpaca_eval.evaluate().

Supports mid-run resume: if outputs.json already exists, instructions whose
"instruction" field is already present are skipped. Progress is checkpointed
every --checkpoint-every examples so a crash loses at most that many samples.

Usage:
    python generate_outputs.py \\
        --model-path meta-llama/Llama-3-8B-Instruct \\
        --model-id llama-3-8b-instruct \\
        --model-type instruct \\
        [--lora /path/to/adapter] \\
        [--output-dir results/llama-3-8b-instruct] \\
        [--max-new-tokens 2048] \\
        [--batch-size 4] \\
        [--limit 100] \\
        [--checkpoint-every 50]
"""

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.model_loader import load_model_and_tokenizer
from common.prompt_builder import build_prompt
from common.generate import generate_batch

_LLAMA_SYSTEM_PROMPT = (
    "You are a helpful, friendly, and knowledgeable assistant. "
    "Be engaging and comprehensive in your responses."
)


def _is_llama(model_path: str) -> bool:
    return "llama" in model_path.lower()


def _load_existing(output_file: str) -> tuple[list, set]:
    """Return (existing_outputs, done_instructions) from a partial outputs.json."""
    if not os.path.exists(output_file):
        return [], set()
    try:
        with open(output_file) as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            return [], set()
        done = {o["instruction"] for o in existing if "instruction" in o}
        return existing, done
    except (json.JSONDecodeError, KeyError):
        return [], set()


def _checkpoint(outputs: list, output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    tmp = output_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(outputs, f, indent=2)
    os.replace(tmp, output_file)


def main(args):
    print("Loading tatsu-lab/alpaca_eval dataset...")
    dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval_gpt4_baseline", split="eval", trust_remote_code=True)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    print(f"Loaded {len(dataset)} examples")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "outputs.json")

    outputs, done_instructions = _load_existing(output_file)
    if done_instructions:
        print(f"Resuming: {len(done_instructions)} instructions already done, "
              f"{len(dataset) - len(done_instructions)} remaining.")

    pending = [ex for ex in dataset if ex["instruction"] not in done_instructions]
    if not pending:
        print("All instructions already generated. Skipping inference.")
        return output_file

    model, tokenizer, model_type = load_model_and_tokenizer(
        model_path=args.model_path,
        model_type=args.model_type,
        lora_path=args.lora,
        dtype=torch.bfloat16,
    )

    prompts = []
    for example in pending:
        messages = [{"role": "user", "content": example["instruction"]}]
        # if model_type in ("instruct", "lora") and _is_llama(args.model_path):
        #     messages = [{"role": "system", "content": _LLAMA_SYSTEM_PROMPT}, *messages]
        prompt = build_prompt(
            tokenizer=tokenizer,
            messages=messages,
            model_type=model_type,
            model_path=args.model_path,
        )
        prompts.append(prompt)

    print(f"Running inference on {len(pending)} examples with batch_size={args.batch_size}...")

    # Process in checkpoint-sized chunks so progress survives crashes
    for chunk_start in tqdm(range(0, len(pending), args.checkpoint_every),
                             desc="Checkpointing", unit="chunk"):
        chunk_examples = pending[chunk_start: chunk_start + args.checkpoint_every]
        chunk_prompts = prompts[chunk_start: chunk_start + args.checkpoint_every]

        completions = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=chunk_prompts,
            model_type=model_type,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            batch_size=args.batch_size,
        )

        for example, completion in zip(chunk_examples, completions):
            outputs.append({
                "instruction": example["instruction"],
                "output": completion,
                "generator": args.model_id,
                "dataset": example.get("dataset", "alpaca_eval"),
            })

        _checkpoint(outputs, output_file)

    print(f"Outputs written to {output_file} ({len(outputs)} total examples)")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--model-id", type=str, required=True,
                        help="Short identifier written into the generator field")
    parser.add_argument("--model-type", type=str, default="instruct",
                        choices=["base", "instruct", "lora"])
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N examples (for debugging)")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Write outputs.json every N examples (resume granularity)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), "results", args.model_id)

    if args.model_type == "lora" and not args.lora:
        parser.error("--lora is required when --model-type is 'lora'")

    main(args)
