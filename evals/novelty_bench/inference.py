"""
Novelty-bench inference using the unified common layer.

Generates num_generations responses per prompt using a local HF model
(base, instruct, or LoRA). Writes generations.jsonl in the format expected
by partition.py / score.py.

Usage:
    python inference.py \\
        --model-path Qwen/Qwen2.5-7B-Instruct \\
        --model-type instruct \\
        [--lora /path/to/adapter] \\
        --eval-dir results/qwen-7b-instruct \\
        [--data curated|wildchat] \\
        [--num-generations 10] \\
        [--max-tokens 512]
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
from common.generate import generate as gen_completions

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_FILES = {
    "curated": os.path.join(DATA_DIR, "curated.jsonl"),
    "wildchat": os.path.join(DATA_DIR, "wildchat-1k.jsonl"),
}


def load_local_dataset(data_source: str):
    path = DATA_FILES[data_source]
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def generate_for_prompt(model, tokenizer, model_type, model_path, prompt_text, num_generations, max_tokens):
    """Generate num_generations responses for a single prompt."""
    messages = [{"role": "user", "content": prompt_text}]
    formatted = build_prompt(
        tokenizer=tokenizer,
        messages=messages,
        model_type=model_type,
        model_path=model_path,
    )

    responses = gen_completions(
        model=model,
        tokenizer=tokenizer,
        prompt=formatted,
        model_type=model_type,
        max_new_tokens=max_tokens,
        temperature=1.0,
        n=num_generations,
    )
    return responses


def main(args):
    dataset = load_local_dataset(args.data)

    model, tokenizer, model_type = load_model_and_tokenizer(
        model_path=args.model_path,
        model_type=args.model_type,
        lora_path=args.lora,
        dtype=torch.bfloat16,
    )

    os.makedirs(args.eval_dir, exist_ok=True)
    output_file = os.path.join(args.eval_dir, "generations.jsonl")

    # Resume: skip already-completed items
    completed_ids: set = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if len(obj.get("generations", [])) == args.num_generations:
                        completed_ids.add(obj["id"])

    pending = [item for item in dataset if item["id"] not in completed_ids]
    if not pending:
        print("All prompts already have valid generations. Skipping.")
        return

    print(f"Generating {args.num_generations} responses each for {len(pending)} prompts...")

    with open(output_file, "a") as fout:
        for item in tqdm(pending):
            generations = generate_for_prompt(
                model=model,
                tokenizer=tokenizer,
                model_type=model_type,
                model_path=args.model_path,
                prompt_text=item["prompt"],
                num_generations=args.num_generations,
                max_tokens=args.max_tokens,
            )

            record = {
                "id": item["id"],
                "prompt": item["prompt"],
                "model": args.model_path,
                "generations": generations,
            }
            fout.write(json.dumps(record) + "\n")

    print(f"Generations written to {output_file}")

    # Release the eval model from VRAM before partition.py loads the DeBERTa classifier.
    import gc
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--model-type", type=str, default="instruct",
                        choices=["base", "instruct", "lora"])
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--eval-dir", type=str, required=True,
                        help="Directory to save generations.jsonl")
    parser.add_argument("--data", type=str, default="curated",
                        choices=["curated", "wildchat"],
                        help="Dataset to use: curated or wildchat")
    parser.add_argument("--num-generations", type=int, default=10,
                        help="Number of generations per prompt")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max new tokens per generation")
    args = parser.parse_args()

    if args.model_type == "lora" and not args.lora:
        parser.error("--lora is required when --model-type is 'lora'")

    main(args)
