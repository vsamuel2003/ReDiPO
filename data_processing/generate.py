"""Generate diverse responses from multiple models for a set of prompts."""

import argparse
import gc
import json
import logging
import os
from collections import defaultdict

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_prompts(input_file: str) -> list[str]:
    """Load prompts from a jsonl file. Each line must have a 'prompt' field."""
    prompts = []
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data["prompt"])
    logger.info(f"Loaded {len(prompts)} prompts from {input_file}")
    return prompts


def is_instruct_model(model_name: str) -> bool:
    return "instruct" in model_name.lower()


def format_prompt_str(prompt: str, tokenizer, model_name: str) -> str:
    """Format a prompt to a plain string for vLLM input."""
    if is_instruct_model(model_name):
        messages = [{"role": "user", "content": prompt}]
        if "qwen" in model_name.lower():
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        else:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    else:
        if "llama" in model_name.lower():
            return f"Question: {prompt}\nAnswer:"
        else:
            return prompt


def load_existing_outputs(output_path: str, current_model: str, k: int) -> tuple[set, list]:
    """Load existing outputs to determine what still needs generation.

    Deduplicates by keeping only the first k responses per (prompt, model) pair.
    Partial completions for current_model (< k responses) are dropped and will be regenerated.

    Returns:
        completed_set: set of prompt strings with exactly k responses for current_model
        responses_to_keep: list of entries to preserve when writing the output file
    """
    if not os.path.exists(output_path):
        return set(), []

    groups = defaultdict(list)
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            key = (entry["prompt"], entry["model"])
            if len(groups[key]) < k:
                groups[key].append(entry)

    completed_set = set()
    responses_to_keep = []

    for (prompt, model), entries in groups.items():
        if model == current_model:
            if len(entries) == k:
                completed_set.add(prompt)
                responses_to_keep.extend(entries)
            # else: partial — drop so it gets regenerated
        else:
            responses_to_keep.extend(entries)

    return completed_set, responses_to_keep


def generate_for_model(
    model_name: str,
    prompts: list[str],
    k: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    output_path: str,
    chunk_size: int = 200,
    max_model_len: int = 4096,
):
    """Load a model via vLLM, generate k responses per prompt in chunks, then offload."""
    completed_set, responses_to_keep = load_existing_outputs(output_path, model_name, k)

    prompts_to_process = [p for p in prompts if p not in completed_set]
    n_skip = len(prompts) - len(prompts_to_process)
    if n_skip > 0:
        logger.info(f"  [{model_name}] Resuming: {n_skip}/{len(prompts)} prompts already done, {len(prompts_to_process)} remaining")

    # Write the baseline (kept responses) upfront so incremental appends are safe.
    with open(output_path, "w") as f:
        for entry in responses_to_keep:
            f.write(json.dumps(entry) + "\n")

    n_new = 0

    if prompts_to_process:
        logger.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        llm = LLM(
            model=model_name,
            dtype="bfloat16",
            gpu_memory_utilization=0.92,
            trust_remote_code=True,
            max_model_len=max_model_len,
        )
        sampling_params = SamplingParams(
            n=k,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )

        # Pre-format all prompts to strings once (handles instruct vs base templating)
        formatted = [format_prompt_str(p, tokenizer, model_name) for p in prompts_to_process]

        for chunk_start in range(0, len(prompts_to_process), chunk_size):
            chunk_prompts = prompts_to_process[chunk_start:chunk_start + chunk_size]
            chunk_formatted = formatted[chunk_start:chunk_start + chunk_size]

            outputs = llm.generate(chunk_formatted, sampling_params)

            with open(output_path, "a") as f:
                for prompt, request_output in zip(chunk_prompts, outputs):
                    for completion in request_output.outputs:
                        f.write(json.dumps({"prompt": prompt, "output": completion.text.strip(), "model": model_name}) + "\n")
                        n_new += 1

            chunk_end = min(chunk_start + chunk_size, len(prompts_to_process))
            logger.info(f"  [{model_name}] Checkpoint: {chunk_end}/{len(prompts_to_process)} prompts done")

        del llm
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Offloaded model: {model_name}")

    total = len(responses_to_keep) + n_new
    logger.info(f"Finished {model_name}: {n_new} new generations, {total} total in file")


def main():
    parser = argparse.ArgumentParser(description="Generate diverse responses from multiple models.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input jsonl with prompts")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated list of HuggingFace model names")
    parser.add_argument("--k", type=int, required=True, help="Number of samples per model per prompt")
    parser.add_argument("--output_dir", type=str, default="./generated_data", help="Output directory")
    parser.add_argument("--output_file_name", type=str, default="generations.jsonl", help="Output file name")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum new tokens to generate")
    parser.add_argument("--max_model_len", type=int, default=4096, help="vLLM max model length; cap to fit GPU KV cache")
    parser.add_argument("--chunk_size", type=int, default=200, help="Prompts per vLLM batch chunk (controls checkpoint frequency)")
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file_name)

    prompts = load_prompts(args.input_file)

    logger.info(f"Generating {args.k} samples per prompt for {len(model_list)} models")
    logger.info(f"Sampling params: temperature={args.temperature}, top_p={args.top_p}, max_new_tokens={args.max_new_tokens}")

    for model_name in model_list:
        generate_for_model(
            model_name=model_name,
            prompts=prompts,
            k=args.k,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            output_path=output_path,
            chunk_size=args.chunk_size,
            max_model_len=args.max_model_len,
        )

    with open(output_path, "r") as f:
        total = sum(1 for _ in f)
    logger.info(f"Done. Total entries written: {total} -> {output_path}")


if __name__ == "__main__":
    main()
