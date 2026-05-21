"""Clean base model outputs using an instruct model while preserving the underlying answer."""

import argparse
import gc
import json
import logging
import os
import random

import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# SYSTEM_PROMPT = (
#     "You are an editor. You will be given a user prompt and a draft response. "
#     "Rewrite to improve the quality of the writing of the response and to remove repetitions"
#     "You MUST preserve the draft's underlying topic, subject, answer, stance, tone, and facts."
#     "Do NOT introduce new information that is not closely tied to what the draft has. "
#     "Your task is to simply improve the quality of the response while keeping the fundamental response as intact as possible, "
#     "Return ONLY the cleaned response text, nothing else."
# )

SYSTEM_PROMPT = (
    "You are an editor. You will be given a user prompt and a draft response. "
    "Rewrite the draft response the way you would write it with the following restrictions"
    "You MUST preserve the draft's underlying topic, subject, answer, stance, tone, and facts."
    "Do NOT introduce new information that is not closely tied to what the draft has. You however may introduce stylistic elements in the way you would write the same repsonse for the given prompt."
    "Your task is to simply improve the quality of the response while keeping the fundamental response as intact as possible, "
    "Return ONLY the cleaned response text, nothing else."
)


def build_messages(prompt: str, base_output: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"<prompt>\n{prompt}\n</prompt>\n\n"
                f"<draft_response>\n{base_output}\n</draft_response>"
            ),
        },
    ]


def format_cleaner_prompt(prompt: str, base_output: str, tokenizer, cleaner_model: str) -> str:
    messages = build_messages(prompt, base_output)
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if "qwen" in cleaner_model.lower():
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def load_already_done(output_file: str) -> set[tuple[str, str, str]]:
    if not os.path.exists(output_file):
        return set()
    done = set()
    with open(output_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            orig = e.get("original_output", e["output"])
            done.add((e["prompt"], orig, e["model"]))
    logger.info(f"Resuming: {len(done)} rows already in {output_file}")
    return done


def main():
    parser = argparse.ArgumentParser(description="Clean base model outputs using an instruct model.")
    parser.add_argument("--input_file", type=str, default="generated_data/subsample.jsonl")
    parser.add_argument("--output_file", type=str, default="generated_data/subset_cleans.jsonl")
    parser.add_argument("--cleaner_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--chunk_size", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    parser.add_argument("--max_model_len", type=int, default=4096, help="vLLM max model length; cap to fit GPU KV cache")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load input
    rows = []
    with open(args.input_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    logger.info(f"Loaded {len(rows)} rows from {args.input_file}")

    # Resume: track already-done by (prompt, original_output, model)
    # For base rows: key uses the original base output (before cleaning).
    # For instruct rows: key uses the instruct output directly.
    already_done = load_already_done(args.output_file)

    # Separate rows that need processing from those already done
    base_rows_pending = []
    instruct_rows_pending = []
    n_matched = 0

    for row in rows:
        is_instruct = "instruct" in row["model"].lower()
        key = (row["prompt"], row["output"], row["model"])
        if key in already_done:
            n_matched += 1
            continue
        if is_instruct:
            instruct_rows_pending.append(row)
        else:
            base_rows_pending.append(row)

    n_already = n_matched
    n_total = len(rows)
    n_pending = len(base_rows_pending) + len(instruct_rows_pending)
    logger.info(
        f"{n_already} already done, {n_pending} pending "
        f"({len(base_rows_pending)} base, {len(instruct_rows_pending)} instruct pass-through)"
    )

    # Load model only if there are base rows to clean
    tokenizer = None
    llm = None
    if base_rows_pending:
        logger.info(f"Loading cleaner model: {args.cleaner_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.cleaner_model, trust_remote_code=True)
        llm = LLM(
            model=args.cleaner_model,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            seed=args.seed,
            max_model_len=args.max_model_len,
        )
        sampling_params = SamplingParams(
            n=1,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            max_tokens=args.max_new_tokens,
        )

        # Pre-format all base prompts once
        formatted = [
            format_cleaner_prompt(row["prompt"], row["output"], tokenizer, args.cleaner_model)
            for row in base_rows_pending
        ]

    # Write output (append mode for resume)
    mode = "a" if already_done else "w"

    with open(args.output_file, mode) as out_f, \
         tqdm(total=n_total, initial=n_already, desc="Cleaning") as pbar:

        # Write instruct pass-throughs first
        for row in instruct_rows_pending:
            out_f.write(json.dumps({"prompt": row["prompt"], "output": row["output"], "model": row["model"]}) + "\n")
            out_f.flush()
            pbar.update(1)

        # Process base rows in chunks
        for chunk_start in range(0, len(base_rows_pending), args.chunk_size):
            chunk_rows = base_rows_pending[chunk_start:chunk_start + args.chunk_size]
            chunk_formatted = formatted[chunk_start:chunk_start + args.chunk_size]

            outputs = llm.generate(chunk_formatted, sampling_params)

            for row, request_output in zip(chunk_rows, outputs):
                cleaned = request_output.outputs[0].text.strip()
                entry = {
                    "prompt": row["prompt"],
                    "output": cleaned,
                    "original_output": row["output"],
                    "model": row["model"],
                }
                out_f.write(json.dumps(entry) + "\n")
                out_f.flush()
                pbar.update(1)

            chunk_end = min(chunk_start + args.chunk_size, len(base_rows_pending))
            logger.info(f"Checkpoint: {chunk_end}/{len(base_rows_pending)} base rows cleaned")

    if llm is not None:
        del llm
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"Done. Output written to {args.output_file}")


if __name__ == "__main__":
    main()
