"""
Generate model answers for Arena-Hard hard_prompt questions using vLLM.

Supports base, instruct, and LoRA models. Chat templates (including
enable_thinking=False for Qwen instruct/LoRA) are applied via
evals/common/prompt_builder.build_prompt before passing text to vLLM.

Usage:
    python gen_answers.py \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --model-id llama-3.1-8b-instruct \
        --model-type instruct \
        --question-file evals/arena_hard/data/question.jsonl \
        --output-file /results/arena_hard/model_answer/llama-3.1-8b-instruct.jsonl

    # LoRA:
    python gen_answers.py \
        --model-path meta-llama/Llama-3.1-8B \
        --model-id my-lora \
        --model-type lora \
        --lora /path/to/adapter \
        --question-file evals/arena_hard/data/question.jsonl \
        --output-file /results/arena_hard/model_answer/my-lora.jsonl

Resume: questions with existing (non-error) answers in --output-file are skipped.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.prompt_builder import build_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ERROR_SENTINEL = "$ERROR$"


def load_questions(path: str) -> list[dict]:
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def load_existing(output_file: str) -> tuple[set, list]:
    """Return (done_uids, rows_to_keep). Drops $ERROR$ rows so they're retried."""
    done: set = set()
    keep: list = []
    if not os.path.exists(output_file):
        return done, keep
    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if d.get("answer") != ERROR_SENTINEL:
                    done.add(d["uid"])
                    keep.append(d)
            except (json.JSONDecodeError, KeyError):
                pass
    return done, keep


def read_lora_rank(lora_path: str, default: int = 64) -> int:
    cfg = os.path.join(lora_path, "adapter_config.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            d = json.load(f)
        return d.get("r", default)
    return default


def main() -> None:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Generate Arena-Hard answers via vLLM.")
    parser.add_argument("--model-path", required=True, help="HF model ID or local path. For LoRA: the base model.")
    parser.add_argument("--model-id", required=True, help="Short identifier for output file naming.")
    parser.add_argument("--model-type", required=True, choices=["base", "instruct", "lora"])
    parser.add_argument("--lora", default="", help="Path to PEFT adapter directory (required for --model-type lora).")
    parser.add_argument("--question-file", default=str(here / "data" / "question.jsonl"))
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    args = parser.parse_args()

    if args.model_type == "lora" and not args.lora:
        parser.error("--lora is required when --model-type is lora")

    questions = load_questions(args.question_file)
    done, keep_rows = load_existing(args.output_file)

    todo = [q for q in questions if q["uid"] not in done]
    logger.info(f"Questions: {len(questions)} total | {len(done)} already done | {len(todo)} remaining")

    if not todo:
        logger.info("All answers present, skipping generation.")
        return

    # Write back the rows we're keeping (clears any $ERROR$ entries from prior crashes)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in keep_rows:
            f.write(json.dumps(row) + "\n")

    # Load tokenizer for prompt formatting
    tokenizer_source = args.model_path
    if args.lora and os.path.exists(os.path.join(args.lora, "tokenizer_config.json")):
        tokenizer_source = args.lora
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

    # Build vLLM engine
    llm_kwargs: dict = dict(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
    )
    lora_request = None
    if args.model_type == "lora":
        lora_rank = read_lora_rank(args.lora)
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = lora_rank
        lora_request = LoRARequest("adapter", 1, args.lora)
        logger.info(f"LoRA enabled: adapter={args.lora}, rank={lora_rank}")

    logger.info(f"Loading vLLM engine: {args.model_path}")
    llm = LLM(**llm_kwargs)

    stop_strings = ["\nQuestion:", "\n\nQuestion:"] if args.model_type == "base" else []
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop=stop_strings,
    )

    # Format all prompts upfront
    formatted = []
    for q in todo:
        messages = [{"role": "user", "content": q["prompt"]}]
        formatted.append(build_prompt(tokenizer, messages, args.model_type, args.model_path))

    n_new = 0
    for chunk_start in range(0, len(todo), args.chunk_size):
        chunk_qs = todo[chunk_start : chunk_start + args.chunk_size]
        chunk_prompts = formatted[chunk_start : chunk_start + args.chunk_size]

        try:
            gen_kwargs: dict = dict(prompts=chunk_prompts, sampling_params=sampling_params)
            if lora_request is not None:
                gen_kwargs["lora_request"] = lora_request
            outputs = llm.generate(**gen_kwargs)
        except Exception as e:
            logger.error(f"Generation failed for chunk {chunk_start}: {e}")
            # Write error sentinels so we can see what failed; they'll be retried next run
            with open(out_path, "a") as f:
                for q in chunk_qs:
                    f.write(json.dumps({"uid": q["uid"], "model": args.model_id,
                                        "answer": ERROR_SENTINEL, "tstamp": time.time()}) + "\n")
            continue

        with open(out_path, "a") as f:
            for q, out in zip(chunk_qs, outputs):
                answer = out.outputs[0].text.strip()
                row = {"uid": q["uid"], "model": args.model_id, "answer": answer, "tstamp": time.time()}
                f.write(json.dumps(row) + "\n")
                n_new += 1

        chunk_end = min(chunk_start + args.chunk_size, len(todo))
        logger.info(f"Checkpoint: {chunk_end}/{len(todo)} prompts done")

    del llm
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Generation complete. New: {n_new} | Total in file: {len(keep_rows) + n_new}")


if __name__ == "__main__":
    main()
