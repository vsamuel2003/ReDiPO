"""Filter generated responses for safety and instruction-following quality."""

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict

from openai import AsyncOpenAI
from tqdm import tqdm

# Add project root to path for scoring imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scoring.safety import score_safety, _call_judge_async
from scoring.instruction_following import (
    score_instruction_following,
    load_armorm,
    load_skywork,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(input_file: str) -> list[dict]:
    """Load entries from a jsonl file."""
    entries = []
    with open(input_file, "r") as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(entries)} entries from {input_file}")
    return entries


async def _process_safety_batch(client: AsyncOpenAI, batch: list[dict]) -> list[dict]:
    """Try concurrent batch first; on any error, fall back to sequential per-item."""
    tasks = [_call_judge_async(client, e["prompt"], e["output"]) for e in batch]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    if not any(isinstance(r, BaseException) for r in results):
        return [{"is_safe": r} for r in results]

    logger.warning(f"Batch of {len(batch)} had errors; falling back to sequential")
    out = []
    for e in batch:
        try:
            is_safe = await _call_judge_async(client, e["prompt"], e["output"])
            out.append({"is_safe": is_safe})
        except Exception as exc:
            logger.warning(f"Sequential fallback failed; filtering out entry: {exc}")
            out.append({"is_safe": False, "error": str(exc)})
    return out


async def _safety_filter_judge_async(
    entries: list[dict],
    output_file: str,
    batch_size: int = 25,
    save_interval: int = 1000,
) -> list[dict]:
    checkpoint_path = f"{output_file}.safety_checkpoint.jsonl"

    already_done = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            already_done = sum(1 for line in f if line.strip())
        logger.info(f"Resuming safety check: {already_done} entries already in checkpoint")

    pending = entries[already_done:]
    mode = "a" if already_done > 0 else "w"

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    async with AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]) as client:
        with open(checkpoint_path, mode) as cp_f, \
             tqdm(total=len(entries), initial=already_done, desc="Safety check (judge)") as pbar:
            for chunk_start in range(0, len(pending), save_interval):
                chunk = pending[chunk_start:chunk_start + save_interval]
                for batch_start in range(0, len(chunk), batch_size):
                    batch = chunk[batch_start:batch_start + batch_size]
                    results = await _process_safety_batch(client, batch)
                    for entry, result in zip(batch, results):
                        cp_f.write(json.dumps({**entry, **result}) + "\n")
                    pbar.update(len(batch))
                cp_f.flush()
                os.fsync(cp_f.fileno())
                logger.info(
                    f"Checkpoint flushed: {already_done + chunk_start + len(chunk)}/{len(entries)}"
                )

    safe_entries = []
    with open(checkpoint_path) as f:
        for line in f:
            e = json.loads(line)
            if e.get("is_safe"):
                e.pop("is_safe", None)
                e.pop("error", None)
                safe_entries.append(e)

    removed = len(entries) - len(safe_entries)
    pct = (removed / len(entries) * 100) if entries else 0
    logger.info(f"Safety filtering (judge): {removed}/{len(entries)} removed ({pct:.1f}%)")
    return safe_entries


def safety_filter(entries: list[dict], method: str, output_file: str) -> list[dict]:
    """Filter entries by safety. Returns entries that are safe."""
    if method == "judge":
        return asyncio.run(_safety_filter_judge_async(entries, output_file))

    # llama_guard: sequential, no checkpointing needed
    total = len(entries)
    safe_entries = []

    for i, entry in enumerate(entries):
        is_safe = score_safety(entry["prompt"], entry["output"], method=method)
        if is_safe:
            safe_entries.append(entry)

        if (i + 1) % 100 == 0:
            logger.info(f"  Safety check progress: {i + 1}/{total}")

    removed = total - len(safe_entries)
    pct = (removed / total * 100) if total > 0 else 0
    logger.info(f"Safety filtering ({method}): {removed}/{total} removed ({pct:.1f}%)")

    return safe_entries


def instruction_following_filter(
    entries: list[dict],
    instruct_method: str,
    scorer_method: str,
    threshold: float | None,
    armorm_model=None,
    armorm_tokenizer=None,
    skywork_model=None,
    skywork_tokenizer=None,
    skip_threshold: bool = False,
) -> list[dict]:
    """Filter entries by instruction following and assign scores.

    For judge_hard: binary pass/fail filter, then score survivors with scorer_method.
    When skip_threshold=True with judge_hard: skip binary filter, score all entries.

    For all other methods: per-prompt threshold filtering.
      - Score every entry with instruct_method.
      - For each prompt, compute the mean score of instruct-model responses.
      - Drop the entire prompt if it has no instruct-model responses.
      - Keep entries whose score >= (1 - threshold) * per_prompt_mean.
      - Assign instruction_following_score using scorer_method (reuse computed score if
        instruct_method == scorer_method).
    When skip_threshold=True: skip the per-prompt threshold cutoff; score all entries
      and attach instruction_following_score, but only drop prompts with no instruct
      responses (required for downstream min-samples filter).
    """
    total = len(entries)

    if instruct_method == "judge_hard":
        if skip_threshold:
            logger.info("Skipping judge_hard binary filter (--skip_if_filter); scoring all entries.")
            filtered_entries = entries
        else:
            filtered_entries = []
            for i, entry in enumerate(entries):
                if score_instruction_following(entry["prompt"], entry["output"], "judge_hard"):
                    filtered_entries.append(entry)
                if (i + 1) % 100 == 0:
                    logger.info(f"  IF filter progress: {i + 1}/{total}")

            removed = total - len(filtered_entries)
            pct = (removed / total * 100) if total > 0 else 0
            logger.info(f"IF filtering (judge_hard): {removed}/{total} removed ({pct:.1f}%)")

        scored_entries = []
        for i, entry in enumerate(filtered_entries):
            entry["instruction_following_score"] = score_instruction_following(
                entry["prompt"], entry["output"], scorer_method,
                armorm_model=armorm_model,
                armorm_tokenizer=armorm_tokenizer,
                skywork_model=skywork_model,
                skywork_tokenizer=skywork_tokenizer,
            )
            scored_entries.append(entry)
            if (i + 1) % 100 == 0:
                logger.info(f"  Scoring progress: {i + 1}/{len(filtered_entries)}")
        return scored_entries

    # Per-prompt threshold filtering: score everything first
    logger.info(f"Scoring {total} entries with {instruct_method} for per-prompt thresholding...")
    entry_scores: dict[int, float] = {}
    for i, entry in enumerate(entries):
        entry_scores[id(entry)] = score_instruction_following(
            entry["prompt"], entry["output"], instruct_method,
            armorm_model=armorm_model,
            armorm_tokenizer=armorm_tokenizer,
            skywork_model=skywork_model,
            skywork_tokenizer=skywork_tokenizer,
        )
        if (i + 1) % 100 == 0:
            logger.info(f"  Scoring progress: {i + 1}/{total}")

    # Group by prompt, compute per-prompt mean of instruct-model scores, apply cutoff
    prompt_groups: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        prompt_groups[entry["prompt"]].append(entry)

    filtered_entries = []
    dropped_prompts = 0
    for prompt, group in prompt_groups.items():
        instruct_scores = [
            entry_scores[id(e)] for e in group if "instruct" in e["model"].lower()
        ]
        if not instruct_scores:
            logger.info(f"Prompt '{prompt[:50]}...' dropped: no instruct-model responses")
            dropped_prompts += 1
            continue
        if skip_threshold:
            # Keep all entries for this prompt; only the no-instruct-responses guard above applies.
            filtered_entries.extend(group)
        else:
            mean_instruct = sum(instruct_scores) / len(instruct_scores)
            cutoff = (1 - threshold) * mean_instruct
            filtered_entries.extend(e for e in group if entry_scores[id(e)] >= cutoff)

    if dropped_prompts > 0:
        logger.info(f"Per-prompt filter: {dropped_prompts} prompts dropped (no instruct responses)")

    removed = total - len(filtered_entries)
    pct = (removed / total * 100) if total > 0 else 0
    if skip_threshold:
        logger.info(
            f"IF threshold skipped (--skip_if_filter); {removed}/{total} entries removed "
            f"(only prompts lacking instruct responses)"
        )
    else:
        logger.info(
            f"IF filtering ({instruct_method}, threshold={threshold}): "
            f"{removed}/{total} entries removed ({pct:.1f}%)"
        )

    # Assign instruction_following_score; reuse computed score if methods match
    scored_entries = []
    for i, entry in enumerate(filtered_entries):
        if instruct_method == scorer_method:
            entry["instruction_following_score"] = entry_scores[id(entry)]
        else:
            entry["instruction_following_score"] = score_instruction_following(
                entry["prompt"], entry["output"], scorer_method,
                armorm_model=armorm_model,
                armorm_tokenizer=armorm_tokenizer,
                skywork_model=skywork_model,
                skywork_tokenizer=skywork_tokenizer,
            )
        scored_entries.append(entry)
        if (i + 1) % 100 == 0:
            logger.info(f"  Final scoring progress: {i + 1}/{len(filtered_entries)}")

    return scored_entries


def minimum_samples_filter(
    entries: list[dict], min_samples: int = 10, min_base_samples: int = 2
) -> list[dict]:
    """Remove all entries for prompts that don't meet minimum sample requirements.

    A prompt is dropped if fewer than min_samples total responses remain, OR if fewer
    than min_base_samples base-model (non-instruct) responses remain.
    """
    prompt_groups = defaultdict(list)
    for entry in entries:
        prompt_groups[entry["prompt"]].append(entry)

    filtered = []
    removed_prompts = 0
    for prompt, group in prompt_groups.items():
        total_count = len(group)
        base_count = sum(1 for e in group if "instruct" not in e["model"].lower())

        if total_count < min_samples:
            logger.info(
                f"Prompt '{prompt[:50]}...' removed: only {total_count} total samples (< {min_samples})"
            )
            removed_prompts += 1
        elif base_count < min_base_samples:
            logger.info(
                f"Prompt '{prompt[:50]}...' removed: only {base_count} base-model samples (< {min_base_samples})"
            )
            removed_prompts += 1
        else:
            filtered.extend(group)

    if removed_prompts > 0:
        logger.info(f"Minimum samples filter: {removed_prompts} prompts removed")

    return filtered


def main():
    parser = argparse.ArgumentParser(description="Filter generated responses for safety and quality.")
    parser.add_argument("--input_file", type=str, required=True, help="Input jsonl with generations")
    parser.add_argument("--safety", type=str, required=True, choices=["llama_guard", "judge"],
                        help="Safety filtering method")
    parser.add_argument("--instruct", type=str, default="skywork",
                        choices=["judge_hard", "armorm", "judge_soft", "skywork"],
                        help="Instruction following filtering method (default: skywork)")
    parser.add_argument("--scorer", type=str, required=True,
                        choices=["armorm", "judge_soft", "skywork"],
                        help="Scorer for assigning IF scores to all passing entries")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Score threshold for armorm/judge_soft/skywork filtering (required when --instruct is not judge_hard and --skip_if_filter is not set)")
    parser.add_argument("--skip_if_filter", action="store_true", default=False,
                        help="Skip the per-prompt IF threshold step; still scores all entries and attaches instruction_following_score")
    parser.add_argument("--min_samples", type=int, default=10,
                        help="Minimum total responses per prompt after filtering (default: 10)")
    parser.add_argument("--min_base_samples", type=int, default=2,
                        help="Minimum base-model responses per prompt after filtering (default: 2)")
    parser.add_argument("--output_file", type=str, default="filtered_data.jsonl", help="Output jsonl path")
    args = parser.parse_args()

    # Validate threshold requirement
    if args.instruct != "judge_hard" and args.threshold is None and not args.skip_if_filter:
        parser.error("--threshold is required when --instruct is 'armorm', 'judge_soft', or 'skywork' (unless --skip_if_filter is set)")

    entries = load_data(args.input_file)
    original_count = len(entries)

    # Pre-load ArmoRM if needed by either instruct or scorer
    armorm_model, armorm_tokenizer = None, None
    if args.instruct == "armorm" or args.scorer == "armorm":
        logger.info("Loading ArmoRM model...")
        armorm_model, armorm_tokenizer = load_armorm()
        logger.info("ArmoRM model loaded.")

    # Pre-load Skywork if needed by either instruct or scorer
    skywork_model, skywork_tokenizer = None, None
    if args.instruct == "skywork" or args.scorer == "skywork":
        logger.info("Loading Skywork model...")
        skywork_model, skywork_tokenizer = load_skywork()
        logger.info("Skywork model loaded.")

    # Stage 1: Safety filtering
    entries = safety_filter(entries, method=args.safety, output_file=args.output_file)

    # Stage 2: Instruction following filtering + scoring
    entries = instruction_following_filter(
        entries,
        instruct_method=args.instruct,
        scorer_method=args.scorer,
        threshold=args.threshold,
        armorm_model=armorm_model,
        armorm_tokenizer=armorm_tokenizer,
        skywork_model=skywork_model,
        skywork_tokenizer=skywork_tokenizer,
        skip_threshold=args.skip_if_filter,
    )

    # Stage 3: Minimum samples check
    entries = minimum_samples_filter(
        entries, min_samples=args.min_samples, min_base_samples=args.min_base_samples
    )

    # Count unique prompts
    unique_prompts = len(set(e["prompt"] for e in entries))
    logger.info(f"Final: {len(entries)}/{original_count} entries retained across {unique_prompts} prompts")

    # Write output
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        for entry in entries:
            out = {
                "prompt": entry["prompt"],
                "output": entry["output"],
                "model": entry["model"],
                "instruction_following_score": entry["instruction_following_score"],
            }
            f.write(json.dumps(out) + "\n")

    logger.info(f"Output written to {args.output_file}")


if __name__ == "__main__":
    main()
