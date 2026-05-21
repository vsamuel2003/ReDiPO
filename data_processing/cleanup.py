"""Clean up generated responses using GPT-5.4-nano before filtering."""

import argparse
import asyncio
import json
import logging
import os
import re

from openai import AsyncOpenAI, BadRequestError
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy internal retry logs from the OpenAI SDK and httpx transport
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

MODEL = "gpt-5.4-nano"
MAX_RETRIES = 3

CLEANUP_SYSTEM_PROMPT = """You are a text cleanup assistant. You will receive a piece of text.

If the input contains unsafe or policy-violating content, do not modify it and return it unchanged with "changed": false.

Your job is to make MINIMAL corrections ONLY when necessary. You must NEVER add words or phrases that are not already in the original text.

You may ONLY perform these three types of corrections:
1. DELETE text at the very beginning or very end if the generation is cut off mid-sentence or does not start at a natural sentence boundary. Do NOT add new text to complete sentences — only delete the partial or broken parts.
2. Fix punctuation errors and delete any leading whitespace.
3. Fix clear grammatical errors in the existing words.

CRITICAL RULES:
- Do NOT add any new words, phrases, or sentences that are not in the original text.
- Do NOT rephrase, paraphrase, or reword anything.
- Do NOT expand abbreviations or add explanations.
- If the text needs no changes, return it exactly as-is with changed set to false.
- Keep your output as close to the original as possible.
- Only make changes when they are clearly necessary.

Respond with valid JSON in this exact format and nothing else:
{"changed": <true or false>, "text": "<the cleaned text>"}

The "changed" field must be true if you modified anything, false if the text is returned unchanged."""


VALID_END_PUNCT = (".", "!", "?")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def rule_based_cleanup(text: str) -> tuple[bool, str]:
    original = text
    stripped = text.strip()
    if not stripped:
        return (stripped != original, stripped)

    sentences = SENTENCE_SPLIT_RE.split(stripped)

    while sentences:
        first_char = sentences[0].lstrip()[:1]
        if first_char.isupper() or first_char.isdigit():
            break
        sentences.pop(0)

    while sentences and not sentences[-1].rstrip().endswith(VALID_END_PUNCT):
        sentences.pop()

    cleaned = " ".join(sentences)
    return (cleaned != original, cleaned)


def run_pipeline_rule(
    entries: list[dict],
    out_f,
    diag_f,
    already_done: int,
    total_entries: int,
    save_interval: int,
) -> tuple[int, int]:
    total_changed = 0
    total_processed = already_done
    buffer_since_flush = 0
    with tqdm(total=len(entries), desc="Cleaning up responses", unit="entry") as pbar:
        for entry in entries:
            original = entry["output"]
            changed, cleaned = rule_based_cleanup(original)
            out_f.write(json.dumps({
                "prompt": entry["prompt"],
                "output": cleaned,
                "model": entry["model"],
            }) + "\n")
            if changed:
                diag_f.write(json.dumps({
                    "prompt": entry["prompt"],
                    "model": entry["model"],
                    "original_output": original,
                    "cleaned_output": cleaned,
                }) + "\n")
                total_changed += 1
            total_processed += 1
            buffer_since_flush += 1
            pbar.update(1)
            if buffer_since_flush >= save_interval:
                out_f.flush()
                diag_f.flush()
                logger.info(f"Checkpoint: saved {total_processed}/{total_entries} entries")
                buffer_since_flush = 0
        out_f.flush()
        diag_f.flush()
    return total_changed, total_processed


def _parse_response(raw: str, original_text: str) -> tuple[bool, str]:
    """Parse the JSON response from GPT. Returns (changed, cleaned_text).

    Strips markdown code fences if present before parsing.
    Raises json.JSONDecodeError if parsing fails.
    """
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    parsed = json.loads(raw)
    changed = bool(parsed.get("changed", False))
    cleaned_text = parsed.get("text", original_text)
    return changed, cleaned_text


async def cleanup_single_async(async_client: AsyncOpenAI, entry: dict) -> dict:
    """Clean up a single entry asynchronously.

    Catches BadRequestError (400) immediately — no retry.
    Does NOT catch general exceptions — lets them propagate for batch-level fallback detection.

    Returns the entry dict augmented with: changed (bool), original (str), cleaned (str).
    """
    text = entry["output"]

    try:
        completion = await async_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
                {"role": "user", "content": f"Here is the text to clean up:\n\n---\n{text}\n---"},
            ],
            temperature=0,
            timeout=30.0,
        )
        raw = completion.choices[0].message.content.strip()
        changed, cleaned_text = _parse_response(raw, text)
        return {**entry, "changed": changed, "original": text, "cleaned": cleaned_text}

    except BadRequestError as e:
        logger.warning(f"HTTP 400 error — skipping cleanup for this entry: {e}")
        return {**entry, "changed": False, "original": text, "cleaned": text}


async def process_batch(async_client: AsyncOpenAI, batch: list[dict]) -> list[dict]:
    """Process a batch of entries concurrently.

    If any task raises an exception, the entire batch is treated as unchanged
    (original text returned for all entries) and processing continues.

    Returns a list of result dicts, one per entry.
    """
    tasks = [cleanup_single_async(async_client, entry) for entry in batch]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    errors = [r for r in raw_results if isinstance(r, BaseException)]
    if errors:
        logger.warning(
            f"Batch of {len(batch)} entries had {len(errors)} error(s) "
            f"(e.g. {errors[0]}). Treating entire batch as unchanged."
        )
        return [{**e, "changed": False, "original": e["output"], "cleaned": e["output"]} for e in batch]

    return list(raw_results)


async def process_all(async_client: AsyncOpenAI, entries: list[dict], batch_size: int) -> tuple[list[dict], list[dict]]:
    """Process all entries in concurrent batches. Errored batches are skipped (entries unchanged).

    Returns:
        cleaned_entries: All entries with output replaced by cleaned text (same schema as input).
        diagnostics_entries: Only entries where cleanup made a change.
    """
    batches = [entries[i:i + batch_size] for i in range(0, len(entries), batch_size)]

    all_results = []
    with tqdm(total=len(entries), desc="Cleaning up responses", unit="entry") as pbar:
        for batch in batches:
            batch_results = await process_batch(async_client, batch)
            all_results.extend(batch_results)
            pbar.update(len(batch))

    cleaned_entries = []
    diagnostics_entries = []
    for result in all_results:
        cleaned_entries.append({
            "prompt": result["prompt"],
            "output": result["cleaned"],
            "model": result["model"],
        })
        if result["changed"]:
            diagnostics_entries.append({
                "prompt": result["prompt"],
                "model": result["model"],
                "original_output": result["original"],
                "cleaned_output": result["cleaned"],
            })

    return cleaned_entries, diagnostics_entries


async def run_pipeline(
    entries: list[dict],
    batch_size: int,
    out_f,
    diag_f,
    already_done: int,
    total_entries: int,
    save_interval: int,
) -> tuple[int, int]:
    """Run the full cleanup pipeline under a single event loop with one shared client."""
    async with AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]) as async_client:
        total_changed = 0
        total_processed = already_done
        for chunk_start in range(0, len(entries), save_interval):
            chunk = entries[chunk_start:chunk_start + save_interval]
            cleaned_entries, diagnostics_entries = await process_all(async_client, chunk, batch_size)
            for entry in cleaned_entries:
                out_f.write(json.dumps(entry) + "\n")
            out_f.flush()
            for entry in diagnostics_entries:
                diag_f.write(json.dumps(entry) + "\n")
            diag_f.flush()
            total_changed += len(diagnostics_entries)
            total_processed += len(cleaned_entries)
            logger.info(f"Checkpoint: saved {total_processed}/{total_entries} entries")
        return total_changed, total_processed


def load_data(input_file: str) -> list[dict]:
    """Load entries from a jsonl file."""
    entries = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    logger.info(f"Loaded {len(entries)} entries from {input_file}")
    return entries


def main():
    parser = argparse.ArgumentParser(description="Clean up generated responses using GPT-5.4-nano.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL with generations")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL with cleaned responses")
    parser.add_argument("--diagnostics_file", type=str, required=True, help="Output JSONL with changed entries only")
    parser.add_argument("--batch_size", type=int, default=25, help="Number of entries per concurrent batch (default: 25)")
    parser.add_argument("--mode", choices=["llm", "rule"], default="rule", help="Cleanup mode: rule (deterministic, no API calls) or llm (GPT judge)")
    args = parser.parse_args()

    entries = load_data(args.input_file)

    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.diagnostics_file) if os.path.dirname(args.diagnostics_file) else ".", exist_ok=True)

    already_done = 0
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            already_done = sum(1 for line in f if line.strip())
        if already_done:
            logger.info(f"Resuming: {already_done} entries already saved, skipping ahead")
            entries = entries[already_done:]

    file_mode = "a" if already_done else "w"
    with open(args.output_file, file_mode) as out_f, open(args.diagnostics_file, file_mode) as diag_f:
        if args.mode == "llm":
            total_changed, total_processed = asyncio.run(
                run_pipeline(entries, args.batch_size, out_f, diag_f, already_done, already_done + len(entries), save_interval=1000)
            )
        else:
            total_changed, total_processed = run_pipeline_rule(
                entries, out_f, diag_f, already_done, already_done + len(entries), save_interval=1000
            )

    pct = (total_changed / total_processed * 100) if total_processed > 0 else 0
    logger.info(
        f"Cleanup complete: {total_changed}/{total_processed} entries modified ({pct:.1f}%). "
        f"Diagnostics written to {args.diagnostics_file}"
    )


if __name__ == "__main__":
    main()
