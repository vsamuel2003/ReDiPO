"""Collect and filter prompts from multiple HuggingFace datasets."""

import argparse
import json
import logging
import os
import random
import re
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FILTER_STRINGS = ["https", ".py", ".c", "Markdown", "json", "java", "python", "html"]


def has_unicode(text: str) -> bool:
    """Check if text contains non-ASCII characters or literal \\u escape sequences."""
    if "\\u" in text:
        return True
    return not all(ord(c) < 128 for c in text)


def collect_dolly(hf_token: str | None) -> list[dict]:
    """Collect prompts from databricks/databricks-dolly-15k."""
    logger.info("Loading databricks/databricks-dolly-15k...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train", token=hf_token)

    valid_categories = {"open_qa", "creative_writing", "brainstorming", "summarization"}
    results = []

    for row in tqdm(ds, desc="Filtering dolly"):
        if row["category"] not in valid_categories:
            continue
        if len(row["response"].split()) <= 10:
            continue

        context = row.get("context", "").strip()
        instruction = row["instruction"].strip()
        prompt = f"{context} {instruction}" if context else instruction

        results.append({"prompt": prompt, "source": "dolly", "type": "instruct"})

    return results


def collect_or_bench(hf_token: str | None) -> list[dict]:
    """Collect prompts from bench-llm/or-bench (hard + toxic subsets)."""
    results = []

    for subset in ["or-bench-hard-1k", "or-bench-toxic"]:
        logger.info(f"Loading bench-llm/or-bench [{subset}]...")
        ds = load_dataset("bench-llm/or-bench", subset, split="train", token=hf_token)

        for row in tqdm(ds, desc=f"Processing or-bench/{subset}"):
            results.append({"prompt": row["prompt"], "source": "or-bench", "type": "safety"})

    return results


def collect_wildjailbreak(hf_token: str | None) -> list[dict]:
    """Collect prompts from allenai/wildjailbreak (eval subset)."""
    logger.info("Loading allenai/wildjailbreak [eval]...")
    ds = load_dataset("allenai/wildjailbreak", "eval", split="train", token=hf_token)

    results = []
    for row in tqdm(ds, desc="Filtering wildjailbreak"):
        prompt = row["adversarial"]
        if any(s.lower() in prompt.lower() for s in FILTER_STRINGS):
            continue
        results.append({"prompt": prompt, "source": "wildjailbreak", "type": "safety"})

    return results


def collect_jailbreakbench(hf_token: str | None) -> list[dict]:
    """Collect prompts from JailbreakBench/JBB-Behaviors (harmful + benign splits)."""
    results = []

    for split in ["harmful", "benign"]:
        logger.info(f"Loading JailbreakBench/JBB-Behaviors [behaviors, {split}]...")
        ds = load_dataset(
            "JailbreakBench/JBB-Behaviors", "behaviors", split=split, token=hf_token
        )

        for row in tqdm(ds, desc=f"Filtering JBB-Behaviors/{split}"):
            if "Harmbench" in str(row.get("Source", "")):
                continue
            results.append({"prompt": row["Goal"], "source": "jailbreak", "type": "safety"})

    return results


def collect_writingprompts(hf_token: str | None, n: int = 5500, seed: int = 42) -> list[dict]:
    """Collect prompts from euclaise/WritingPrompts_preferences."""
    logger.info("Loading euclaise/WritingPrompts_preferences...")
    ds = load_dataset("euclaise/WritingPrompts_preferences", split="train", token=hf_token)

    _bracket_re = re.compile(r"^\[.*?\]\s*")

    def clean_title(title: str) -> str:
        return _bracket_re.sub("", title).strip()

    pool = []
    seen = set()

    for row in tqdm(ds, desc="Filtering WritingPrompts"):
        title = clean_title(row.get("post_title", ""))
        body = row.get("post_text", "").strip()
        prompt = f"{title}\n{body}".strip() if body else title
        if not prompt:
            continue
        if has_unicode(prompt):
            continue
        if prompt in seen:
            continue
        seen.add(prompt)
        pool.append({"prompt": prompt, "source": "writingprompts", "type": "creative"})

    rng = random.Random(seed)
    if len(pool) > n:
        pool = rng.sample(pool, n)

    return pool


def write_jsonl(data: list[dict], path: str) -> None:
    """Write a list of dicts to a JSONL file."""
    with open(path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


def log_stats(all_data: list[dict], subset_data: list[dict], raw_counts: dict[str, int]) -> None:
    """Log a summary table of per-source statistics."""
    source_counts = defaultdict(int)
    subset_counts = defaultdict(int)
    source_types = {}

    for entry in all_data:
        source_counts[entry["source"]] += 1
        source_types[entry["source"]] = entry["type"]
    for entry in subset_data:
        subset_counts[entry["source"]] += 1

    logger.info("=" * 70)
    logger.info("DATA COLLECTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Source':<20} {'Type':<10} {'Raw':<10} {'Filtered':<10} {'Subset':<10}")
    logger.info("-" * 70)

    total_raw, total_filtered, total_subset = 0, 0, 0
    for source in sorted(source_counts.keys()):
        raw = raw_counts.get(source, 0)
        filtered = source_counts[source]
        subset = subset_counts[source]
        stype = source_types[source]
        logger.info(f"{source:<20} {stype:<10} {raw:<10} {filtered:<10} {subset:<10}")
        total_raw += raw
        total_filtered += filtered
        total_subset += subset

    logger.info("-" * 70)
    logger.info(f"{'TOTAL':<20} {'':<10} {total_raw:<10} {total_filtered:<10} {total_subset:<10}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Collect prompts from HuggingFace datasets.")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subset sampling")
    parser.add_argument("--subset_size", type=int, default=50, help="Samples per source for subset (or total for writingprompts)")
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Comma-separated list of sources to collect from. "
             "Options: dolly,or-bench,wildjailbreak,jailbreak,writingprompts. "
             "Default: all four legacy sources.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Override the output filename for the main data file (e.g. writingprompts_5500.jsonl). "
             "If not set, uses full_data.jsonl / subset_data.jsonl as usual.",
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    random.seed(args.seed)

    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None

    all_legacy_collectors = [
        ("dolly", collect_dolly),
        ("or-bench", collect_or_bench),
        ("wildjailbreak", collect_wildjailbreak),
        ("jailbreak", collect_jailbreakbench),
    ]

    # WritingPrompts: single-source mode — collect and write directly
    if sources == ["writingprompts"]:
        entries = collect_writingprompts(hf_token, n=args.subset_size, seed=args.seed)
        os.makedirs(args.output_dir, exist_ok=True)
        out_name = args.output_name or "writingprompts.jsonl"
        out_path = os.path.join(args.output_dir, out_name)
        write_jsonl(entries, out_path)
        logger.info(f"WritingPrompts data written to {out_path} ({len(entries)} entries)")
        return

    # Legacy multi-source mode
    collectors = [
        (name, fn) for name, fn in all_legacy_collectors
        if sources is None or name in sources
    ]
    if sources and "writingprompts" in sources:
        collectors.append(("writingprompts", lambda tok: collect_writingprompts(tok, n=args.subset_size, seed=args.seed)))

    all_data = []
    raw_counts = {}

    for source_name, collector_fn in collectors:
        entries = collector_fn(hf_token)
        raw_counts[source_name] = len(entries)
        all_data.extend(entries)
        logger.info(f"  {source_name}: {len(entries)} prompts after filtering")

    # Unicode filter across all sources
    pre_unicode = len(all_data)
    all_data = [
        entry for entry in tqdm(all_data, desc="Filtering unicode")
        if not has_unicode(entry["prompt"])
    ]
    logger.info(f"Unicode filter: {pre_unicode - len(all_data)}/{pre_unicode} removed")

    # Build subset: sample up to subset_size per source
    by_source = defaultdict(list)
    for entry in all_data:
        by_source[entry["source"]].append(entry)

    subset_data = []
    for source, entries in by_source.items():
        if len(entries) <= args.subset_size:
            subset_data.extend(entries)
        else:
            subset_data.extend(random.sample(entries, args.subset_size))

    # Write outputs
    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_name:
        out_path = os.path.join(args.output_dir, args.output_name)
        write_jsonl(subset_data, out_path)
        logger.info(f"Data written to {out_path} ({len(subset_data)} entries)")
    else:
        full_path = os.path.join(args.output_dir, "full_data.jsonl")
        subset_path = os.path.join(args.output_dir, "subset_data.jsonl")
        write_jsonl(all_data, full_path)
        write_jsonl(subset_data, subset_path)
        logger.info(f"Full data written to {full_path} ({len(all_data)} entries)")
        logger.info(f"Subset data written to {subset_path} ({len(subset_data)} entries)")

    # Log summary stats
    log_stats(all_data, subset_data, raw_counts)


if __name__ == "__main__":
    main()
