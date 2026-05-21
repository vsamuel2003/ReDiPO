"""Group flat .jsonl responses by prompt into per-prompt records."""

import argparse
import json
import logging
import os
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_entries(input_file: str) -> list[dict]:
    entries = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    logger.info(f"Loaded {len(entries)} entries from {input_file}")
    return entries


def group_by_prompt(entries: list[dict]) -> list[dict]:
    """Group entries by prompt, preserving all fields per response."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        prompt = entry["prompt"]
        response = {k: v for k, v in entry.items() if k != "prompt"}
        groups[prompt].append(response)

    grouped = [{"prompt": prompt, "responses": responses} for prompt, responses in groups.items()]
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Group flat .jsonl responses by prompt.")
    parser.add_argument("--input_file", type=str, required=True, help="Flat input JSONL (one response per line)")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL (one prompt per line)")
    args = parser.parse_args()

    entries = load_entries(args.input_file)
    grouped = group_by_prompt(entries)

    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        for record in grouped:
            f.write(json.dumps(record) + "\n")

    total_responses = sum(len(r["responses"]) for r in grouped)
    avg = total_responses / len(grouped) if grouped else 0
    logger.info(
        f"Grouped {total_responses} responses into {len(grouped)} prompts "
        f"(avg {avg:.1f} responses/prompt). Written to {args.output_file}"
    )


if __name__ == "__main__":
    main()
