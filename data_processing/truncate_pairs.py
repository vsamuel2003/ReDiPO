"""Truncate a preference pairs JSONL to match a reference file's line count.

Used by ablations to ensure all conditions train on the same number of pairs as the
full pipeline.

  rank_by=diversity_diff  Sort by |chosen_div - rejected_div| descending, keep top N.
                          Retains the highest-contrast pairs (appropriate for ablations
                          that still produce diversity scores, e.g. no_rewrite, no_if_filter,
                          no_match).
  rank_by=none            Keep first N lines as written.

Usage:
    python data_processing/truncate_pairs.py \\
        --input_file  preference_data/olmo_no_rewrite_pairs_raw.jsonl \\
        --output_file preference_data/olmo_no_rewrite_pairs.jsonl \\
        --reference_pair_file preference_data/olmo_pairs.jsonl \\
        --rank_by diversity_diff
"""

import argparse
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def count_lines(path: str) -> int:
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(entries: list[dict], path: str) -> None:
    dirpart = os.path.dirname(path)
    if dirpart:
        os.makedirs(dirpart, exist_ok=True)
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _div_diff(entry: dict) -> float:
    c = entry.get("chosen_marginal_diversity")
    r = entry.get("rejected_marginal_diversity")
    if c is None or r is None:
        return 0.0
    return abs(float(c) - float(r))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_file", required=True, help="Raw pairs JSONL to truncate")
    parser.add_argument("--output_file", required=True, help="Output JSONL path")
    parser.add_argument("--reference_pair_file", required=True, help="File whose line count sets N_target")
    parser.add_argument("--rank_by", choices=["diversity_diff", "none"], default="diversity_diff",
                        help="How to rank before truncating (default: diversity_diff)")
    args = parser.parse_args()

    n_target = count_lines(args.reference_pair_file)
    entries = load_jsonl(args.input_file)
    logger.info(f"Loaded {len(entries)} pairs; N_target={n_target} (from {args.reference_pair_file})")

    if args.rank_by == "diversity_diff":
        entries.sort(key=_div_diff, reverse=True)

    truncated = entries[:n_target]

    if len(entries) < n_target:
        logger.warning(
            f"Input ({len(entries)} pairs) < N_target ({n_target}). "
            f"Emitting all {len(entries)} pairs."
        )

    write_jsonl(truncated, args.output_file)
    logger.info(f"Wrote {len(truncated)} pairs to {args.output_file}")


if __name__ == "__main__":
    main()
