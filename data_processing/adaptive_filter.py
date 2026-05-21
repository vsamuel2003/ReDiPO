"""Compute adaptive instruction-following threshold from instruct model data, then run filter."""

import argparse
import json
import logging
import os
import subprocess
import sys

# Add project root to path for scoring imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tqdm import tqdm
from scoring.instruction_following import score_instruction_following, load_armorm, load_skywork

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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


def get_instruct_entries(entries: list[dict]) -> list[dict]:
    """Return entries where the model field indicates an instruct model."""
    instruct = [e for e in entries if "instruct" in e["model"].lower()]
    logger.info(f"Found {len(instruct)}/{len(entries)} entries from instruct models")
    return instruct


def compute_mean_score(entries: list[dict], method: str) -> float:
    """Score all entries with the given method and return the mean score.

    Args:
        entries: List of data entries (instruct model outputs).
        method: "armorm", "judge_soft", or "skywork".

    Returns:
        Mean instruction-following score across all entries.
    """
    armorm_model, armorm_tokenizer = None, None
    if method == "armorm":
        logger.info("Loading ArmoRM model...")
        armorm_model, armorm_tokenizer = load_armorm()
        logger.info("ArmoRM model loaded.")

    skywork_model, skywork_tokenizer = None, None
    if method == "skywork":
        logger.info("Loading Skywork model...")
        skywork_model, skywork_tokenizer = load_skywork()
        logger.info("Skywork model loaded.")

    total_score = 0.0
    for entry in tqdm(entries, desc=f"Scoring instruct entries ({method})", unit="entry"):
        score = score_instruction_following(
            entry["prompt"],
            entry["output"],
            method,
            armorm_model=armorm_model,
            armorm_tokenizer=armorm_tokenizer,
            skywork_model=skywork_model,
            skywork_tokenizer=skywork_tokenizer,
        )
        total_score += score

    mean = total_score / len(entries)
    return mean


def run_filter(args: argparse.Namespace, threshold: float) -> None:
    """Invoke filter.py as a subprocess with the computed threshold."""
    cmd = [
        "python", "data_processing/filter.py",
        "--input_file", args.input_file,
        "--safety", args.safety,
        "--instruct", args.if_method,
        "--scorer", args.scorer,
        "--threshold", str(threshold),
        "--min_samples", str(args.min_samples),
        "--min_base_samples", str(args.min_base_samples),
        "--output_file", args.output_file,
    ]
    logger.info(f"Running filter: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compute adaptive IF threshold from instruct model data and run filter."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL with generations")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL path (forwarded to filter.py)")
    parser.add_argument(
        "--if_method", type=str, required=True, choices=["armorm", "judge_soft", "skywork"],
        help="Instruction-following scoring method for computing mean and filtering",
    )
    parser.add_argument(
        "--safety", type=str, required=True, choices=["judge", "llama_guard"],
        help="Safety filtering method (forwarded to filter.py)",
    )
    parser.add_argument(
        "--scorer", type=str, required=True, choices=["armorm", "judge_soft", "skywork"],
        help="Scorer for assigning IF scores to passing entries (forwarded to filter.py)",
    )
    parser.add_argument(
        "--deviance", type=float, required=True,
        help="Fraction below |mean| to set threshold: threshold = mean - deviance * |mean_score|",
    )
    parser.add_argument("--min_samples", type=int, default=10,
                        help="Minimum total responses per prompt (forwarded to filter.py, default: 10)")
    parser.add_argument("--min_base_samples", type=int, default=2,
                        help="Minimum base-model responses per prompt (forwarded to filter.py, default: 2)")
    args = parser.parse_args()

    if not (0.0 <= args.deviance <= 1.0):
        parser.error("--deviance must be between 0.0 and 1.0")

    entries = load_data(args.input_file)

    instruct_entries = get_instruct_entries(entries)
    if not instruct_entries:
        logger.error("No instruct model entries found in input file. Cannot compute adaptive threshold.")
        sys.exit(1)

    mean_score = compute_mean_score(instruct_entries, args.if_method)
    threshold = mean_score - args.deviance * abs(mean_score)

    print("--------------------------")
    print(f'THRESHOLD IS {threshold}')
    print("--------------------------")

    logger.info(
        f"Instruct entries: {len(instruct_entries)} | "
        f"Mean IF score ({args.if_method}): {mean_score:.4f} | "
        f"Deviance: {args.deviance} | "
        f"Computed threshold (mean - deviance * |mean|): {threshold:.4f}"
    )

    run_filter(args, threshold)


if __name__ == "__main__":
    main()
