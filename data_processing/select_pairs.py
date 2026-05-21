"""Select preference pairs from scored responses for DPO training.

Reads a grouped JSONL with instruction_following_score and marginal_diversity on
each response, and outputs a flat JSONL with (chosen, rejected) pairs.

Three modes (default: epsilon):
  epsilon  - For each prompt, enumerate all response pairs and filter those
             with |IF diff| > epsilon. Rank by |diversity diff| descending.
             Apply per-response n-cap (greedy, highest-ranked first). Keep
             top quantile per prompt. chosen = higher diversity response.
  bin      - Bin responses by IF score. From the top-2 bins by marginal_diversity
             range, chosen = highest diversity, rejected = lowest. Up to 2 pairs
             per prompt.
  weighted - Score each response as alpha*IF + (1-alpha)*diversity. Sample chosen
             via weighted random draw; sample rejected via inverted weights.
             1 pair per prompt.

Output schema per line:
    {
      "prompt": "...",
      "chosen": "...",
      "rejected": "...",
      "chosen_marginal_diversity": float,
      "chosen_instruction_following": float,
      "rejected_marginal_diversity": float,
      "rejected_instruction_following": float
    }

Usage:
    # Epsilon mode (default)
    python data_processing/select_pairs.py \
        --input_file filtered_data/pilot_armro-filtered_embedded.jsonl \
        --output_file filtered_data/preference_pairs.jsonl \
        --epsilon 0.2 --n_cap 4 --top_quantile 0.25

    # Epsilon mode with base-model requirement
    python data_processing/select_pairs.py \
        --input_file filtered_data/pilot_armro-filtered_embedded.jsonl \
        --output_file filtered_data/preference_pairs.jsonl \
        --epsilon 0.2 --n_cap 4 --top_quantile 0.25 --require_base_in_pair

    # Bin mode
    python data_processing/select_pairs.py \
        --input_file filtered_data/pilot_armro-filtered_embedded.jsonl \
        --output_file filtered_data/preference_pairs.jsonl \
        --mode bin \
        --bin_width 0.5

    # Weighted mode
    python data_processing/select_pairs.py \
        --input_file filtered_data/pilot_armro-filtered_embedded.jsonl \
        --output_file filtered_data/preference_pairs.jsonl \
        --mode weighted \
        --alpha 0.5 \
        --seed 42
"""

import argparse
import itertools
import json
import logging
import math
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.embed_responses import load_grouped

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_base(response: dict) -> bool:
    """Heuristic: a response is from the base model iff its model name
    does not contain 'instruct' (case-insensitive)."""
    return "instruct" not in (response.get("model") or "").lower()


def _format_pair(prompt: str, chosen: dict, rejected: dict) -> dict:
    return {
        "prompt": prompt,
        "chosen": chosen["output"],
        "rejected": rejected["output"],
        "chosen_marginal_diversity": chosen["marginal_diversity"],
        "chosen_instruction_following": chosen["instruction_following_score"],
        "rejected_marginal_diversity": rejected["marginal_diversity"],
        "rejected_instruction_following": rejected["instruction_following_score"],
    }


def _bin_responses(responses: list[dict], bin_width: float) -> dict[float, list[dict]]:
    """Group responses into IF score bins."""
    bins: dict[float, list[dict]] = {}
    for r in responses:
        score = r.get("instruction_following_score")
        if score is None:
            continue
        bin_start = math.floor((score - 1.0) / bin_width) * bin_width + 1.0
        bins.setdefault(bin_start, []).append(r)
    return bins


def _normalize_weights(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0,1] then sum-normalize to a probability distribution.

    If all scores are identical, returns uniform distribution.
    """
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.ones(len(scores)) / len(scores)
    normalized = (scores - lo) / (hi - lo)
    total = normalized.sum()
    if total == 0:
        return np.ones(len(scores)) / len(scores)
    return normalized / total


# ---------------------------------------------------------------------------
# Mode C: Epsilon-based pair selection (default)
# ---------------------------------------------------------------------------

def select_pairs_epsilon(
    record: dict,
    epsilon: float,
    n_cap: int,
    top_quantile: float,
    require_base: bool = False,
) -> list[dict]:
    """Select preference pairs using epsilon IF-tolerance and diversity ranking.

    Args:
        record:       Single grouped prompt record with scored responses.
        epsilon:      Max allowed |IF score diff| between paired responses.
        n_cap:        Max times a single response may appear across kept pairs.
        top_quantile: Fraction of n-capped pairs to keep (top by |div diff|).
        require_base: If True, drop any pair where neither response is from the
                      base model (identified by the absence of 'instruct' in
                      the model name, case-insensitive). Applied before the
                      epsilon IF filter.

    Returns:
        List of pair dicts, possibly empty.
    """
    prompt = record["prompt"]
    responses = [
        r for r in record["responses"]
        if "marginal_diversity" in r and "instruction_following_score" in r
    ]
    if len(responses) < 2:
        return []

    # Step 1+2: Enumerate all pairs and filter by epsilon IF tolerance.
    candidates = []
    for r_i, r_j in itertools.combinations(responses, 2):
        if require_base and not (_is_base(r_i) or _is_base(r_j)):
            continue
        if_diff = abs(r_i["instruction_following_score"] - r_j["instruction_following_score"])
        if if_diff > epsilon:
            continue
        div_diff = abs(r_i["marginal_diversity"] - r_j["marginal_diversity"])
        candidates.append((div_diff, r_i, r_j))

    if not candidates:
        return []

    # Step 3: Rank by |diversity diff| descending (stable).
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Step 4: Per-response n-cap via greedy forward pass.
    counts: dict[int, int] = {}
    capped = []
    for div_diff, r_i, r_j in candidates:
        id_i = id(r_i)
        id_j = id(r_j)
        if counts.get(id_i, 0) < n_cap and counts.get(id_j, 0) < n_cap:
            counts[id_i] = counts.get(id_i, 0) + 1
            counts[id_j] = counts.get(id_j, 0) + 1
            capped.append((div_diff, r_i, r_j))

    if not capped:
        return []

    # Step 5: Top quantile per prompt (ceiling to keep at least 1).
    keep_n = math.ceil(top_quantile * len(capped))
    top = capped[:keep_n]

    # Step 6: Assign chosen/rejected; tiebreak on IF score.
    pairs = []
    for _, r_i, r_j in top:
        div_i = r_i["marginal_diversity"]
        div_j = r_j["marginal_diversity"]
        if div_i > div_j:
            chosen, rejected = r_i, r_j
        elif div_j > div_i:
            chosen, rejected = r_j, r_i
        else:
            # Tiebreak: higher IF score → chosen
            if r_i["instruction_following_score"] >= r_j["instruction_following_score"]:
                chosen, rejected = r_i, r_j
            else:
                chosen, rejected = r_j, r_i
        pairs.append(_format_pair(prompt, chosen, rejected))

    return pairs


# ---------------------------------------------------------------------------
# Mode A: Bin-based pair selection
# ---------------------------------------------------------------------------

def select_pairs_bin(record: dict, bin_width: float) -> list[dict]:
    """Select up to 2 preference pairs from the top-2 diversity-range bins.

    Args:
        record:    Single grouped prompt record with scored responses.
        bin_width: Width of IF score bins.

    Returns:
        List of pair dicts (0, 1, or 2 entries).
    """
    prompt = record["prompt"]
    responses = [
        r for r in record["responses"]
        if "marginal_diversity" in r and "instruction_following_score" in r
    ]
    if len(responses) < 2:
        return []

    binned = _bin_responses(responses, bin_width)

    # Keep only bins with >=2 responses
    qualifying = {bs: rs for bs, rs in binned.items() if len(rs) >= 2}
    if not qualifying:
        return []

    # Rank bins by diversity range descending
    def diversity_range(rs):
        divs = [r["marginal_diversity"] for r in rs]
        return max(divs) - min(divs)

    ranked = sorted(qualifying.values(), key=diversity_range, reverse=True)
    top_bins = ranked[:2]

    pairs = []
    for bin_responses in top_bins:
        sorted_by_div = sorted(bin_responses, key=lambda r: r["marginal_diversity"])
        rejected = sorted_by_div[0]   # lowest diversity
        chosen = sorted_by_div[-1]    # highest diversity
        pairs.append(_format_pair(prompt, chosen, rejected))

    return pairs


# ---------------------------------------------------------------------------
# Mode B: Weighted sampling pair selection
# ---------------------------------------------------------------------------

def select_pairs_weighted(
    record: dict,
    alpha: float,
    rng: np.random.Generator,
) -> list[dict]:
    """Select 1 preference pair via weighted random sampling.

    Args:
        record: Single grouped prompt record with scored responses.
        alpha:  Weight for IF score vs diversity. score = alpha*IF + (1-alpha)*diversity.
        rng:    numpy random generator for reproducibility.

    Returns:
        List with exactly 1 pair dict, or empty list if <2 valid responses.
    """
    prompt = record["prompt"]
    responses = [
        r for r in record["responses"]
        if "marginal_diversity" in r and "instruction_following_score" in r
    ]
    if len(responses) < 2:
        return []

    # Composite score per response
    raw_scores = np.array([
        alpha * r["instruction_following_score"] + (1.0 - alpha) * r["marginal_diversity"]
        for r in responses
    ])

    weights = _normalize_weights(raw_scores)

    # Sample chosen
    chosen_idx = rng.choice(len(responses), p=weights)

    # Invert: zero out chosen, invert remaining, re-normalize
    inv_weights = 1.0 - weights
    inv_weights[chosen_idx] = 0.0
    total = inv_weights.sum()
    if total == 0:
        # Edge case: only 2 responses, the other must be rejected
        rejected_idx = 1 - chosen_idx
    else:
        inv_weights /= total
        rejected_idx = rng.choice(len(responses), p=inv_weights)

    return [_format_pair(prompt, responses[chosen_idx], responses[rejected_idx])]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Select preference pairs from scored grouped JSONL for DPO training."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Grouped JSONL with IF scores and marginal_diversity")
    parser.add_argument("--output_file", type=str, required=True, help="Output flat JSONL with (prompt, chosen, rejected, ...) pairs")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["epsilon", "bin", "weighted"],
        default="epsilon",
        help="Pair selection strategy (default: epsilon)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="Max |IF score diff| allowed between paired responses in epsilon mode (default: 0.2)",
    )
    parser.add_argument(
        "--n_cap",
        type=int,
        default=4,
        help="Max times a response may appear across pairs per prompt in epsilon mode (default: 4)",
    )
    parser.add_argument(
        "--top_quantile",
        type=float,
        default=0.25,
        help="Fraction of pairs to keep per prompt in epsilon mode (default: 0.25)",
    )
    parser.add_argument(
        "--bin_width",
        type=float,
        default=0.5,
        help="IF score bin width for bin mode (default: 0.5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for IF score in weighted mode: score = alpha*IF + (1-alpha)*diversity (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for weighted mode (default: 42)",
    )
    parser.add_argument(
        "--require_base_in_pair",
        action="store_true",
        help="Epsilon mode only: drop pairs that do not contain at least one "
             "base-model response (base = model name lacks 'instruct', "
             "case-insensitive). Applied before the epsilon IF filter.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")
    if not 0.0 < args.top_quantile <= 1.0:
        parser.error("--top_quantile must be in (0, 1]")

    if args.require_base_in_pair and args.mode != "epsilon":
        logger.warning(
            "--require_base_in_pair has no effect in '%s' mode; it is only supported for epsilon mode.",
            args.mode,
        )

    records = load_grouped(args.input_file)
    rng = np.random.default_rng(args.seed)

    all_pairs: list[dict] = []
    skipped = 0

    for record in tqdm(records, desc="Selecting pairs", unit="prompt"):
        if args.mode == "epsilon":
            pairs = select_pairs_epsilon(
                record, args.epsilon, args.n_cap, args.top_quantile,
                require_base=args.require_base_in_pair,
            )
        elif args.mode == "bin":
            pairs = select_pairs_bin(record, args.bin_width)
        else:
            pairs = select_pairs_weighted(record, args.alpha, rng)

        if not pairs:
            skipped += 1
        all_pairs.extend(pairs)

    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    mode_info = (
        f"epsilon={args.epsilon}, n_cap={args.n_cap}, top_q={args.top_quantile}, require_base={args.require_base_in_pair}"
        if args.mode == "epsilon"
        else f"bin_width={args.bin_width}" if args.mode == "bin"
        else f"alpha={args.alpha}"
    )
    logger.info(
        f"Mode={args.mode} ({mode_info}): wrote {len(all_pairs)} pairs from {len(records)} prompts "
        f"({skipped} prompts yielded no pairs). Output: {args.output_file}"
    )


if __name__ == "__main__":
    main()
