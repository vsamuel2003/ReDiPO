"""
Compute win-rate from Arena-Hard pairwise judgments and write summary.json.

Usage:
    python show_result.py \
        --judgment-file <out_dir>/model_judgment/gpt-5.4-mini-2026-03-17.jsonl \
        --output-file   <out_dir>/summary.json \
        [--bootstrap 100]
"""

import argparse
import json
import math
import random
from pathlib import Path


def load_judgments(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def per_question_score(row: dict) -> float | None:
    """Average of g1_score and g2_score. Returns None if either is NaN (error)."""
    g1 = row.get("g1_score", float("nan"))
    g2 = row.get("g2_score", float("nan"))
    if math.isnan(g1) or math.isnan(g2):
        return None
    return (g1 + g2) / 2.0


def win_rate(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


def bootstrap_ci(scores: list[float], n: int = 100, seed: int = 42) -> tuple[float, float]:
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        resample = rng.choices(scores, k=len(scores))
        samples.append(win_rate(resample))
    samples.sort()
    lo_idx = max(0, int(0.05 * n) - 1)
    hi_idx = min(n - 1, int(0.95 * n))
    return samples[lo_idx], samples[hi_idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Arena-Hard win-rate from judgments.")
    parser.add_argument("--judgment-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--bootstrap", type=int, default=100)
    args = parser.parse_args()

    rows = load_judgments(args.judgment_file)
    if not rows:
        print("ERROR: judgment file is empty.")
        return

    sample = rows[0]
    model = sample.get("model", "unknown")
    judge = sample.get("judge", "unknown")
    baseline = sample.get("baseline", "unknown")

    scores = []
    n_errors = 0
    for row in rows:
        s = per_question_score(row)
        if s is None:
            n_errors += 1
        else:
            scores.append(s)

    wr = win_rate(scores)
    ci_low, ci_high = bootstrap_ci(scores, n=args.bootstrap) if len(scores) > 1 else (wr, wr)

    summary = {
        "model":    model,
        "judge":    judge,
        "baseline": baseline,
        "win_rate": round(wr, 4),
        "ci_low":   round(ci_low, 4),
        "ci_high":  round(ci_high, 4),
        "n_total":  len(rows),
        "n_errors": n_errors,
    }

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(
        f"Arena-Hard  model={model}  judge={judge}  baseline={baseline}\n"
        f"  win_rate = {wr:.4f}  ({wr*100:.1f}%)  "
        f"95% CI [{ci_low*100:.1f}%, {ci_high*100:.1f}%]  "
        f"n={len(scores)}  errors={n_errors}"
    )
    print(f"Summary written to {out_path}")


if __name__ == "__main__":
    main()
