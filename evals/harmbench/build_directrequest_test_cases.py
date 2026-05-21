"""
Build a DirectRequest test_cases.json from a HarmBench behaviors CSV.

Mirrors the logic in HarmBench/baselines/direct_request/direct_request.py:
  - One test case per behavior.
  - If ContextString is present, prepend it: "{ctx}\n\n---\n\n{behavior}".
  - Output format: {BehaviorID: [test_case_string]}  (one-element list).

Usage:
    python build_directrequest_test_cases.py --split test
    python build_directrequest_test_cases.py --split test --include-hash-check \\
        --out data/test_cases_directrequest_test_full.json
"""

import argparse
import csv
import json
import os

BEHAVIORS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "behavior_datasets")


def build(split: str, include_hash_check: bool, out_path: str) -> None:
    csv_name = f"harmbench_behaviors_text_{split}.csv"
    csv_path = os.path.join(BEHAVIORS_DIR, csv_name)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Behaviors CSV not found: {csv_path}")

    test_cases: dict[str, list[str]] = {}
    skipped = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tags = [t.strip() for t in (row.get("Tags") or "").split(",") if t.strip()]
            if not include_hash_check and "hash_check" in tags:
                skipped += 1
                continue

            ctx = (row.get("ContextString") or "").strip()
            tc = f"{ctx}\n\n---\n\n{row['Behavior']}" if ctx else row["Behavior"]
            test_cases[row["BehaviorID"]] = [tc]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(test_cases, f, indent=2)

    print(f"Wrote {len(test_cases)} behaviors to {out_path}")
    if skipped:
        print(f"Skipped {skipped} hash_check behaviors (pass --include-hash-check to include them)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build DirectRequest test_cases.json from HarmBench behaviors CSV"
    )
    parser.add_argument(
        "--split", choices=["test", "val", "all"], default="test",
        help="Which behaviors split to use (default: test)"
    )
    parser.add_argument(
        "--include-hash-check", action="store_true",
        help="Include hash_check behaviors (requires copyright_classifier_hashes/ .pkl files)"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output path (default: data/test_cases_directrequest_<split>[_full].json)"
    )
    args = parser.parse_args()

    if args.out is None:
        suffix = "_full" if args.include_hash_check else ""
        args.out = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            f"test_cases_directrequest_{args.split}{suffix}.json",
        )

    build(args.split, args.include_hash_check, args.out)
