"""
Build paired_scores.csv by joining GPT-4 and new judge single-grade judgments.

Dedup logic mirrors show_result.py: sort by tstamp, keep last per (question_id, model, turn).
Rows with score == -1 (parse/API failure) are dropped.
Joins on (question_id, model, turn); drops any rows without all three judges present.

Output: evals/judge_agreement/paired_scores.csv
Columns: model, question_id, turn, category, judge_template, gpt4_score, mini_score, nano_score
"""

import json
import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_DIR = os.path.join(REPO_ROOT, "evals", "results")
QUESTION_FILE = os.path.join(REPO_ROOT, "evals", "mtbench", "data", "question.jsonl")
OUT_CSV = os.path.join(SCRIPT_DIR, "paired_scores.csv")

EVALUATED_MODELS = ["qwen3-4b-base", "qwen3-4b-instruct-2507"]
JUDGES = {
    "gpt4":  "gpt-4",
    "mini":  "gpt-5.4-mini-2026-03-17",
    "nano":  "gpt-5.4-nano-2026-03-17",
}


def load_judgments(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append({
                    "question_id": obj["question_id"],
                    "model": obj["model"],
                    "turn": obj["turn"],
                    "score": obj["score"],
                    "tstamp": obj["tstamp"],
                    "judge_template": obj["judge"][1] if isinstance(obj["judge"], (list, tuple)) else obj["judge"],
                })
            except (KeyError, json.JSONDecodeError):
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Dedup: keep latest by tstamp
    df = df.sort_values("tstamp").drop_duplicates(subset=["question_id", "model", "turn"], keep="last")
    # Drop parse/API failures
    df = df[df["score"] != -1].copy()
    return df[["question_id", "model", "turn", "score", "judge_template"]]


def load_question_categories() -> dict:
    cats = {}
    with open(QUESTION_FILE) as f:
        for line in f:
            obj = json.loads(line.strip())
            cats[obj["question_id"]] = obj["category"]
    return cats


def main():
    question_cats = load_question_categories()
    frames = []

    for eval_model in EVALUATED_MODELS:
        judgment_dir = os.path.join(RESULTS_DIR, eval_model, "mtbench", "model_judgment")
        dfs = {}
        for col_name, judge_model in JUDGES.items():
            path = os.path.join(judgment_dir, f"{judge_model}_single.jsonl")
            df = load_judgments(path)
            if df.empty:
                print(f"WARNING: No judgments found for judge={judge_model}, model={eval_model} at {path}")
            dfs[col_name] = df

        if any(dfs[k].empty for k in JUDGES):
            missing = [k for k in JUDGES if dfs[k].empty]
            print(f"Skipping {eval_model}: missing judgments for {missing}")
            continue

        merged = dfs["gpt4"].rename(columns={"score": "gpt4_score", "judge_template": "judge_template"})
        merged = merged.merge(
            dfs["mini"][["question_id", "model", "turn", "score"]].rename(columns={"score": "mini_score"}),
            on=["question_id", "model", "turn"], how="inner"
        )
        merged = merged.merge(
            dfs["nano"][["question_id", "model", "turn", "score"]].rename(columns={"score": "nano_score"}),
            on=["question_id", "model", "turn"], how="inner"
        )
        merged["category"] = merged["question_id"].map(question_cats)
        frames.append(merged)

    if not frames:
        print("ERROR: No paired data built. Run run_rejudge.sh first.")
        sys.exit(1)

    paired = pd.concat(frames, ignore_index=True)
    paired = paired[["model", "question_id", "turn", "category", "judge_template",
                      "gpt4_score", "mini_score", "nano_score"]]
    paired.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(paired)} rows to {OUT_CSV}")
    print(f"Breakdown by model:\n{paired.groupby('model').size()}")
    print(f"Breakdown by turn:\n{paired.groupby('turn').size()}")


if __name__ == "__main__":
    main()
