"""
Run pairwise LLM-as-judge for Arena-Hard: compare our model's answers against
the gpt-4o-2024-08-06 baseline using two rounds (A/B swap) per question.

Usage:
    python run_judge.py \
        --question-file   evals/arena_hard/data/question.jsonl \
        --baseline-file   evals/arena_hard/data/baseline_answer.jsonl \
        --answer-file     <out_dir>/model_answer/<model_id>.jsonl \
        --output-file     <out_dir>/model_judgment/gpt-5.4-mini-2026-03-17.jsonl \
        --judge-model     gpt-5.4-mini-2026-03-17 \
        --max-tokens      16000 \
        --parallel        8 \
        [--rejudge]

Resume: uids already present in --output-file are skipped (unless --rejudge).
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arena_hard.judge_utils import (  # noqa: E402
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    chat_completion_openai,
    parse_verdict,
    score_for_b,
)


def load_jsonl_by_uid(path: str) -> dict:
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                data[d["uid"]] = d
    return data


def load_done_uids(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                    # Only mark as done if both rounds were judged
                    if "g1_verdict" in d and "g2_verdict" in d:
                        done.add(d["uid"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def build_user_prompt(question: str, answer_a: str, answer_b: str) -> str:
    return USER_TEMPLATE.format(QUESTION=question, ANSWER_A=answer_a, ANSWER_B=answer_b)


def judge_one(uid: str, question: str, baseline_answer: str, model_answer: str,
              judge_model: str, max_tokens: int, baseline_name: str, model_name: str) -> dict:
    """Run two judge rounds for a single question. Returns a judgment record."""
    messages_g1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, baseline_answer, model_answer)},
    ]
    messages_g2 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, model_answer, baseline_answer)},
    ]

    g1_judgment = chat_completion_openai(judge_model, messages_g1, temperature=0, max_tokens=max_tokens)
    g2_judgment = chat_completion_openai(judge_model, messages_g2, temperature=0, max_tokens=max_tokens)

    # g1: A=baseline, B=model → score_for_b gives model's score
    g1_verdict = parse_verdict(g1_judgment)
    g1_score = score_for_b(g1_verdict)

    # g2: A=model, B=baseline → model plays A, so its score = 1 - score_for_b
    g2_verdict = parse_verdict(g2_judgment)
    g2_score_raw = score_for_b(g2_verdict)
    g2_score = (1.0 - g2_score_raw) if not math.isnan(g2_score_raw) else float("nan")

    return {
        "uid": uid,
        "model": model_name,
        "baseline": baseline_name,
        "judge": judge_model,
        "g1_verdict": g1_verdict,
        "g1_judgment": g1_judgment,
        "g1_score": g1_score,
        "g2_verdict": g2_verdict,
        "g2_judgment": g2_judgment,
        "g2_score": g2_score,
        "tstamp": time.time(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise judge for Arena-Hard.")
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--baseline-file", required=True)
    parser.add_argument("--answer-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--judge-model", default="gpt-5.4-mini-2026-03-17")
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--rejudge", action="store_true",
                        help="Re-run all judgments, overwriting existing output.")
    args = parser.parse_args()

    questions = load_jsonl_by_uid(args.question_file)
    baselines = load_jsonl_by_uid(args.baseline_file)
    answers = load_jsonl_by_uid(args.answer_file)

    if not answers:
        print("ERROR: answer file is empty or missing.", file=sys.stderr)
        sys.exit(1)

    # Determine model name from the answer file
    sample = next(iter(answers.values()))
    model_name = sample.get("model", "unknown")
    baseline_name = next(iter(baselines.values())).get("model", "gpt-4o-2024-08-06") if baselines else "gpt-4o-2024-08-06"

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.rejudge and out_path.exists():
        out_path.unlink()
        done_uids: set = set()
    else:
        done_uids = load_done_uids(args.output_file)

    # Only judge questions that have both a baseline answer and a model answer
    todo_uids = [
        uid for uid in questions
        if uid not in done_uids and uid in baselines and uid in answers
    ]

    missing_baseline = sum(1 for uid in questions if uid not in baselines)
    missing_answer = sum(1 for uid in questions if uid not in answers)
    if missing_baseline:
        print(f"WARNING: {missing_baseline} questions have no baseline answer — skipping them.")
    if missing_answer:
        print(f"WARNING: {missing_answer} questions have no model answer — skipping them.")

    print(f"Questions: {len(questions)} | Done: {len(done_uids)} | To judge: {len(todo_uids)}")

    if not todo_uids:
        print("All questions already judged.")
        return

    errors = 0
    with open(out_path, "a") as fout:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    judge_one,
                    uid,
                    questions[uid]["prompt"],
                    baselines[uid]["answer"],
                    answers[uid]["answer"],
                    args.judge_model,
                    args.max_tokens,
                    baseline_name,
                    model_name,
                ): uid
                for uid in todo_uids
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
                uid = futures[future]
                try:
                    result = future.result()
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()
                except Exception as e:
                    print(f"\nERROR on uid={uid}: {e}", file=sys.stderr)
                    errors += 1

    total = len(done_uids) + len(todo_uids) - errors
    print(f"Judging complete. New: {len(todo_uids) - errors} | Errors: {errors} | Total judged: {total}")
    if errors:
        print(f"  {errors} questions failed — rerun to retry (they are absent from output).")


if __name__ == "__main__":
    main()
