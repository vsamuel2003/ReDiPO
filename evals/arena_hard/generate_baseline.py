"""
One-time bootstrap: generate gpt-4o-2024-08-06 baseline answers for all Arena-Hard
hard_prompt questions and write them to data/baseline_answer.jsonl.

Run once, then check the output file into the repo. Subsequent eval runs reuse
the cached file — no re-generation needed.

Usage:
    python generate_baseline.py [--question-file data/question.jsonl]
                                [--out data/baseline_answer.jsonl]
                                [--parallel 8]

Resume: already-answered uids are skipped automatically.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mtbench.judge_utils import chat_completion_openai

BASELINE_MODEL = "gpt-4o-2024-08-06"
MAX_TOKENS = 1024
TEMPERATURE = 0.0


def load_questions(path: str) -> list[dict]:
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def load_done(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                    done.add(d["uid"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def answer_question(q: dict) -> dict:
    messages = [{"role": "user", "content": q["prompt"]}]
    answer = chat_completion_openai(BASELINE_MODEL, messages, TEMPERATURE, MAX_TOKENS)
    return {"uid": q["uid"], "model": BASELINE_MODEL, "answer": answer, "tstamp": time.time()}


def main() -> None:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Generate gpt-4o-2024-08-06 baseline answers.")
    parser.add_argument("--question-file", default=str(here / "data" / "question.jsonl"))
    parser.add_argument("--out", default=str(here / "data" / "baseline_answer.jsonl"))
    parser.add_argument("--parallel", type=int, default=8)
    args = parser.parse_args()

    questions = load_questions(args.question_file)
    done = load_done(args.out)

    todo = [q for q in questions if q["uid"] not in done]
    print(f"Total questions: {len(questions)} | Already done: {len(done)} | Remaining: {len(todo)}")

    if not todo:
        print("All answers present. Nothing to do.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    errors = 0
    completed = 0
    with open(out_path, "a") as fout:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(answer_question, q): q for q in todo}
            for future in as_completed(futures):
                q = futures[future]
                try:
                    result = future.result()
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()
                    completed += 1
                    if completed % 50 == 0:
                        print(f"  Progress: {completed}/{len(todo)}")
                except Exception as e:
                    print(f"ERROR on uid={q['uid']}: {e}", file=sys.stderr)
                    errors += 1

    total_done = len(done) + completed
    print(f"Done. Generated {completed} new answers ({errors} errors). Total in file: {total_done}")
    if errors:
        print(f"  {errors} questions failed — rerun to retry them.")


if __name__ == "__main__":
    main()
