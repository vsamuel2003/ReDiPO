"""
Run LLM judge over MTBench model answers.

Supports mid-run resume: already-judged (question_id, model, judge_template, turn)
tuples are read from the output file and skipped before dispatching.

Usage:
    python gen_judgment.py \\
        --model-list llama-3-8b-instruct \\
        --judge-model gpt-5.4-mini-2026-03-17 \\
        --mode single \\
        [--parallel 4]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from judge_utils import (
    NEED_REF_CATS,
    Judge,
    MatchPair,
    MatchSingle,
    check_data,
    get_model_list,
    load_judge_prompts,
    load_model_answers,
    load_questions,
    play_a_match_pair,
    play_a_match_single,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_done_single(output_file: str) -> set:
    """Return set of (question_id, model, judge_template_name, turn) already judged."""
    done = set()
    if not os.path.exists(output_file):
        return done
    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                judge_name = obj["judge"][1] if isinstance(obj["judge"], (list, tuple)) else obj["judge"]
                done.add((obj["question_id"], obj["model"], judge_name, obj["turn"]))
            except (KeyError, json.JSONDecodeError):
                continue
    return done


def reorg_judgment_file(output_file: str):
    """Deduplicate judgment file, keeping the most recent entry per (qid, model, judge, turn)."""
    if not os.path.exists(output_file):
        return
    entries = {}
    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            judge_name = obj["judge"][1] if isinstance(obj["judge"], (list, tuple)) else obj["judge"]
            key = (obj["question_id"], obj["model"], judge_name, obj["turn"])
            if key not in entries or obj["tstamp"] > entries[key]["tstamp"]:
                entries[key] = obj
    with open(output_file, "w") as f:
        for obj in entries.values():
            f.write(json.dumps(obj) + "\n")


def _load_done_pair(output_file: str) -> set:
    """Return set of (question_id, model_1, model_2, judge_template_name, turn) already judged."""
    done = set()
    if not os.path.exists(output_file):
        return done
    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                judge_name = obj["judge"][1] if isinstance(obj["judge"], (list, tuple)) else obj["judge"]
                done.add((obj["question_id"], obj["model_1"], obj["model_2"], judge_name, obj["turn"]))
            except (KeyError, json.JSONDecodeError):
                continue
    return done


def make_match(questions, models, model_answers, judge, baseline_model, ref_answers=None, ref_model=None, multi_turn=False):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for m_1 in models:
            if m_1 == baseline_model:
                continue
            q_id = q["question_id"]
            a_1 = model_answers[m_1][q_id]
            a_2 = model_answers[baseline_model][q_id]
            ref = ref_answers[ref_model][q_id] if (ref_answers and ref_model) else None
            matches.append(MatchPair(dict(q), m_1, baseline_model, a_1, a_2, judge, ref_answer=ref, multi_turn=multi_turn))
    return matches


def make_match_all_pairs(questions, models, model_answers, judge, baseline_model=None, ref_answers=None, ref_model=None, multi_turn=False):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                q_id = q["question_id"]
                a_1 = model_answers[models[i]][q_id]
                a_2 = model_answers[models[j]][q_id]
                ref = ref_answers[ref_model][q_id] if (ref_answers and ref_model) else None
                matches.append(MatchPair(dict(q), models[i], models[j], a_1, a_2, judge, ref_answer=ref, multi_turn=multi_turn))
    return matches


def make_match_single(questions, models, model_answers, judge, baseline_model=None, ref_answers=None, ref_model=None, multi_turn=False):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for m in models:
            q_id = q["question_id"]
            a = model_answers[m][q_id]
            ref = ref_answers[ref_model][q_id] if (ref_answers and ref_model) else None
            matches.append(MatchSingle(dict(q), m, a, judge, ref_answer=ref, multi_turn=multi_turn))
    return matches


def make_judge_pairwise(judge_model, judge_prompts):
    return {
        "default": Judge(judge_model, judge_prompts["pair-v2"]),
        "math": Judge(judge_model, judge_prompts["pair-math-v1"], ref_based=True),
        "default-mt": Judge(judge_model, judge_prompts["pair-v2-multi-turn"], multi_turn=True),
        "math-mt": Judge(judge_model, judge_prompts["pair-math-v1-multi-turn"], ref_based=True, multi_turn=True),
    }


def make_judge_single(judge_model, judge_prompts):
    return {
        "default": Judge(judge_model, judge_prompts["single-v1"]),
        "math": Judge(judge_model, judge_prompts["single-math-v1"], ref_based=True),
        "default-mt": Judge(judge_model, judge_prompts["single-v1-multi-turn"], multi_turn=True),
        "math-mt": Judge(judge_model, judge_prompts["single-math-v1-multi-turn"], ref_based=True, multi_turn=True),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--judge-file", type=str, default=os.path.join(DATA_DIR, "judge_prompts.jsonl"))
    parser.add_argument("--judge-model", type=str, default="gpt-5.4-mini-2026-03-17")
    parser.add_argument("--ref-answer-model", type=str, default="gpt-4",
                        help="Key in reference_answer/ to use as gold refs for math/coding/reasoning. "
                             "Always gpt-4 (the original MTBench gold answers), independent of judge model.")
    parser.add_argument("--rejudge", action="store_true",
                        help="Delete existing judgment output file and re-run all matches from scratch.")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["pairwise-baseline", "pairwise-all", "single"])
    parser.add_argument("--model-list", type=str, nargs="+", default=None)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--first-n", type=int, default=None)
    parser.add_argument("--answer-dir", type=str, default=os.path.join(DATA_DIR, "model_answer"))
    parser.add_argument("--ref-answer-dir", type=str, default=os.path.join(DATA_DIR, "reference_answer"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(DATA_DIR, "model_judgment"))
    args = parser.parse_args()

    question_file = os.path.join(DATA_DIR, "question.jsonl")
    questions = load_questions(question_file, None, None)
    if args.first_n:
        questions = questions[: args.first_n]

    model_answers = load_model_answers(args.answer_dir)
    ref_answers = load_model_answers(args.ref_answer_dir)
    judge_prompts = load_judge_prompts(args.judge_file)

    models = args.model_list or get_model_list(args.answer_dir)

    if args.mode == "single":
        judges = make_judge_single(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        output_file = os.path.join(args.output_dir, f"{args.judge_model}_single.jsonl")
        make_match_func = make_match_single
        baseline_model = None
    else:
        judges = make_judge_pairwise(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_pair
        output_file = os.path.join(args.output_dir, f"{args.judge_model}_pair.jsonl")
        if args.mode == "pairwise-all":
            make_match_func = make_match_all_pairs
            baseline_model = None
        else:
            make_match_func = make_match
            baseline_model = args.baseline_model

    check_data(questions, model_answers, ref_answers, models, judges, ref_model=args.ref_answer_model)

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    matches = []
    matches += make_match_func(question_default, models, model_answers, judges["default"], baseline_model)
    matches += make_match_func(question_math, models, model_answers, judges["math"], baseline_model, ref_answers, ref_model=args.ref_answer_model)
    matches += make_match_func(question_default, models, model_answers, judges["default-mt"], baseline_model, multi_turn=True)
    matches += make_match_func(question_math, models, model_answers, judges["math-mt"], baseline_model, ref_answers, ref_model=args.ref_answer_model, multi_turn=True)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.rejudge and os.path.exists(output_file):
        os.remove(output_file)
        print("--rejudge set: removed previous output file.")

    # Resume: filter out already-judged matches
    if args.mode == "single":
        done = _load_done_single(output_file)
        def _match_key_single(m: MatchSingle) -> tuple:
            turn = 1 if not m.multi_turn else 2
            return (m.question["question_id"], m.model, m.judge.prompt_template["name"], turn)
        pending = [m for m in matches if _match_key_single(m) not in done]
    else:
        done = _load_done_pair(output_file)
        def _match_key_pair(m: MatchPair) -> tuple:
            turn = 1 if not m.multi_turn else 2
            return (m.question["question_id"], m.model_1, m.model_2, m.judge.prompt_template["name"], turn)
        pending = [m for m in matches if _match_key_pair(m) not in done]

    skipped = len(matches) - len(pending)
    if skipped:
        print(f"Resuming: {skipped} matches already judged, {len(pending)} remaining.")

    print(f"Total matches: {len(pending)}, output: {output_file}")

    if not pending:
        print("All matches already judged. Nothing to do.")
    elif args.parallel == 1:
        for match in tqdm(pending):
            play_a_match_func(match, output_file=output_file)
    else:
        def _play(match):
            play_a_match_func(match, output_file=output_file)

        np.random.seed(0)
        np.random.shuffle(pending)
        with ThreadPoolExecutor(args.parallel) as executor:
            for _ in tqdm(executor.map(_play, pending), total=len(pending)):
                pass

    reorg_judgment_file(output_file)
