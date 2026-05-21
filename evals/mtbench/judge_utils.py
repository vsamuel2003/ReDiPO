"""
MTBench data structures and utilities.
Trimmed from FastChat — no fastchat.* imports.
"""

import ast
import dataclasses
import glob
import json
import os
import re
import time
from typing import Optional

import openai


# API constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

TIE_DELTA = 0.1

NEED_REF_CATS = ["math", "reasoning", "coding", "arena-hard-200"]

two_score_pattern = re.compile(r"\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]")
two_score_pattern_backup = re.compile(r"\[(\d+\.?\d*),\s?(\d+\.?\d*)\]")
one_score_pattern = re.compile(r"\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile(r"\[(\d+\.?\d*)\]")

temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}

OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5.4-mini-2026-03-17",
    "gpt-5.4-nano-2026-03-17",
)


@dataclasses.dataclass
class Judge:
    model_name: str
    prompt_template: dict
    ref_based: bool = False
    multi_turn: bool = False


@dataclasses.dataclass
class MatchSingle:
    question: dict
    model: str
    answer: dict
    judge: Judge
    ref_answer: dict = None
    multi_turn: bool = False


@dataclasses.dataclass
class MatchPair:
    question: dict
    model_1: str
    model_2: str
    answer_1: dict
    answer_2: dict
    judge: Judge
    ref_answer: dict = None
    multi_turn: bool = False


def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions[begin:end]


def load_model_answers(answer_dir: str):
    """Returns Dict[model_name -> Dict[question_id -> answer_dict]]."""
    filenames = glob.glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}
    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer
    return model_answers


def load_judge_prompts(prompt_file: str):
    prompts = {}
    with open(prompt_file) as fin:
        for line in fin:
            line = json.loads(line)
            prompts[line["name"]] = line
    return prompts


def get_model_list(answer_dir: str):
    files = glob.glob(os.path.join(answer_dir, "*.jsonl"))
    return sorted([os.path.basename(f)[:-6] for f in files])


def check_data(questions, model_answers, ref_answers, models, judges, ref_model: str = None):
    # verify all models have answers
    for m in models:
        assert m in model_answers, f"Missing answers for model {m}"
    if ref_model is not None and any(j.ref_based for j in judges.values()):
        assert ref_model in ref_answers, (
            f"Missing reference answers for ref_model '{ref_model}'. "
            f"Available: {list(ref_answers.keys())}"
        )


def chat_completion_openai(model, messages, temperature, max_tokens):
    client = openai.OpenAI()
    for _ in range(API_MAX_RETRY):
        try:
            token_kwarg = "max_completion_tokens" if model.startswith("gpt-5") else "max_tokens"
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                **{token_kwarg: max_tokens},
            )
            return response.choices[0].message.content
        except openai.RateLimitError:
            time.sleep(API_RETRY_SLEEP)
        except Exception as e:
            print(f"OpenAI error: {e}")
            time.sleep(API_RETRY_SLEEP)
    return API_ERROR_OUTPUT



def run_judge_single(question, answer, judge, ref_answer, multi_turn=False):
    kwargs = {}
    if ref_answer is not None:
        kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
        if multi_turn:
            kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]

    if multi_turn:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question_1=question["turns"][0],
            question_2=question["turns"][1],
            answer_1=answer["choices"][0]["turns"][0],
            answer_2=answer["choices"][0]["turns"][1],
            **kwargs,
        )
    else:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question=question["turns"][0],
            answer=answer["choices"][0]["turns"][0],
            **kwargs,
        )

    system_prompt = judge.prompt_template["system_prompt"]
    model = judge.model_name
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    rating = -1
    if model in OPENAI_MODEL_LIST:
        judgment = chat_completion_openai(model, messages, temperature=0, max_tokens=2048)
    else:
        raise ValueError(f"Invalid judge model: {model}")

    if judge.prompt_template["output_format"] == "[[rating]]":
        match = re.search(one_score_pattern, judgment)
        if not match:
            match = re.search(one_score_pattern_backup, judgment)
        rating = ast.literal_eval(match.groups()[0]) if match else -1
    else:
        raise ValueError(f"Invalid output format: {judge.prompt_template['output_format']}")

    return rating, user_prompt, judgment


def play_a_match_single(match: MatchSingle, output_file: str):
    question, model, answer, judge, ref_answer, multi_turn = (
        match.question, match.model, match.answer,
        match.judge, match.ref_answer, match.multi_turn,
    )

    if judge.prompt_template["type"] != "single":
        raise ValueError(f"Expected single judge type, got {judge.prompt_template['type']}")

    score, user_prompt, judgment = run_judge_single(
        question, answer, judge, ref_answer, multi_turn=multi_turn
    )

    turn = 1 if not multi_turn else 2
    result = {
        "question_id": question["question_id"],
        "model": model,
        "judge": (judge.model_name, judge.prompt_template["name"]),
        "user_prompt": user_prompt,
        "judgment": judgment,
        "score": score,
        "turn": turn,
        "tstamp": time.time(),
    }
    print(
        f"question: {question['question_id']}, turn: {turn}, model: {model}, "
        f"score: {score}"
    )

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as fout:
            fout.write(json.dumps(result) + "\n")

    return result


def run_judge_pair(question, answer_a, answer_b, judge, ref_answer, multi_turn=False):
    kwargs = {}
    if ref_answer is not None:
        kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
        if multi_turn:
            kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]

    if multi_turn:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question_1=question["turns"][0],
            question_2=question["turns"][1],
            answer_a_1=answer_a["choices"][0]["turns"][0],
            answer_b_1=answer_b["choices"][0]["turns"][0],
            answer_a_2=answer_a["choices"][0]["turns"][1],
            answer_b_2=answer_b["choices"][0]["turns"][1],
            **kwargs,
        )
    else:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question=question["turns"][0],
            answer_a=answer_a["choices"][0]["turns"][0],
            answer_b=answer_b["choices"][0]["turns"][0],
            **kwargs,
        )

    system_prompt = judge.prompt_template["system_prompt"]
    model = judge.model_name
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    winner = "error"
    if model in OPENAI_MODEL_LIST:
        judgment = chat_completion_openai(model, messages, temperature=0, max_tokens=2048)
    else:
        raise ValueError(f"Invalid judge model: {model}")

    fmt = judge.prompt_template["output_format"]
    if fmt == "[[A]]":
        if "[[A]]" in judgment:
            winner = "A"
        elif "[[B]]" in judgment:
            winner = "B"
        elif "[[C]]" in judgment:
            winner = "tie"
        else:
            winner = "error"
    elif fmt == "[[rating_a,rating_b]]":
        match = re.search(two_score_pattern, judgment) or re.search(two_score_pattern_backup, judgment)
        if match:
            scores = [ast.literal_eval(s.strip()) for s in match.groups()]
            if abs(scores[0] - scores[1]) <= TIE_DELTA:
                winner = "tie"
            elif scores[0] > scores[1]:
                winner = "A"
            else:
                winner = "B"
        else:
            winner = "error"
    else:
        raise ValueError(f"Invalid output format: {fmt}")

    return winner, user_prompt, judgment


def play_a_match_pair(match: MatchPair, output_file: str):
    question, model_1, model_2, answer_1, answer_2, judge, ref_answer, multi_turn = (
        match.question, match.model_1, match.model_2,
        match.answer_1, match.answer_2,
        match.judge, match.ref_answer, match.multi_turn,
    )

    if judge.prompt_template["type"] == "pairwise":
        g1_winner, g1_user_prompt, g1_judgment = run_judge_pair(
            question, answer_1, answer_2, judge, ref_answer, multi_turn=multi_turn
        )
        g2_winner, g2_user_prompt, g2_judgment = run_judge_pair(
            question, answer_2, answer_1, judge, ref_answer, multi_turn=multi_turn
        )

        g1_map = {"A": "model_1", "B": "model_2"}
        g2_map = {"A": "model_2", "B": "model_1"}
        g1_winner = g1_map.get(g1_winner, g1_winner)
        g2_winner = g2_map.get(g2_winner, g2_winner)
        turn = 1 if not multi_turn else 2

        result = {
            "question_id": question["question_id"],
            "model_1": model_1,
            "model_2": model_2,
            "g1_winner": g1_winner,
            "g2_winner": g2_winner,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "g1_user_prompt": g1_user_prompt,
            "g1_judgment": g1_judgment,
            "g2_user_prompt": g2_user_prompt,
            "g2_judgment": g2_judgment,
            "turn": turn,
            "tstamp": time.time(),
        }
        print(
            f"question: {question['question_id']}, turn: {turn}, "
            f"model_1: {model_1}, model_2: {model_2}, "
            f"g1_winner: {g1_winner}, g2_winner: {g2_winner}"
        )
    elif judge.prompt_template["type"] == "single":
        m1_score, m1_user_prompt, m1_judgment = run_judge_single(question, answer_1, judge, ref_answer)
        m2_score, m2_user_prompt, m2_judgment = run_judge_single(question, answer_2, judge, ref_answer)

        if abs(m1_score - m2_score) <= TIE_DELTA:
            winner = "tie"
        elif m1_score > m2_score:
            winner = "model_1"
        else:
            winner = "model_2"

        result = {
            "question_id": question["question_id"],
            "model_1": model_1,
            "model_2": model_2,
            "g1_winner": winner,
            "g2_winner": winner,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "g1_user_prompt": m1_user_prompt,
            "g1_judgment": m1_judgment,
            "g2_user_prompt": m2_user_prompt,
            "g2_judgment": m2_judgment,
            "m1_score": m1_score,
            "m2_score": m2_score,
            "tstamp": time.time(),
        }
    else:
        raise ValueError(f"Invalid judge type: {judge.prompt_template['type']}")

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as fout:
            fout.write(json.dumps(result) + "\n")

    return result
