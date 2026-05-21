"""
Generate MTBench model answers using the unified common layer.

Supports mid-run resume: if the answer file already exists, question_ids
whose answers have the correct number of turns are skipped.

Usage:
    python gen_model_answer.py \\
        --model-path meta-llama/Llama-3-8B-Instruct \\
        --model-id llama-3-8b-instruct \\
        --model-type instruct \\
        [--lora /path/to/adapter] \\
        [--output-dir data/mt_bench/model_answer] \\
        [--max-new-tokens 1024] \\
        [--num-choices 1]
"""

import argparse
import json
import os
import random
import sys
import time

import shortuuid
import torch
from tqdm import tqdm

# Allow imports from unified_evals root (for common/) and mtbench/ (for judge_utils)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from common.model_loader import load_model_and_tokenizer
from common.prompt_builder import build_prompt
from common.generate import generate as gen_completions
from judge_utils import load_questions, temperature_config


def _load_completed(answer_file: str, num_choices: int) -> dict:
    """Return {question_id: answer_dict} for fully-answered questions."""
    if not os.path.exists(answer_file):
        return {}
    completed = {}
    with open(answer_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = obj.get("question_id")
            choices = obj.get("choices", [])
            turns = choices[0].get("turns", []) if choices else []
            if qid is not None and len(choices) == num_choices and "ERROR" not in turns:
                completed[qid] = obj
    return completed


def run_eval(args):
    questions = load_questions(args.question_file, args.question_begin, args.question_end)
    random.shuffle(questions)

    answer_file = os.path.join(args.output_dir, f"{args.model_id}.jsonl")
    completed = _load_completed(answer_file, args.num_choices)
    if completed:
        print(f"Resuming: {len(completed)} questions already answered, "
              f"{len(questions) - len(completed)} remaining.")

    pending = []
    for q in questions:
        qid = q["question_id"]
        if qid in completed:
            # Verify the number of turns matches (guards against partial multi-turn)
            existing_turns = completed[qid]["choices"][0].get("turns", [])
            if len(existing_turns) == len(q["turns"]):
                continue
        pending.append(q)

    if not pending:
        print("All questions already answered. Skipping model loading.")
        reorg_answer_file(answer_file)
        return

    model, tokenizer, model_type = load_model_and_tokenizer(
        model_path=args.model_path,
        model_type=args.model_type,
        lora_path=args.lora,
        dtype=torch.bfloat16,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for question in tqdm(pending):
        temperature = temperature_config.get(question["category"], 0.7)

        choices = []
        for choice_idx in range(args.num_choices):
            torch.manual_seed(choice_idx)
            conv_messages: list[dict] = []
            turns: list[str] = []

            for turn_idx, turn_text in enumerate(question["turns"]):
                conv_messages.append({"role": "user", "content": turn_text})

                prompt = build_prompt(
                    tokenizer=tokenizer,
                    messages=conv_messages,
                    model_type=model_type,
                    model_path=args.model_path,
                )

                try:
                    outputs = gen_completions(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        model_type=model_type,
                        max_new_tokens=args.max_new_tokens,
                        temperature=temperature,
                        n=1,
                    )
                    output = outputs[0]
                except RuntimeError as e:
                    print(f"ERROR question_id={question['question_id']} turn={turn_idx}: {e}")
                    output = "ERROR"

                conv_messages.append({"role": "assistant", "content": output})
                turns.append(output)

            choices.append({"index": choice_idx, "turns": turns})

        ans_json = {
            "question_id": question["question_id"],
            "answer_id": shortuuid.uuid(),
            "model_id": args.model_id,
            "choices": choices,
            "tstamp": time.time(),
        }
        with open(answer_file, "a") as fout:
            fout.write(json.dumps(ans_json) + "\n")

    print(f"Answers written to {answer_file}")


def reorg_answer_file(answer_file):
    """De-duplicate by question_id, keeping the last entry."""
    answers = {}
    with open(answer_file, "r") as fin:
        for line in fin:
            obj = json.loads(line)
            answers[obj["question_id"]] = line
    with open(answer_file, "w") as fout:
        for line in sorted(answers.values()):
            fout.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="HuggingFace model ID or local path (base model for LoRA)")
    parser.add_argument("--model-id", type=str, required=True,
                        help="Short identifier written into the output filename")
    parser.add_argument("--model-type", type=str, default="instruct",
                        choices=["base", "instruct", "lora"],
                        help="Model type: base | instruct | lora")
    parser.add_argument("--lora", type=str, default=None,
                        help="Path to LoRA adapter directory (required when --model-type lora)")
    parser.add_argument("--question-file", type=str,
                        default=os.path.join(os.path.dirname(__file__), "data/question.jsonl"))
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "data/model_answer"))
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--question-begin", type=int, default=None)
    parser.add_argument("--question-end", type=int, default=None)
    args = parser.parse_args()

    if args.model_type == "lora" and not args.lora:
        parser.error("--lora is required when --model-type is 'lora'")

    run_eval(args)

    answer_file = os.path.join(args.output_dir, f"{args.model_id}.jsonl")
    reorg_answer_file(answer_file)
