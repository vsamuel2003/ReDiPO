"""
Evaluate HarmBench completions with the HarmBench classifier.

Uses HuggingFace transformers text-generation pipeline only (no vLLM).
Supports mid-run resume: behaviors already present in the results file with
all entries labelled are skipped. Results are written after each behavior.

Usage:
    python evaluate_completions.py \\
        --completions-path results/completions.json \\
        --save-path results/results.json \\
        [--cls-path cais/HarmBench-Llama-2-13b-cls] \\
        [--behaviors-path data/behavior_datasets/harmbench_behaviors_text_all.csv] \\
        [--num-tokens 512]
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from eval_utils import (
    compute_results_advbench,
    compute_results_classifier,
    compute_results_hashing,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_existing_results(save_path: str) -> dict:
    if not os.path.exists(save_path):
        return {}
    try:
        with open(save_path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _all_labelled(entries: list) -> bool:
    return all("label" in e for e in entries)


def _checkpoint(results: dict, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tmp = save_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=4)
    os.replace(tmp, save_path)


def main(args):
    # Load behaviors
    with open(args.behaviors_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        behaviors = {row["BehaviorID"]: row for row in reader}

    # Load completions
    with open(args.completions_path, "r") as f:
        completions = json.load(f)

    # Load existing results for resume
    results = _load_existing_results(args.save_path)

    # Determine which behaviors still need evaluation
    pending_behaviors = []
    for behavior_id in completions:
        if behavior_id not in behaviors:
            print(f"Behavior {behavior_id} not found in behaviors file; skipping")
            continue
        existing = results.get(behavior_id, [])
        if existing and _all_labelled(existing):
            continue
        pending_behaviors.append(behavior_id)

    skipped = len(completions) - len(pending_behaviors)
    if skipped:
        print(f"Resuming: {skipped} behaviors already classified, "
              f"{len(pending_behaviors)} remaining.")

    if not pending_behaviors:
        print("All behaviors already classified.")
    else:
        # Clip generations to num_tokens
        clip_tokenizer = AutoTokenizer.from_pretrained(args.cls_path)
        clip_tokenizer.truncation_side = "right"
        for behavior_id, completion_list in completions.items():
            for completion in completion_list:
                tokenized = clip_tokenizer.encode(
                    completion["generation"],
                    max_length=args.num_tokens,
                    truncation=True,
                )
                completion["generation"] = clip_tokenizer.decode(tokenized, skip_special_tokens=True)

        # Load classifier once
        print(f"Loading classifier: {args.cls_path}")
        _cls_pipe = pipeline(
            "text-generation",
            model=args.cls_path,
            tokenizer=args.cls_path,
            device_map="auto",
            torch_dtype=torch.float16,
            max_new_tokens=1,
            do_sample=False,
            return_full_text=False,
            trust_remote_code=True,
        )

        def cls_pipeline(inputs):
            return _cls_pipe(inputs, batch_size=8)

        for behavior_id in tqdm(pending_behaviors):
            data = completions[behavior_id]
            behavior_dict = behaviors[behavior_id]
            tags = behavior_dict["Tags"].split(", ")

            if "hash_check" in tags:
                current_results = compute_results_hashing(behavior_dict, data)
            else:
                current_results = compute_results_classifier(behavior_dict, data, cls_pipeline, args.num_tokens)

            if args.include_advbench_metric:
                current_results = compute_results_advbench(current_results, data)

            results[behavior_id] = current_results
            _checkpoint(results, args.save_path)

    # Print summary over all results (including previously cached)
    success_rates = []
    for i, (behavior_id, data) in enumerate(results.items()):
        asr = np.mean([d["label"] == 1 for d in data])
        print(f"{i}. {behavior_id} === ASR: {asr:.3f}")
        success_rates.append(asr)

    if success_rates:
        print(f"\nAverage ASR: {np.mean(success_rates):.3f}")

    print(f"Results saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls-path", type=str, default="cais/HarmBench-Llama-2-13b-cls")
    parser.add_argument("--completions-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--behaviors-path", type=str,
                        default=os.path.join(DATA_DIR, "behavior_datasets/harmbench_behaviors_text_all.csv"))
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--include-advbench-metric", action="store_true")
    args = parser.parse_args()
    main(args)
