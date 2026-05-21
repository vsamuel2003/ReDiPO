"""
Run AlpacaEval annotation on previously generated outputs.

Usage:
    python run_evaluation.py \\
        --outputs-path results/llama-3-8b-instruct/outputs.json \\
        [--annotator weighted_alpaca_eval_gpt4_0613] \\
        [--output-dir results/llama-3-8b-instruct] \\
        [--first-call-log results/llama-3-8b-instruct/first_annotation.json]

Annotator resolution:
  - A bare name (no "/" or ".") → looked up in evals/alpaca_eval_suite/configs/<name>/configs.yaml
  - Any path containing "/" or ending in ".yaml" → used as-is (absolute or relative)

The first annotator API call is logged to --first-call-log for debugging (model name,
prompt sent, raw response with logprobs). Subsequent calls are not logged.
"""

import argparse
import json
import os
import threading


# -------------------------------------------------------------------
# Annotator path resolution
# -------------------------------------------------------------------

_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def resolve_annotator(name: str) -> str:
    """Return an absolute path to configs.yaml for the given annotator name or path."""
    if "/" in name or name.endswith(".yaml"):
        return os.path.abspath(name)
    candidate = os.path.join(_CONFIGS_DIR, name, "configs.yaml")
    if os.path.exists(candidate):
        return candidate
    # Fall back to name as-is (built-in alpaca_eval config by name)
    return name


# -------------------------------------------------------------------
# First-call debug logger (monkey-patch at OpenAI SDK level)
# -------------------------------------------------------------------

def _install_gpt5_compat_patch() -> None:
    """Rename max_tokens → max_completion_tokens for gpt-5.x models at the SDK level."""
    try:
        import openai
    except ImportError:
        return
    _orig = openai.resources.chat.completions.Completions.create

    def _patched(self, *args, **kwargs):
        if kwargs.get("model", "").startswith("gpt-5") and "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        return _orig(self, *args, **kwargs)

    openai.resources.chat.completions.Completions.create = _patched


def _install_first_call_logger(log_path: str) -> None:
    """Patch openai.resources.chat.completions.Completions.create to log the first call."""
    try:
        import openai
    except ImportError:
        return

    _lock = threading.Lock()
    _state = {"done": False}
    _orig = openai.resources.chat.completions.Completions.create

    def _patched(self, *args, **kwargs):
        resp = _orig(self, *args, **kwargs)
        with _lock:
            if not _state["done"]:
                try:
                    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
                    payload = {"request": kwargs, "response": resp.model_dump()}
                    with open(log_path, "w") as f:
                        json.dump(payload, f, indent=2, default=str)
                    print(f"[debug] first annotator call logged to {log_path}")
                except Exception as exc:
                    print(f"[debug] WARNING: could not write first-call log: {exc}")
                finally:
                    _state["done"] = True
        return resp

    openai.resources.chat.completions.Completions.create = _patched


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main(args):
    try:
        import alpaca_eval
    except ImportError:
        raise ImportError(
            "alpaca_eval is not installed. Run: pip install alpaca-eval"
        )

    with open(args.outputs_path) as f:
        model_outputs = json.load(f)

    annotators_config = resolve_annotator(args.annotator)
    print(f"Evaluating {len(model_outputs)} outputs with annotator: {annotators_config}")

    _install_gpt5_compat_patch()
    _install_first_call_logger(args.first_call_log)

    df_leaderboard, annotations = alpaca_eval.evaluate(
        model_outputs=model_outputs,
        annotators_config=annotators_config,
        output_path=args.output_dir,
        is_return_instead_of_print=True,
        precomputed_leaderboard=None,
        caching_path=os.path.join(args.output_dir, "annotations_cache.json"),
    )

    print("\n=== AlpacaEval Results ===")
    print(df_leaderboard.to_string())

    os.makedirs(args.output_dir, exist_ok=True)
    leaderboard_path = os.path.join(args.output_dir, "leaderboard.json")
    df_leaderboard.to_json(leaderboard_path, orient="index", indent=2)
    print(f"\nLeaderboard saved to {leaderboard_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-path", type=str, required=True,
                        help="Path to outputs.json produced by generate_outputs.py")
    parser.add_argument("--annotator", type=str,
                        default="weighted_alpaca_eval_gpt4_0613",
                        help=(
                            "Annotator config. Bare name → looked up in "
                            "alpaca_eval_suite/configs/<name>/configs.yaml. "
                            "A path (contains '/' or ends in '.yaml') → used as-is."
                        ))
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save leaderboard (defaults to outputs-path parent dir)")
    parser.add_argument("--first-call-log", type=str, default=None,
                        help="Path to write the first annotator API call for debugging "
                             "(default: <output-dir>/first_annotation.json)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.outputs_path))

    if args.first_call_log is None:
        args.first_call_log = os.path.join(args.output_dir, "first_annotation.json")

    main(args)
