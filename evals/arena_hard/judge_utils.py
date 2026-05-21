"""
Arena-Hard judge utilities: prompt templates, verdict parsing, and scoring.

Vendored from arena-hard-auto/utils/judge_utils.py and arena-hard-v2.0.yaml.
Reuses chat_completion_openai from evals/mtbench/judge_utils.py (handles
max_completion_tokens for gpt-5* and max_tokens for all others automatically).
"""

import re
import sys
import os

# Allow importing from sibling evals directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mtbench.judge_utils import chat_completion_openai  # noqa: F401

SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants "
    "to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. "
    "Your job is to evaluate which assistant's answer is better.\n\n"
    "Begin your evaluation by generating your own answer to the prompt. You must provide your answers before "
    "judging any answers.\n\n"
    "When evaluating the assistants' answers, compare both assistants' answers with your answer. You must "
    "identify and correct any mistakes or inaccurate information.\n\n"
    "Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer "
    "correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or "
    "more than one interpretation, it is more helpful and appropriate to ask for clarifications or more "
    "information from the user than providing an answer based on assumptions. Relevant means all parts of the "
    "response closely connect or are appropriate to what is being asked. Concise means the response is clear "
    "and not verbose or excessive.\n\n"
    "Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any "
    "missing important information in the assistants' answers that would be beneficial to include when "
    "responding to the user prompt.\n\n"
    "After providing your explanation, you must output only one of the following choices as your final verdict "
    "with a label:\n\n"
    "1. Assistant A is significantly better: [[A>>B]]\n"
    "2. Assistant A is slightly better: [[A>B]]\n"
    "3. Tie, relatively the same: [[A=B]]\n"
    "4. Assistant B is slightly better: [[B>A]]\n"
    "5. Assistant B is significantly better: [[B>>A]]\n\n"
    'Example output: "My final verdict is tie: [[A=B]]".'
)

USER_TEMPLATE = (
    "<|User Prompt|>\n{QUESTION}\n\n"
    "<|The Start of Assistant A's Answer|>\n{ANSWER_A}\n<|The End of Assistant A's Answer|>\n\n"
    "<|The Start of Assistant B's Answer|>\n{ANSWER_B}\n<|The End of Assistant B's Answer|>"
)

# Ordered from most-specific to least-specific
_VERDICT_PATTERNS = [
    re.compile(r"\[\[([AB<>=]+)\]\]"),
    re.compile(r"\[([AB<>=]+)\]"),
]

# Normalized forms of each verdict
_VERDICT_NORM = {
    "A>>B": "A>>B",
    "A>B":  "A>B",
    "A=B":  "A=B",
    "B=A":  "A=B",
    "B>A":  "B>A",
    "B>>A": "B>>A",
}


def parse_verdict(judgment: str) -> str:
    """Extract verdict string from judge output. Returns 'error' if unparseable."""
    for pattern in _VERDICT_PATTERNS:
        m = pattern.search(judgment)
        if m:
            raw = m.group(1).strip()
            return _VERDICT_NORM.get(raw, "error")
    return "error"


# Score from the perspective of Model B (our model) in a round where B is being judged.
_SCORE_FOR_B = {
    "A>>B": 0.00,
    "A>B":  0.25,
    "A=B":  0.50,
    "B>A":  0.75,
    "B>>A": 1.00,
    "error": float("nan"),
}


def score_for_b(verdict: str) -> float:
    """Return our model's score [0,1] when it plays the role of 'B' in a round."""
    return _SCORE_FOR_B.get(verdict, float("nan"))
