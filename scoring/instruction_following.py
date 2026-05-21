"""Instruction following scoring for prompt-response pairs."""

import os
import time
import logging

import torch
from openai import OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL = "gpt-5.4-nano"
MAX_RETRIES = 3
ARMORM_MODEL_ID = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
SKYWORK_MODEL_ID = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"

JUDGE_HARD_PROMPT = """You are evaluating a piece of text. Determine if the response meets BOTH criteria:
1. Well-formatted: reasonable punctuation, grammar, and coherence (not gibberish)
2. Plausible: the response is a reasonable answer to the given prompt. You are not judging full correctness here. Only that the response is roughly on topic.

User Prompt: {prompt}
Response: {response}

Does this response meet both criteria? Answer with exactly "Yes" or "No".

---
Examples:

User Prompt: Explain how photosynthesis works.
Response: Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen. It occurs primarily in the chloroplasts of leaf cells.
Yes

User Prompt: What is the speed of light?
Response: asdkjh the light goes wooosh very fast colors rainbow spacetime brrr quantum
No

User Prompt: Write a short poem about the ocean.
Response: The stock market experienced a significant downturn in Q3 2024 due to rising interest rates and geopolitical tensions.
No

User Prompt: What are the benefits of exercise?
Response: Regular exercise improves cardiovascular health, boosts mood through endorphin release, helps maintain a healthy weight, and can reduce the risk of chronic diseases like diabetes and heart disease.
Yes"""

JUDGE_SOFT_PROMPT = """You are evaluating an AI response on a scale from 1 to 5. Consider these factors:
- Coherence: Is the response logically structured and easy to follow?
- Tone: Is the tone appropriate for the given prompt?
- Quality: How well does the response address the prompt?

User Prompt: {prompt}
AI Response: {response}

Provide a single score from 1 to 5 (1=very poor, 5=excellent). Respond with only the number.

---
Examples:

User Prompt: What causes rain?
AI Response: Rain forms when water vapor in the atmosphere condenses into droplets within clouds. When these droplets grow heavy enough, they fall as precipitation. This is part of the water cycle, driven primarily by solar energy.
5

User Prompt: Recommend a good book.
AI Response: Book good read yes. Maybe try one from shelf. Reading is activity people do sometimes with pages.
1

User Prompt: How do I improve my writing skills?
AI Response: Practice writing regularly. Read more books.
3

User Prompt: Explain quantum computing.
AI Response: Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to perform computations. Unlike classical bits that are 0 or 1, qubits can exist in multiple states simultaneously, enabling certain problems to be solved exponentially faster. However, quantum computers are still in early stages and face challenges like decoherence and error correction.
5"""


def _call_openai(formatted_prompt: str) -> str:
    """Make an OpenAI API call with retry logic."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                logger.warning(f"OpenAI API error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"OpenAI API failed after {MAX_RETRIES} attempts: {e}")
                raise

    return ""


def _judge_hard(prompt: str, response: str) -> bool:
    """LLM-as-judge: is the response well-formatted and plausible? Returns bool."""
    formatted = JUDGE_HARD_PROMPT.format(prompt=prompt, response=response)
    answer = _call_openai(formatted).lower()
    if answer.startswith("yes"):
        return True
    elif answer.startswith("no"):
        return False
    else:
        logger.warning(f"Unexpected judge_hard response: '{answer}', treating as False")
        return False


def _judge_soft(prompt: str, response: str) -> float:
    """LLM-as-judge: score response 1-5 for coherence, tone, and quality."""
    formatted = JUDGE_SOFT_PROMPT.format(prompt=prompt, response=response)
    answer = _call_openai(formatted)
    try:
        score = float(answer)
        score = max(1.0, min(5.0, score))
        return score
    except ValueError:
        logger.warning(f"Could not parse judge_soft score: '{answer}', defaulting to 1.0")
        return 1.0


def load_armorm(device: str = "cuda"):
    """Load the ArmoRM model and tokenizer. Call once and pass to _armorm_score."""
    model = AutoModelForSequenceClassification.from_pretrained(
        ARMORM_MODEL_ID,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(ARMORM_MODEL_ID, use_fast=True)
    return model, tokenizer


def _armorm_score(prompt: str, response: str, model=None, tokenizer=None) -> float:
    """Score using ArmoRM. Averages helpfulness, correctness, coherence, complexity
    and normalizes from HelpSteer 0-5 scale to 1-5.

    Args:
        prompt: The user prompt.
        response: The model response.
        model: Pre-loaded ArmoRM model (optional, will load if None).
        tokenizer: Pre-loaded tokenizer (optional, will load if None).

    Returns:
        Float score in range [1, 5].
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_armorm()

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model(input_ids)
        multi_obj_rewards = output.rewards.cpu().float()

    # Indices 0-3: helpfulness, correctness, coherence, complexity
    helpsteer_raw = multi_obj_rewards[0, :4] * 5 - 0.5  # Convert to 0-5 scale
    avg_score = helpsteer_raw.mean().item()

    # Normalize from [0, 5] to [1, 5]
    normalized = avg_score * 0.8 + 1.0
    normalized = max(1.0, min(5.0, normalized))

    return normalized


def load_skywork(device: str = "cuda"):
    """Load the Skywork reward model and tokenizer. Call once and pass to _skywork_score."""
    model = AutoModelForSequenceClassification.from_pretrained(
        SKYWORK_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(SKYWORK_MODEL_ID)
    return model, tokenizer


def _skywork_score(prompt: str, response: str, model=None, tokenizer=None) -> float:
    """Score using Skywork reward model. Returns raw scalar logit (can be negative).

    Args:
        prompt: The user prompt.
        response: The model response.
        model: Pre-loaded Skywork model (optional, will load if None).
        tokenizer: Pre-loaded tokenizer (optional, will load if None).

    Returns:
        Raw float scalar from reward model logits.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_skywork()

    conv = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(conv, tokenize=False)
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token):]
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        return model(**inputs).logits[0][0].item()


def score_instruction_following(
    prompt: str,
    response: str,
    method: str,
    armorm_model=None,
    armorm_tokenizer=None,
    skywork_model=None,
    skywork_tokenizer=None,
) -> bool | float:
    """Score instruction following quality.

    Args:
        prompt: The user prompt.
        response: The model response.
        method: "judge_hard" (bool), "armorm" (float 1-5), "judge_soft" (float 1-5),
                or "skywork" (raw float scalar).
        armorm_model: Pre-loaded ArmoRM model (optional, for armorm method).
        armorm_tokenizer: Pre-loaded ArmoRM tokenizer (optional, for armorm method).
        skywork_model: Pre-loaded Skywork model (optional, for skywork method).
        skywork_tokenizer: Pre-loaded Skywork tokenizer (optional, for skywork method).

    Returns:
        bool for judge_hard, float (1-5) for armorm/judge_soft, raw float for skywork.
    """
    if method == "judge_hard":
        return _judge_hard(prompt, response)
    elif method == "armorm":
        return _armorm_score(prompt, response, model=armorm_model, tokenizer=armorm_tokenizer)
    elif method == "judge_soft":
        return _judge_soft(prompt, response)
    elif method == "skywork":
        return _skywork_score(prompt, response, model=skywork_model, tokenizer=skywork_tokenizer)
    else:
        raise ValueError(f"Unknown IF method: {method}. Must be 'judge_hard', 'armorm', 'judge_soft', or 'skywork'.")
