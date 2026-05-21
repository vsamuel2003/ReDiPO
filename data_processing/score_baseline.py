"""Score instruct-model responses with Skywork RM (IF) and negative token-length-normalized
log-probability under the same instruct model. Drops base-model responses.

Two-stage pipeline to avoid co-residency of Skywork RM + instruct model on the GPU:
  Stage A: Skywork pass  ->  saves <output_file>.skywork.jsonl (intermediate)
  Stage B: NLL pass      ->  saves <output_file> (final)

Re-running after a Stage B crash skips Stage A automatically via the intermediate file.
"""

import argparse
import json
import logging
import os
import re
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scoring.instruction_following import _skywork_score, load_skywork

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def slug_of(model_path: str) -> str:
    """Derive a short slug from a HuggingFace model path."""
    basename = model_path.rsplit("/", 1)[-1].lower()
    if "llama" in basename:
        return "llama"
    if "qwen" in basename:
        return "qwen"
    if "olmo" in basename:
        return "olmo"
    return re.sub(r"[^a-z0-9]+", "_", basename).strip("_")


def is_instruct(model_name: str) -> bool:
    return "instruct" in model_name.lower()


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(entries: list[dict], path: str) -> None:
    dirpart = os.path.dirname(path)
    if dirpart:
        os.makedirs(dirpart, exist_ok=True)
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def intermediate_path(output_file: str) -> str:
    base = output_file[:-6] if output_file.endswith(".jsonl") else output_file
    return base + ".skywork.jsonl"


# ── Stage A ─────────────────────────────────────────────────────────────────

def stage_a_skywork(entries: list[dict], inter_path: str, device: str) -> list[dict]:
    """Score every entry with Skywork RM; unloads model when done."""
    if os.path.exists(inter_path):
        loaded = load_jsonl(inter_path)
        if len(loaded) == len(entries):
            logger.info(f"Skywork intermediate file complete ({len(loaded)} entries). Skipping Stage A.")
            return loaded
        logger.info(f"Skywork intermediate file partial ({len(loaded)}/{len(entries)}). Resuming Stage A.")
        already_scored = len(loaded)
    else:
        loaded = []
        already_scored = 0

    logger.info(f"Stage A: Scoring {len(entries) - already_scored} remaining entries with Skywork RM…")
    sw_model, sw_tokenizer = load_skywork(device)

    for i, entry in enumerate(tqdm(entries[already_scored:], desc="Skywork", initial=already_scored, total=len(entries))):
        score = _skywork_score(entry["prompt"], entry["output"], model=sw_model, tokenizer=sw_tokenizer)
        e = dict(entry)
        e["instruction_following_score"] = score
        loaded.append(e)
        if (i + 1) % 500 == 0:
            write_jsonl(loaded, inter_path)

    write_jsonl(loaded, inter_path)
    logger.info(f"Stage A complete → {inter_path}")

    del sw_model
    torch.cuda.empty_cache()
    return loaded


# ── Stage B ─────────────────────────────────────────────────────────────────

def compute_neg_logprob(
    entry: dict,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_length: int,
    device: str,
) -> float:
    """Return -(sum_response_logp / num_response_tokens) for one entry, or NaN if truncated."""
    prompt = entry["prompt"]
    output = entry["output"]

    # Chat-formatted prompt only (to measure prompt token length)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # Chat-formatted full conversation
    full_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": output}],
        add_generation_prompt=False,
        return_tensors="pt",
    )

    if full_ids.size(1) > max_length:
        return float("nan")

    prompt_len = prompt_ids.size(1)
    full_ids = full_ids.to(device)

    labels = full_ids.clone()
    labels[:, :prompt_len] = -100  # mask prompt tokens; only response counts

    with torch.no_grad():
        logits = model(full_ids).logits  # (1, seq, vocab)

    # Shift for next-token prediction
    logits = logits[:, :-1, :]   # (1, seq-1, vocab)
    targets = labels[:, 1:]       # (1, seq-1)
    mask = targets != -100

    if mask.sum() == 0:
        return float("nan")

    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    sum_logp = (token_logps * mask).sum().item()
    num_resp_tokens = mask.sum().item()
    return -(sum_logp / num_resp_tokens)


def stage_b_nll(
    entries: list[dict],
    output_path: str,
    model_path: str,
    max_length: int,
    device: str,
) -> list[dict]:
    """Compute NLL for all entries under the instruct model; saves final JSONL."""
    logger.info(f"Stage B: Computing NLL under {model_path} for {len(entries)} entries…")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    final = []
    skipped = 0
    for entry in tqdm(entries, desc="NLL"):
        nll = compute_neg_logprob(entry, model, tokenizer, max_length, device)
        if nll != nll:  # NaN
            skipped += 1
        e = dict(entry)
        e["neg_logprob_per_token"] = nll
        final.append(e)

    if skipped:
        logger.warning(f"{skipped}/{len(entries)} entries exceeded max_length={max_length}; NLL set to NaN.")

    write_jsonl(final, output_path)
    logger.info(f"Stage B complete → {output_path}")

    del model
    torch.cuda.empty_cache()
    return final


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_file", required=True, help="Input JSONL (prompt/output/model entries)")
    parser.add_argument("--model_path", required=True, help="HF model ID for the instruct model (NLL)")
    parser.add_argument("--output_file", required=True, help="Output scored JSONL (instruct-model rows only)")
    parser.add_argument("--slug", default=None, help="Model slug override (auto-derived if omitted)")
    parser.add_argument("--skywork_batch_size", type=int, default=8, help="Unused; reserved for future batching")
    parser.add_argument("--nll_batch_size", type=int, default=4, help="Unused; reserved for future batching")
    parser.add_argument("--max_length", type=int, default=2048, help="Max tokens; longer entries get NaN NLL")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    slug = args.slug or slug_of(args.model_path)
    logger.info(f"Model slug: {slug}")

    # Filter to instruct-model entries only
    all_entries = load_jsonl(args.input_file)
    entries = [e for e in all_entries if is_instruct(e.get("model", ""))]
    logger.info(
        f"Loaded {len(all_entries)} entries; dropped {len(all_entries) - len(entries)} base-model rows; "
        f"{len(entries)} instruct-model rows remain."
    )

    inter = intermediate_path(args.output_file)
    scored = stage_a_skywork(entries, inter, args.device)
    stage_b_nll(scored, args.output_file, args.model_path, args.max_length, args.device)


if __name__ == "__main__":
    main()
