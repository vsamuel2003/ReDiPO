"""Add embeddings to grouped .jsonl responses.

Supports two embedding models:
  - openai: text-embedding-3-large (3072-dim), stored as 'embedding_openai'
  - bge:    BAAI/bge-m3 (1024-dim),            stored as 'embedding_bge'
  - both:   runs bge first, then openai

Resume-safe: skips responses that already have the target embedding field.
"""

import argparse
import asyncio
import json
import logging
import os

import numpy as np

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

OPENAI_MODEL = "text-embedding-3-large"
BGE_MODEL = "BAAI/bge-m3"
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_grouped(input_file: str) -> list[dict]:
    records = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} prompt groups from {input_file}")
    return records


def save_grouped(records: list[dict], output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    logger.info(f"Saved {len(records)} prompt groups to {output_file}")


# ---------------------------------------------------------------------------
# OpenAI embeddings (async, with semaphore for concurrency control)
# ---------------------------------------------------------------------------

async def _embed_single_openai(client, semaphore: asyncio.Semaphore, text: str) -> list[float]:
    """Embed a single text string via OpenAI API, with retries."""
    text = text.replace("\n", " ")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with semaphore:
                response = await client.embeddings.create(input=[text], model=OPENAI_MODEL)
            return response.data[0].embedding
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            logger.warning(f"OpenAI embedding attempt {attempt} failed ({e}). Retrying in {wait}s...")
            await asyncio.sleep(wait)


async def embed_openai(records: list[dict], concurrency: int = 20) -> list[dict]:
    """Add 'embedding_openai' to each response that doesn't already have it."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    semaphore = asyncio.Semaphore(concurrency)

    # Collect all (record_idx, response_idx, text) tuples that need embedding
    tasks_meta = []
    for r_idx, record in enumerate(records):
        for s_idx, response in enumerate(record["responses"]):
            if "embedding_openai" not in response:
                tasks_meta.append((r_idx, s_idx, response["output"]))

    if not tasks_meta:
        logger.info("OpenAI: all responses already have embeddings, nothing to do.")
        return records

    logger.info(f"OpenAI: embedding {len(tasks_meta)} responses...")

    async def run_one(r_idx, s_idx, text):
        embedding = await _embed_single_openai(client, semaphore, text)
        return r_idx, s_idx, embedding

    results = []
    with tqdm(total=len(tasks_meta), desc="OpenAI embeddings", unit="response") as pbar:
        # Process in batches to keep tqdm updating smoothly
        batch_size = concurrency * 2
        for i in range(0, len(tasks_meta), batch_size):
            batch = tasks_meta[i:i + batch_size]
            batch_tasks = [run_one(r, s, t) for r, s, t in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for j, result in enumerate(batch_results):
                if isinstance(result, BaseException):
                    r_idx, s_idx, _ = batch[j]
                    logger.error(f"Failed to embed response [{r_idx}][{s_idx}]: {result}")
                else:
                    results.append(result)
            pbar.update(len(batch))

    for r_idx, s_idx, embedding in results:
        records[r_idx]["responses"][s_idx]["embedding_openai"] = embedding

    logger.info(f"OpenAI: added embeddings to {len(results)} responses.")
    return records


# ---------------------------------------------------------------------------
# BGE embeddings (local, batched)
# ---------------------------------------------------------------------------

def embed_bge(records: list[dict], batch_size: int = 12, max_length: int = 8192) -> list[dict]:
    """Add 'embedding_bge' to each response that doesn't already have it."""
    from FlagEmbedding import BGEM3FlagModel

    # Collect all (record_idx, response_idx, text) tuples that need embedding
    tasks_meta = []
    for r_idx, record in enumerate(records):
        for s_idx, response in enumerate(record["responses"]):
            if "embedding_bge" not in response:
                tasks_meta.append((r_idx, s_idx, response["output"]))

    if not tasks_meta:
        logger.info("BGE: all responses already have embeddings, nothing to do.")
        return records

    logger.info(f"BGE: loading {BGE_MODEL}...")
    model = BGEM3FlagModel(BGE_MODEL, use_fp16=True)

    texts = [t for _, _, t in tasks_meta]
    logger.info(f"BGE: encoding {len(texts)} responses in batches of {batch_size}...")

    all_embeddings = []
    with tqdm(total=len(texts), desc="BGE embeddings", unit="response") as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            result = model.encode(batch_texts, batch_size=batch_size, max_length=max_length)
            batch_vecs = result["dense_vecs"]
            all_embeddings.extend(batch_vecs.tolist())
            pbar.update(len(batch_texts))

    for (r_idx, s_idx, _), embedding in zip(tasks_meta, all_embeddings):
        records[r_idx]["responses"][s_idx]["embedding_bge"] = embedding

    logger.info(f"BGE: added embeddings to {len(all_embeddings)} responses.")
    return records


# ---------------------------------------------------------------------------
# Centroid computation
# ---------------------------------------------------------------------------

def compute_centroids(records: list[dict], methods: list[str]) -> list[dict]:
    """Compute mean embedding (centroid) per prompt for each embedding method.

    For each record, averages all response embeddings for each method and stores
    the result as a top-level field: centroid_bge, centroid_openai, etc.

    Args:
        records: Grouped JSONL records with embedded responses.
        methods: Embedding method names, e.g. ["bge", "openai"].

    Returns:
        records with centroid_{method} added at the top level.
    """
    for method in methods:
        field = f"embedding_{method}"
        centroid_field = f"centroid_{method}"
        skipped = 0
        for record in tqdm(records, desc=f"Computing centroids ({method})", unit="prompt"):
            embeddings = [r[field] for r in record["responses"] if field in r]
            if not embeddings:
                skipped += 1
                continue
            record[centroid_field] = np.mean(embeddings, axis=0).tolist()
        if skipped:
            logger.warning(f"compute_centroids({method}): skipped {skipped} records with no embeddings.")
        logger.info(f"compute_centroids: stored {centroid_field} on {len(records) - skipped} records.")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Add embeddings to grouped response .jsonl.")
    parser.add_argument("--input_file", type=str, required=True, help="Grouped JSONL (one prompt per line)")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL with embeddings added")
    parser.add_argument(
        "--model",
        type=str,
        choices=["openai", "bge", "both"],
        default="both",
        help="Embedding model to use (default: both)",
    )
    parser.add_argument("--concurrency", type=int, default=20, help="OpenAI concurrent requests (default: 20)")
    parser.add_argument("--bge_batch_size", type=int, default=12, help="BGE encoding batch size (default: 12)")
    args = parser.parse_args()

    records = load_grouped(args.input_file)

    if args.model in ("bge", "both"):
        records = embed_bge(records, batch_size=args.bge_batch_size)
        # Save intermediate progress after BGE in case OpenAI step is interrupted
        if args.model == "both":
            save_grouped(records, args.output_file)
            logger.info("Intermediate save after BGE complete.")

    if args.model in ("openai", "both"):
        records = asyncio.run(embed_openai(records, concurrency=args.concurrency))

    methods = {"bge": ["bge"], "openai": ["openai"], "both": ["bge", "openai"]}[args.model]
    records = compute_centroids(records, methods)

    save_grouped(records, args.output_file)

    # Summary
    total_responses = sum(len(r["responses"]) for r in records)
    has_openai = sum(
        1 for r in records for s in r["responses"] if "embedding_openai" in s
    )
    has_bge = sum(
        1 for r in records for s in r["responses"] if "embedding_bge" in s
    )
    logger.info(
        f"Done. {len(records)} prompts, {total_responses} responses. "
        f"embedding_openai: {has_openai}, embedding_bge: {has_bge}"
    )


if __name__ == "__main__":
    main()
