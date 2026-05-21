"""Score marginal diversity for all responses in a grouped embedded JSONL.

Adds a 'marginal_diversity' field to each response.  By default overwrites the
input file in-place; pass --output_file to write to a separate path instead.

Three diversity methods:
  centroid     - 1 - sim(e_i, e_c)  where e_c is the prompt centroid
  maxsim       - 1 - max_{j!=i} sim(e_i, e_j)  most similar other response
  set_coverage - U(S) - U(S\\{y_i}), self-excluded facility-location marginal

Usage:
    python data_processing/score_diversity.py \
        --input_file filtered_data/pilot_armro-filtered_embedded.jsonl \
        --embedding_method bge \
        --diversity_method centroid \
        [--output_file filtered_data/pilot_armro-filtered_embedded_diversity.jsonl]
"""

import argparse
import logging
import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
# Import diversity directly to avoid loading scoring/__init__.py (which requires torch)
sys.path.insert(0, os.path.join(_root, "scoring"))
from diversity import score_marginal_diversity  # noqa: E402
from data_processing.embed_responses import load_grouped, save_grouped
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def score_all_responses(
    records: list[dict],
    embedding_method: str,
    diversity_method: str,
) -> list[dict]:
    """Add marginal_diversity to every response in every record.

    Args:
        records:          Grouped JSONL records.
        embedding_method: Which embedding to use: "bge" or "openai".
        diversity_method: Scoring method: "centroid", "maxsim", or "set_coverage".

    Returns:
        records with marginal_diversity added to each response.
    """
    embed_field = f"embedding_{embedding_method}"
    centroid_field = f"centroid_{embedding_method}"

    skipped_records = 0
    scored_responses = 0

    for record in tqdm(records, desc="Scoring diversity", unit="prompt"):
        responses = record["responses"]

        # Collect embeddings for this prompt
        embeddings = [r.get(embed_field) for r in responses]
        missing = [i for i, e in enumerate(embeddings) if e is None]
        if missing:
            logger.warning(
                f"Prompt has {len(missing)}/{len(responses)} responses missing {embed_field}. "
                f"Skipping entire prompt."
            )
            skipped_records += 1
            continue

        if diversity_method == "centroid":
            centroid = record.get(centroid_field)
            if centroid is None:
                logger.warning(
                    f"Prompt missing {centroid_field}. "
                    f"Run backfill_centroids.py first or use diversity_method=maxsim. Skipping."
                )
                skipped_records += 1
                continue
            for response in responses:
                response["marginal_diversity"] = score_marginal_diversity(
                    method="centroid",
                    embedding=response[embed_field],
                    centroid=centroid,
                )
                scored_responses += 1

        elif diversity_method in ("maxsim", "set_coverage"):
            for i, response in enumerate(responses):
                response["marginal_diversity"] = score_marginal_diversity(
                    method=diversity_method,
                    embedding=response[embed_field],
                    all_embeddings=embeddings,
                    index=i,
                )
                scored_responses += 1

    if skipped_records:
        logger.warning(f"Skipped {skipped_records} prompts due to missing data.")
    logger.info(f"Scored marginal_diversity for {scored_responses} responses across {len(records) - skipped_records} prompts.")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Score marginal diversity for all responses in an embedded grouped JSONL."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Embedded grouped JSONL")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSONL path. Defaults to input_file (in-place overwrite) if not set.",
    )
    parser.add_argument(
        "--embedding_method",
        type=str,
        choices=["bge", "openai"],
        required=True,
        help="Which embedding to use for scoring",
    )
    parser.add_argument(
        "--diversity_method",
        type=str,
        choices=["centroid", "maxsim", "set_coverage"],
        default="centroid",
        help="Diversity scoring method: centroid (default), maxsim, or set_coverage",
    )
    args = parser.parse_args()

    records = load_grouped(args.input_file)
    records = score_all_responses(records, args.embedding_method, args.diversity_method)
    output_path = args.output_file or args.input_file
    save_grouped(records, output_path)
    logger.info(f"Written to {output_path}")


if __name__ == "__main__":
    main()
