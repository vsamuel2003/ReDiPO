"""Diversity scoring for response embeddings.

Three methods:
  - centroid:      1 - cosine_similarity(e_i, e_c)  where e_c is the prompt centroid
  - maxsim:        1 - max_{j≠i} cosine_similarity(e_i, e_j)  over all other responses
  - set_coverage:  U(S) - U(S \\ {y_i}), self-excluded facility-location utility
                   U(A) = Σ_{y∈S} max_{a∈A, a≠y} cos(y, a)
"""

import numpy as np


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns value in [-1, 1]."""
    u = np.asarray(a, dtype=np.float64)
    v = np.asarray(b, dtype=np.float64)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    sim = np.dot(u, v) / (norm_u * norm_v)
    return float(np.clip(sim, -1.0, 1.0))


def marginal_diversity_centroid(embedding: list[float], centroid: list[float]) -> float:
    """Marginal diversity as distance from the prompt centroid.

    Args:
        embedding: Embedding vector for response i.
        centroid:  Mean embedding vector over all responses for the prompt.

    Returns:
        1 - cosine_similarity(embedding, centroid).  Range [0, 2].
        0 = same direction as centroid, 1 = orthogonal, 2 = opposite.
    """
    return 1.0 - _cosine_similarity(embedding, centroid)


def marginal_diversity_maxsim(
    embedding: list[float],
    all_embeddings: list[list[float]],
    index: int,
) -> float:
    """Marginal diversity as distance from the most similar other response.

    Args:
        embedding:      Embedding vector for response i.
        all_embeddings: All response embeddings for the prompt (including i).
        index:          Position of this response in all_embeddings (to skip it).

    Returns:
        1 - max_{j≠i} cosine_similarity(e_i, e_j).  Range [0, 2].
        0 = a near-duplicate exists, 2 = completely dissimilar to all others.
        Returns 1.0 (neutral) if there are no other responses to compare against.
    """
    other_sims = [
        _cosine_similarity(embedding, other)
        for j, other in enumerate(all_embeddings)
        if j != index
    ]
    if not other_sims:
        return 1.0
    return 1.0 - max(other_sims)


def marginal_diversity_set_coverage(
    all_embeddings: list[list[float]],
    index: int,
) -> float:
    """Set-level marginal coverage: U(S) - U(S \\ {y_index}).

    Uses self-excluded utility U(A) = Σ_{y∈S} max_{a∈A, a≠y} cos(y, a).
    For each response y_k whose nearest-non-self neighbor is y_index, adds the gap
    between cos(y_k, y_index) and y_k's second-nearest-non-self similarity.

    Higher = contributes more unique coverage to the pool.
    0 = fully redundant (never the unique nearest neighbor of any other response).

    Args:
        all_embeddings: All response embeddings for the prompt (including index).
        index:          Position of this response in all_embeddings.

    Returns:
        Raw marginal coverage drop (≥ 0). Not bounded to [0, 2].
    """
    n = len(all_embeddings)
    if n <= 1:
        return 0.0

    # L2-normalize and compute pairwise cosine matrix (n x n)
    X = np.array(all_embeddings, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0.0, 1.0, norms)
    X = X / norms
    C = X @ X.T  # shape (n, n)

    # Self-exclusion: set diagonal to -inf so self never wins the max
    np.fill_diagonal(C, -np.inf)

    drop = 0.0
    for k in range(n):
        if k == index:
            continue
        row = C[k]  # similarities from y_k to all others (self=-inf)
        nn1_idx = int(np.argmax(row))
        if nn1_idx != index:
            continue  # y_index is not y_k's nearest neighbor; no contribution
        sim1 = float(row[nn1_idx])
        # Second-nearest: best similarity excluding both self (k) and y_index
        row_masked = row.copy()
        row_masked[index] = -np.inf
        best_remaining = float(np.max(row_masked))
        sim2 = best_remaining if best_remaining > -np.inf else 0.0
        drop += sim1 - sim2

    return drop


def score_marginal_diversity(
    method: str,
    embedding: list[float],
    centroid: list[float] | None = None,
    all_embeddings: list[list[float]] | None = None,
    index: int | None = None,
) -> float:
    """Compute marginal diversity score for a single response.

    Args:
        method:         "centroid", "maxsim", or "set_coverage".
        embedding:      Embedding vector for this response.
        centroid:       Required for method="centroid". Mean embedding of the prompt.
        all_embeddings: Required for method="maxsim"/"set_coverage". All response embeddings.
        index:          Required for method="maxsim"/"set_coverage". Index in all_embeddings.

    Returns:
        Float marginal diversity score. centroid/maxsim are in [0, 2];
        set_coverage is a raw drop (≥ 0, unbounded above).
    """
    if method == "centroid":
        if centroid is None:
            raise ValueError("centroid is required for method='centroid'")
        return marginal_diversity_centroid(embedding, centroid)
    elif method == "maxsim":
        if all_embeddings is None or index is None:
            raise ValueError("all_embeddings and index are required for method='maxsim'")
        return marginal_diversity_maxsim(embedding, all_embeddings, index)
    elif method == "set_coverage":
        if all_embeddings is None or index is None:
            raise ValueError("all_embeddings and index are required for method='set_coverage'")
        return marginal_diversity_set_coverage(all_embeddings, index)
    else:
        raise ValueError(f"Unknown diversity method: '{method}'. Use 'centroid', 'maxsim', or 'set_coverage'.")
