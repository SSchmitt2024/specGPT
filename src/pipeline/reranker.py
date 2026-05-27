"""
Phase 2 - Step 2.3b: Cross-encoder reranker (Voyage AI API)

Precision filter that runs after RRF merge. Takes the ~20 candidates returned
by `retriever.rrf_merge()` and reorders them by relevance, keeping the top 5-7.

Uses the Voyage AI rerank-2-lite API so the server carries zero PyTorch
weight at runtime. Requires VOYAGE_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from functools import lru_cache

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "rerank-2-lite"
DEFAULT_TOP_K = 7

_EMPTY_TEXT_SCORE = float("-inf")


@lru_cache(maxsize=1)
def _voyage_client():
    try:
        import voyageai
    except ImportError:
        raise RuntimeError("Missing dependency: pip install voyageai")
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        raise RuntimeError("VOYAGE_API_KEY must be set for the reranker")
    return voyageai.Client(api_key=key)


def _ensure_shape(item: dict, *, rerank_score: float | None, prior_method: str | None) -> dict:
    out = dict(item)
    out["rerank_score"] = rerank_score
    out["prior_method"] = prior_method
    out["method"] = "rerank"
    return out


def rerank(
    query: str,
    results: list[dict],
    *,
    top_k: int | None = DEFAULT_TOP_K,
    model_name: str = DEFAULT_MODEL,
    text_field: str = "text_raw",
) -> list[dict]:
    """
    Rerank candidate chunks by relevance to `query` using the Voyage rerank API.

    Returns reordered list (length min(len(results), top_k)). Each result gains:
      - `rerank_score`: float or None if reranking failed
      - `method`: "rerank"
      - `prior_method`: the method before reranking
    """
    if not results:
        return []

    if not query or not query.strip():
        return [
            _ensure_shape(r, rerank_score=None, prior_method=r.get("method"))
            for r in (results[:top_k] if top_k is not None else results)
        ]

    raw_docs = [str(r.get(text_field) or "").strip() for r in results]
    is_empty = [not d for d in raw_docs]
    # Voyage rejects empty strings; substitute a placeholder so the indices
    # stay aligned, then override with -inf below.
    documents = [d if d else " " for d in raw_docs]

    try:
        client = _voyage_client()
        resp = client.rerank(query, documents, model=model_name, top_k=len(documents))
        score_map = {item.index: item.relevance_score for item in resp.results}
    except Exception:
        logger.exception("reranker.rerank failed; preserving prior order")
        return [
            _ensure_shape(r, rerank_score=None, prior_method=r.get("method"))
            for r in (results[:top_k] if top_k is not None else results)
        ]

    reranked = []
    for idx, (r, empty) in enumerate(zip(results, is_empty)):
        if empty or idx not in score_map:
            score = _EMPTY_TEXT_SCORE
        else:
            score = float(score_map[idx])
        reranked.append(_ensure_shape(r, rerank_score=score, prior_method=r.get("method")))

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    if top_k is not None:
        reranked = reranked[:top_k]
    return reranked


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Voyage AI reranker.")
    parser.add_argument("query")
    parser.add_argument("results_json", help="path to JSON file or '-' for stdin")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text-field", default="text_raw")
    args = parser.parse_args(argv)

    if args.results_json == "-":
        results = json.load(sys.stdin)
    else:
        from pathlib import Path
        with open(Path(args.results_json), encoding="utf-8") as f:
            results = json.load(f)

    out = rerank(args.query, results, top_k=args.top_k, model_name=args.model, text_field=args.text_field)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
