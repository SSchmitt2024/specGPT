"""
Phase 2 - Step 2.3b: Cross-encoder reranker

Precision filter that runs after RRF merge. Takes the ~20 candidates returned
by `retriever.rrf_merge()` and reorders them by reading each (query, chunk)
pair through a cross-encoder, then keeps the top 5-7.

Why bi-encoder + cross-encoder, not just one of them:
  - vector_search / bm25_search are bi-encoders: they encode query and doc
    separately, so they're cheap (precomputed doc embeddings) but lossy.
  - A cross-encoder reads the (query, doc) pair jointly and can attend
    across both texts — much higher precision, but slow to run for every
    chunk in the corpus. The funnel pattern (cheap recall → precise rerank)
    is the standard fix.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (local, free, ~80MB).
Fine-tuned on MS-MARCO query/passage relevance pairs. First call downloads
the weights and caches them under ~/.cache/huggingface.

CLI:
  python -m src.pipeline.reranker "What is the size of CDW10?" results.json
  cat candidates.json | python -m src.pipeline.reranker "..." - --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    print("Missing dependency: pip install sentence-transformers")
    sys.exit(1)


DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K = 7


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank(
    query: str,
    results: list[dict],
    *,
    top_k: int | None = DEFAULT_TOP_K,
    model_name: str = DEFAULT_MODEL,
    text_field: str = "text_raw",
) -> list[dict]:
    """
    Rerank candidate chunks by cross-encoder relevance to `query`.

    Args:
        query: the user query (the original; not a decomposed sub-query).
            Reranking against the original gives the model the full intent.
        results: candidate dicts from `rrf_merge()` or the search primitives.
            Each must carry chunk text under `text_field`.
        top_k: how many to keep after sorting. None = keep all, just reorder.
        model_name: HuggingFace cross-encoder model id.
        text_field: which key on each result holds the chunk text.

    Returns:
        Reordered list (length min(len(results), top_k)). Each result gains:
          - `rerank_score`: float (cross-encoder logit; higher = more relevant)
          - `method`: "rerank"
          - `prior_method`: the method before reranking (provenance)
    """
    if not results or not query.strip():
        return list(results)

    pairs = [(query, str(r.get(text_field) or "")) for r in results]
    model = _load_model(model_name)
    scores = model.predict(pairs)

    reranked: list[dict] = []
    for r, score in zip(results, scores):
        out = dict(r)
        out["rerank_score"] = float(score)
        out["prior_method"] = r.get("method")
        out["method"] = "rerank"
        reranked.append(out)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    if top_k is not None:
        reranked = reranked[:top_k]
    return reranked


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Cross-encoder reranker.")
    parser.add_argument("query", help="user query (original, not decomposed)")
    parser.add_argument(
        "results_json",
        help="path to JSON file with a list of result dicts, or '-' for stdin",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text-field", default="text_raw")
    args = parser.parse_args(argv)

    if args.results_json == "-":
        results = json.load(sys.stdin)
    else:
        with open(Path(args.results_json), encoding="utf-8") as f:
            results = json.load(f)

    out = rerank(
        args.query,
        results,
        top_k=args.top_k,
        model_name=args.model,
        text_field=args.text_field,
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
