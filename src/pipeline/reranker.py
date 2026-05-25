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
import logging
import sys
from functools import lru_cache
from pathlib import Path

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    print("Missing dependency: pip install sentence-transformers")
    sys.exit(1)


logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K = 7

# Score floor for chunks with empty/missing text — the cross-encoder will
# happily score the (query, "") pair, and the resulting score can outrank
# legitimate chunks. We mark them with -inf so they sort last while still
# being included in the trace.
_EMPTY_TEXT_SCORE = float("-inf")


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def _ensure_shape(item: dict, *, rerank_score: float | None, prior_method: str | None) -> dict:
    """Always-attached output shape so downstream consumers see consistent keys."""
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
              or None if reranking was skipped / failed.
          - `method`: "rerank"
          - `prior_method`: the method before reranking (provenance)
    """
    if not results:
        return []

    # Empty query → preserve original order but keep the output shape stable
    # so downstream code can index `rerank_score` / `prior_method` blindly.
    if not query or not query.strip():
        return [
            _ensure_shape(r, rerank_score=None, prior_method=r.get("method"))
            for r in (results[:top_k] if top_k is not None else results)
        ]

    # Empty text_field values get an explicit -inf score so they don't
    # outrank legitimate hits via cross-encoder noise on empty pairs.
    pairs: list[tuple[str, str]] = []
    is_empty: list[bool] = []
    for r in results:
        text = str(r.get(text_field) or "").strip()
        pairs.append((query, text))
        is_empty.append(not text)

    try:
        model = _load_model(model_name)
        raw_scores = model.predict(pairs)
    except Exception:  # OOM, CUDA OOM, model load fail, etc.
        # logger.exception captures the traceback — we want that for any
        # cross-encoder failure since the failure modes (CUDA OOM, model
        # download, etc.) are notoriously hard to diagnose without it.
        logger.exception("reranker.predict failed — preserving prior order")
        return [
            _ensure_shape(r, rerank_score=None, prior_method=r.get("method"))
            for r in (results[:top_k] if top_k is not None else results)
        ]

    reranked: list[dict] = []
    for r, raw_score, empty in zip(results, raw_scores, is_empty):
        score = _EMPTY_TEXT_SCORE if empty else float(raw_score)
        reranked.append(_ensure_shape(r, rerank_score=score, prior_method=r.get("method")))

    # All scores are floats here (empty/missing text gets -inf, model
    # successes get the cross-encoder logit); a direct sort is safe.
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
