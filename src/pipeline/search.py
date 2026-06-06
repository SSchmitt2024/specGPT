"""
Phase 2 — Step 2.3: Search primitives

Three scoring primitives over the `spec_chunks` table populated by indexer.py.
Higher-level orchestration (RRF merge, reranking, structured lookup path)
belongs in `retriever.py` per PHASE2_BUILD_PLAN.md §2.3.

  1. vector_search(query, ...)    — semantic: Voyage query embedding + pgvector cosine
  2. tsvector_search(query, ...)  — keyword: Postgres tsvector + ts_rank_cd
                                    (stemmed via 'english' config; good for prose)
  3. bm25_search(query, ...)      — keyword: true Okapi BM25 via rank_bm25
                                    (literal tokens; good for spec identifiers
                                    like CDW10, FUSE, MPTR)

All three return a list of normalized result dicts:

  {
    "id":            str,
    "section_id":    str,
    "section_title": str,
    "content_type":  "prose" | "table",
    "text_raw":      str,
    "pdf_pages":     [int, ...],
    "figure_number": str | None,
    "has_normative": bool,
    "score":         float,                 # cosine sim / ts_rank_cd / bm25
    "method":        "vector" | "tsvector" | "bm25",
  }

Downstream stages add their own method tags to this shape:
  - retriever.structured_lookup  → method="structured_lookup"
  - retriever.rrf_merge          → method="rrf" + contributing_methods=[...]
  - reranker.rerank              → method="rerank" + prior_method=<prev>
  - reranker.rerank (empty path) → preserves prior method but still adds
                                   rerank_score=None / method="rerank"

Optional `filter` dict (applied inside both RPCs to scope candidates):

  {
    "content_type":   "prose" | "table",
    "section_prefix": "5.1",         # matches "5.1", "5.1.1", ...
    "has_normative":  True | False,
    "figure_number":  "312",
    "spec_version":   "2.1",
  }

Depends on Supabase RPCs `match_spec_chunks` and `search_spec_chunks_text`,
plus a generated `tsv` column on `spec_chunks`. See
scripts/supabase_schema.sql for the one-time setup. True BM25 has no
server-side dependency — see bm25_index.py.

CLI:
  python -m src.pipeline.search vector   "host memory buffer feature" --top-k 5
  python -m src.pipeline.search tsvector "Set Features"               --section-prefix 5
  python -m src.pipeline.search bm25     "CDW10 FUSE"                 --top-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

from src.pipeline.cache import ttl_cache

try:
    from supabase import Client, create_client
except ImportError:
    print("Missing dependency: pip install supabase")
    sys.exit(1)

try:
    import voyageai
except ImportError:
    print("Missing dependency: pip install voyageai")
    sys.exit(1)


logger = logging.getLogger(__name__)

# A query must contain at least one alphanumeric run after stripping
# punctuation/whitespace, otherwise embedding/tsquery wastes a network call.
_NON_EMPTY_RE = re.compile(r"[A-Za-z0-9]")


def _is_empty_query(query: str | None) -> bool:
    return not (query and _NON_EMPTY_RE.search(query))


def _retry(callable_, *, label: str, attempts: int = 3, base_delay: float = 0.5):
    """Call `callable_` with bounded retry/backoff on transient errors."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            return callable_()
        except Exception as e:  # noqa: BLE001
            last = e
            if i == attempts - 1:
                break
            sleep = base_delay * (2 ** i) + random.uniform(0, 0.25)
            logger.warning("%s failed (attempt %d/%d): %s — retrying in %.2fs",
                           label, i + 1, attempts, e, sleep)
            time.sleep(sleep)
    assert last is not None
    raise last


# ---------------------------------------------------------------------------
# Config

VOYAGE_MODEL = "voyage-3-lite"     # must match embedder.py
DEFAULT_TOP_K = 10

# RPC names (must match scripts/supabase_schema.sql)
RPC_VECTOR   = "match_spec_chunks"
RPC_TSVECTOR = "search_spec_chunks_text"


# ---------------------------------------------------------------------------
# Env / client init  (mirrors embedder.py + indexer.py loading pattern)

def _load_env_var(name: str) -> str | None:
    val = os.environ.get(name)
    if val:
        return val
    env_path = Path(".env")
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


@lru_cache(maxsize=1)
def supabase_client() -> Client:
    url = _load_env_var("SUPABASE_URL")
    key = _load_env_var("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be set (env or .env). "
            "Pipeline cannot run without Supabase credentials."
        )
    return create_client(url, key)


@lru_cache(maxsize=1)
def voyage_client() -> "voyageai.Client":
    key = _load_env_var("VOYAGE_API_KEY")
    if not key:
        raise RuntimeError(
            "VOYAGE_API_KEY must be set (env or .env). "
            "Pipeline cannot run without Voyage credentials."
        )
    return voyageai.Client(api_key=key)


# ---------------------------------------------------------------------------
# Filter + result shaping

# jsonb keys understood by the SQL RPCs.
_FILTER_KEYS = (
    "content_type", "section_prefix", "has_normative",
    "figure_number", "spec_version", "spec",
)


def _filter_to_jsonb(filt: dict | None) -> dict:
    """Coerce filter dict into the jsonb shape the RPCs expect (string values)."""
    if not filt:
        return {}
    out: dict = {}
    for k in _FILTER_KEYS:
        v = filt.get(k)
        if v is None:
            continue
        if isinstance(v, bool):
            out[k] = "true" if v else "false"
        else:
            out[k] = str(v)
    return out


def _shape(row: dict, score: float, method: str) -> dict:
    return {
        "id":            row.get("id"),
        "section_id":    row.get("section_id"),
        "section_title": row.get("section_title"),
        "content_type":  row.get("content_type"),
        "text_raw":      row.get("text_raw"),
        "pdf_pages":     row.get("pdf_pages") or [],
        "figure_number": row.get("figure_number"),
        "has_normative": bool(row.get("has_normative")),
        # Provenance: which spec this chunk came from (present once the RPCs
        # return it; None on older deployments). Lets the UI label citations.
        "spec":          row.get("spec"),
        "spec_document": row.get("spec_document"),
        "score":         float(score),
        "method":        method,
    }


# ---------------------------------------------------------------------------
# 1. Vector search

@ttl_cache(maxsize=1000, ttl=3600)
def vector_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filter: dict | None = None,
) -> list[dict]:
    """
    Embed the query (Voyage `input_type="query"`, asymmetric to the indexed
    documents) and return top_k chunks ordered by cosine similarity.

    Returns ``[]`` on transient embedding/RPC failure so the rest of the
    hybrid pipeline (tsvector + BM25) can still produce results.
    """
    if _is_empty_query(query):
        return []

    try:
        embed_resp = _retry(
            lambda: voyage_client().embed([query], model=VOYAGE_MODEL, input_type="query"),
            label="voyage.embed",
        )
        qvec = embed_resp.embeddings[0]

        resp = _retry(
            lambda: supabase_client().rpc(
                RPC_VECTOR,
                {
                    "query_embedding": qvec,
                    "match_count":     top_k,
                    "filter":          _filter_to_jsonb(filter),
                },
            ).execute(),
            label="supabase.rpc.match_spec_chunks",
        )
    except Exception as e:  # noqa: BLE001
        logger.error("vector_search failed: %s", e)
        return []

    rows = resp.data or []
    return [_shape(r, r.get("similarity", 0.0), "vector") for r in rows]


# ---------------------------------------------------------------------------
# 2. tsvector / full-text search
#
# Postgres' built-in text search via 'english' config + ts_rank_cd: applies
# Porter stemming and English stopword removal, then ranks by normalized
# cover density (TF-IDF-flavored). Strong for natural-language prose
# queries; weaker for exact-token queries like spec identifiers.

@ttl_cache(maxsize=1000, ttl=3600)
def tsvector_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filter: dict | None = None,
) -> list[dict]:
    """Full-text search via Postgres tsvector + websearch_to_tsquery."""
    if _is_empty_query(query):
        return []

    try:
        resp = _retry(
            lambda: supabase_client().rpc(
                RPC_TSVECTOR,
                {
                    "query_text":  query,
                    "match_count": top_k,
                    "filter":      _filter_to_jsonb(filter),
                },
            ).execute(),
            label="supabase.rpc.search_spec_chunks_text",
        )
    except Exception as e:  # noqa: BLE001
        logger.error("tsvector_search failed: %s", e)
        return []

    rows = resp.data or []
    return [_shape(r, r.get("rank", 0.0), "tsvector") for r in rows]


# ---------------------------------------------------------------------------
# 3. True Okapi BM25 (client-side, via rank_bm25)
#
# Managed Supabase doesn't permit installing ParadeDB / pg_search, so BM25
# runs in-process. The ~1,900-row corpus is loaded once per process and
# cached; see bm25_index.py for index + tokenization details.

@ttl_cache(maxsize=1000, ttl=3600)
def bm25_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filter: dict | None = None,
) -> list[dict]:
    """True Okapi BM25 over an in-memory index of spec_chunks."""
    if _is_empty_query(query):
        return []

    from src.pipeline import bm25_index  # local import: avoid load on cold paths

    try:
        hits = bm25_index.get_index().search(query, top_k=top_k, filter=filter)
    except Exception as e:  # noqa: BLE001
        logger.error("bm25_search failed: %s", e)
        return []

    return [_shape(row, score, "bm25") for row, score in hits]


# ---------------------------------------------------------------------------
# CLI

def _print_results(results: list[dict]) -> None:
    if not results:
        print("(no results)")
        return
    for i, r in enumerate(results, 1):
        head = f"[{i}] §{r['section_id']} {r['section_title'] or ''}".rstrip()
        meta = []
        if r["content_type"]:  meta.append(r["content_type"])
        if r["figure_number"]: meta.append(f"Fig {r['figure_number']}")
        if r["pdf_pages"]:     meta.append(f"p.{r['pdf_pages'][0]}")
        if r["has_normative"]: meta.append("normative")
        meta_str = " · ".join(meta)
        snippet = (r["text_raw"] or "").strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        print(f"\n{head}")
        print(f"    {meta_str}    score={r['score']:.3f}  ({r['method']})")
        print(f"    {snippet}")


def _build_filter(args) -> dict:
    filt = {
        "content_type":   args.content_type,
        "section_prefix": args.section_prefix,
        "has_normative":  True if args.has_normative else None,
        "figure_number":  args.figure_number,
        "spec_version":   args.spec_version,
    }
    return {k: v for k, v in filt.items() if v is not None}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="search",
        description="Search spec_chunks (vector | tsvector | bm25).",
    )
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    common.add_argument("--content-type", choices=["prose", "table"])
    common.add_argument("--section-prefix")
    common.add_argument("--has-normative", action="store_true")
    common.add_argument("--figure-number")
    common.add_argument("--spec-version")
    common.add_argument("--json", action="store_true", help="emit raw JSON")

    pv = sub.add_parser("vector", parents=[common], help="semantic (pgvector)")
    pv.add_argument("query")

    pt = sub.add_parser("tsvector", parents=[common], help="Postgres tsvector full-text")
    pt.add_argument("query")

    pb = sub.add_parser("bm25", parents=[common], help="true Okapi BM25 (in-memory)")
    pb.add_argument("query")

    args = p.parse_args(argv)
    filt = _build_filter(args)

    if args.mode == "vector":
        results = vector_search(args.query, top_k=args.top_k, filter=filt or None)
    elif args.mode == "tsvector":
        results = tsvector_search(args.query, top_k=args.top_k, filter=filt or None)
    else:
        results = bm25_search(args.query, top_k=args.top_k, filter=filt or None)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        _print_results(results)


if __name__ == "__main__":
    main()
