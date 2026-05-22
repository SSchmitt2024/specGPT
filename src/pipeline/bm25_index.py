"""
True Okapi BM25 index over spec_chunks (via the `rank_bm25` library).

This is the third retrieval candidate alongside `vector_search` (pgvector
cosine over Voyage embeddings) and `tsvector_search` (Postgres tsvector +
ts_rank_cd). Managed Supabase doesn't permit installing ParadeDB /
pg_search, so true BM25 is implemented client-side.

Complementarity to tsvector_search:
  tsvector uses Postgres' 'english' config — Porter stemming + English
  stopwords. This index uses literal lowercased alphanumeric tokens with
  a tiny stopword list, so exact identifier matches ("CDW10", "FUSE",
  "MPTR", "MQES") rank higher here than under tsvector's stemmed form.
  Different match characteristics → diverse ranks → RRF benefits.

The corpus (~1,900 rows) is fetched from Supabase once per process and
held in memory. `lru_cache` means rebuild only on process restart.
"""

from __future__ import annotations

import re
from functools import lru_cache

try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    raise ImportError("Missing dependency: pip install rank_bm25") from e

from src.pipeline.search import supabase_client

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Intentionally small — over-filtering would strip terms that matter in
# spec text (e.g. "read", "write", "all", "any"). Includes only true
# noise words.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "to", "was", "were", "will", "with",
})

# Repeat the section title N times in the doc so title hits weight higher,
# mirroring the 'A' weight on title vs 'B' weight on body in the
# tsvector setweight() expression.
_TITLE_BOOST = 2

# Columns needed to build the index and shape downstream results.
_CORPUS_COLS = (
    "id, section_id, section_title, content_type, text_raw, "
    "pdf_pages, figure_number, has_normative, spec_version"
)


def tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return [t for t in (m.lower() for m in _TOKEN_RE.findall(text))
            if t not in _STOPWORDS]


def _doc_tokens(row: dict) -> list[str]:
    title_toks = tokenize(row.get("section_title"))
    body_toks  = tokenize(row.get("text_raw"))
    return title_toks * _TITLE_BOOST + body_toks


def _matches_filter(row: dict, filt: dict) -> bool:
    ct = filt.get("content_type")
    if ct and row.get("content_type") != ct:
        return False
    sp = filt.get("section_prefix")
    if sp and not (row.get("section_id") or "").startswith(sp):
        return False
    hn = filt.get("has_normative")
    if hn is not None:
        want = hn is True or str(hn).lower() == "true"
        if bool(row.get("has_normative")) != want:
            return False
    fn = filt.get("figure_number")
    if fn and row.get("figure_number") != fn:
        return False
    sv = filt.get("spec_version")
    if sv and row.get("spec_version") != sv:
        return False
    return True


def _fetch_corpus() -> list[dict]:
    """Page through spec_chunks; Supabase caps single requests at 1000 rows."""
    client = supabase_client()
    rows: list[dict] = []
    page_size = 1000
    start = 0
    while True:
        resp = (
            client.table("spec_chunks")
            .select(_CORPUS_COLS)
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return rows


class BM25Index:
    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        self._tokens = [_doc_tokens(r) for r in corpus]
        self.bm25 = BM25Okapi(self._tokens)

    def search(
        self,
        query: str,
        top_k: int,
        filter: dict | None = None,
    ) -> list[tuple[dict, float]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        if filter:
            candidates = [i for i in range(len(self.corpus))
                          if _matches_filter(self.corpus[i], filter)]
        else:
            candidates = list(range(len(self.corpus)))
        candidates.sort(key=lambda i: scores[i], reverse=True)
        out: list[tuple[dict, float]] = []
        for i in candidates:
            # Exactly 0 means no query token overlapped this doc — skip.
            # Negative scores can happen for terms appearing in >50% of
            # docs under rank_bm25's Robertson-Spärck-Jones IDF; keep
            # those, they still rank above non-overlapping docs.
            if scores[i] == 0:
                continue
            out.append((self.corpus[i], float(scores[i])))
            if len(out) >= top_k:
                break
        return out


@lru_cache(maxsize=1)
def get_index() -> BM25Index:
    return BM25Index(_fetch_corpus())
