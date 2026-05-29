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

import logging
import re
from functools import lru_cache

try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    raise ImportError("Missing dependency: pip install rank_bm25") from e

from src.pipeline.search import supabase_client

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Aligned with Postgres' `english` text-search config used by the tsvector
# path so the two keyword retrievers don't disagree on what counts as a
# stopword. Kept conservative — over-filtering strips terms that matter
# in spec text (e.g. "read", "write", "all", "any").
# Reference: src/postgres english_stop.sample (Snowball stop list).
_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
    "yourself", "yourselves",
})

# Additive title-match bonus. Each query token that appears in a doc's title
# adds this amount to the doc's BM25 score. We apply it at score time
# instead of repeating title tokens in the indexed body — repetition
# inflates |d| and skews avgdl across the whole corpus, partially
# cancelling the boost and distorting BM25 length normalization.
_TITLE_BOOST_PER_MATCH = 1.0

# Treat any score with magnitude below this as "no overlap" — avoids brittle
# float-equality checks on BM25 scores that can be negative for very common
# terms but should sort above true non-overlapping documents.
_SCORE_EPS = 1e-9

# Columns needed to build the index and shape downstream results.
_CORPUS_COLS = (
    "id, section_id, section_title, content_type, text_raw, "
    "pdf_pages, figure_number, has_normative, spec_version, spec, spec_document"
)


def tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return [t for t in (m.lower() for m in _TOKEN_RE.findall(text))
            if t not in _STOPWORDS]


def _doc_tokens(row: dict) -> list[str]:
    """Body-only tokens for BM25 indexing; title bonus is applied at score time."""
    return tokenize(row.get("text_raw"))


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
    sp_spec = filt.get("spec")
    if sp_spec and row.get("spec") != sp_spec:
        return False
    return True


def _fetch_corpus() -> list[dict]:
    """
    Page through spec_chunks with stable ordering.

    Robust against server-side max-rows caps (PostgREST `db-max-rows` can
    silently truncate a single response below the requested page size). We
    always advance `start` by the actual rows returned, terminate on the
    first empty batch, and rely on `.order("id")` for deterministic page
    boundaries that don't shift if rows are inserted between calls.
    """
    client = supabase_client()
    rows: list[dict] = []
    page_size = 1000
    start = 0
    while True:
        resp = (
            client.table("spec_chunks")
            .select(_CORPUS_COLS)
            .order("id")
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = resp.data or []
        if not batch:
            break
        rows.extend(batch)
        start += len(batch)
        # Hard cap to avoid pathological loops if the table grows unbounded
        # or the server returns the same page repeatedly.
        if start > 10_000_000:
            logger.warning("bm25_index pagination exceeded 10M rows; stopping")
            break
    return rows


class BM25Index:
    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        self._tokens = [_doc_tokens(r) for r in corpus]
        self._title_token_sets: list[frozenset[str]] = [
            frozenset(tokenize(r.get("section_title"))) for r in corpus
        ]
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

        # Additive per-doc title bonus — independent of doc length so it
        # doesn't interact with BM25's length normalization.
        q_unique = set(q_tokens)
        if _TITLE_BOOST_PER_MATCH:
            for i, title_set in enumerate(self._title_token_sets):
                matches = len(q_unique & title_set)
                if matches:
                    scores[i] += matches * _TITLE_BOOST_PER_MATCH

        if filter:
            candidates = [i for i in range(len(self.corpus))
                          if _matches_filter(self.corpus[i], filter)]
        else:
            candidates = list(range(len(self.corpus)))
        candidates.sort(key=lambda i: scores[i], reverse=True)

        out: list[tuple[dict, float]] = []
        for i in candidates:
            # Magnitude below epsilon means no body OR title overlap; skip.
            # Negative scores (very common query terms under rank_bm25's
            # Robertson-Spärck-Jones IDF) still beat the non-overlap floor.
            if abs(scores[i]) <= _SCORE_EPS:
                continue
            out.append((self.corpus[i], float(scores[i])))
            if len(out) >= top_k:
                break
        return out


# Cache the index for one process lifetime, but expose `reload_index` for
# explicit reindex without a restart (e.g. after a re-embed run or migration).
@lru_cache(maxsize=1)
def get_index() -> BM25Index:
    return BM25Index(_fetch_corpus())


def reload_index() -> BM25Index:
    """Drop the cached index and rebuild it from Supabase. Returns the new index."""
    get_index.cache_clear()
    return get_index()
