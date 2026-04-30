-- specGPT — Supabase schema augmentation + RPC functions for src/pipeline/search.py
--
-- Assumes spec_chunks already exists (created when you set up the project /
-- when indexer.py first upserted rows). This script only ADDS the parts
-- needed for the three search modes:
--
--   * generated tsvector column + GIN index → bm25_search()
--   * pgvector IVFFLAT index                 → vector_search()
--   * filter-helper btree indexes
--   * match_spec_chunks RPC                  → vector_search()
--   * search_spec_chunks_text RPC            → bm25_search()
--
-- Safe to re-run (everything is IF NOT EXISTS / OR REPLACE).
-- Run in the Supabase SQL editor or via psql.

-- ── Extensions ─────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Generated tsvector column for full-text search ─────────────────────────
-- Title is weighted higher than body so a query that matches the section
-- heading ranks ahead of one that only mentions the term in passing.
ALTER TABLE spec_chunks
  ADD COLUMN IF NOT EXISTS tsv tsvector
  GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(section_title, '')), 'A') ||
    setweight(to_tsvector('english', coalesce(text_raw, '')),       'B')
  ) STORED;

CREATE INDEX IF NOT EXISTS spec_chunks_tsv_idx
  ON spec_chunks USING gin (tsv);

-- ── pgvector ANN index (IVFFLAT, cosine distance) ─────────────────────────
-- `lists` ~= sqrt(rows) is a reasonable starting point. With ~1,900 rows,
-- 50-100 lists is fine. Re-tune if you grow the corpus an order of magnitude.
CREATE INDEX IF NOT EXISTS spec_chunks_embedding_idx
  ON spec_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ── Btree indexes that the filter clauses use ─────────────────────────────
CREATE INDEX IF NOT EXISTS spec_chunks_section_id_idx    ON spec_chunks (section_id);
CREATE INDEX IF NOT EXISTS spec_chunks_content_type_idx  ON spec_chunks (content_type);
CREATE INDEX IF NOT EXISTS spec_chunks_figure_number_idx ON spec_chunks (figure_number);

-- ── RPC: vector search ────────────────────────────────────────────────────
-- Returns the top match_count rows by cosine similarity to query_embedding,
-- after applying the filter object's optional constraints.
CREATE OR REPLACE FUNCTION match_spec_chunks(
  query_embedding vector(1024),
  match_count     int   DEFAULT 10,
  filter          jsonb DEFAULT '{}'::jsonb
) RETURNS TABLE (
  id            text,
  section_id    text,
  section_title text,
  content_type  text,
  text_raw      text,
  pdf_pages     int[],
  figure_number text,
  has_normative boolean,
  similarity    float
)
LANGUAGE sql STABLE AS $$
  SELECT
    c.id, c.section_id, c.section_title, c.content_type,
    c.text_raw, c.pdf_pages, c.figure_number, c.has_normative,
    1 - (c.embedding <=> query_embedding) AS similarity
  FROM spec_chunks c
  WHERE
        (filter->>'content_type'   IS NULL OR c.content_type   = filter->>'content_type')
    AND (filter->>'section_prefix' IS NULL OR c.section_id     LIKE (filter->>'section_prefix') || '%')
    AND (filter->>'has_normative'  IS NULL OR c.has_normative   = (filter->>'has_normative')::boolean)
    AND (filter->>'figure_number'  IS NULL OR c.figure_number   = filter->>'figure_number')
    AND (filter->>'spec_version'   IS NULL OR c.spec_version    = filter->>'spec_version')
  ORDER BY c.embedding <=> query_embedding
  LIMIT match_count;
$$;

-- ── RPC: BM25-style full-text search ──────────────────────────────────────
-- Uses ts_rank_cd over the generated tsv column. Not strict BM25 — swap to
-- paradedb / pg_search if you need true BM25 scoring later.
CREATE OR REPLACE FUNCTION search_spec_chunks_text(
  query_text  text,
  match_count int   DEFAULT 10,
  filter      jsonb DEFAULT '{}'::jsonb
) RETURNS TABLE (
  id            text,
  section_id    text,
  section_title text,
  content_type  text,
  text_raw      text,
  pdf_pages     int[],
  figure_number text,
  has_normative boolean,
  rank          float
)
LANGUAGE sql STABLE AS $$
  WITH q AS (SELECT websearch_to_tsquery('english', query_text) AS tsq)
  SELECT
    c.id, c.section_id, c.section_title, c.content_type,
    c.text_raw, c.pdf_pages, c.figure_number, c.has_normative,
    ts_rank_cd(c.tsv, q.tsq) AS rank
  FROM spec_chunks c, q
  WHERE c.tsv @@ q.tsq
    AND (filter->>'content_type'   IS NULL OR c.content_type   = filter->>'content_type')
    AND (filter->>'section_prefix' IS NULL OR c.section_id     LIKE (filter->>'section_prefix') || '%')
    AND (filter->>'has_normative'  IS NULL OR c.has_normative   = (filter->>'has_normative')::boolean)
    AND (filter->>'figure_number'  IS NULL OR c.figure_number   = filter->>'figure_number')
    AND (filter->>'spec_version'   IS NULL OR c.spec_version    = filter->>'spec_version')
  ORDER BY rank DESC
  LIMIT match_count;
$$;
