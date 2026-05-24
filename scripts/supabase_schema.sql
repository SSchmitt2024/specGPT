-- specGPT — Supabase schema augmentation + RPC functions for src/pipeline/search.py
--
-- Assumes spec_chunks already exists (created when you set up the project /
-- when indexer.py first upserted rows). This script only ADDS the parts
-- needed for the server-backed retrievers:
--
--   * generated tsvector column + GIN index → tsvector_search()
--   * pgvector IVFFLAT index                 → vector_search()
--   * filter-helper btree indexes
--   * match_spec_chunks RPC                  → vector_search()
--   * search_spec_chunks_text RPC            → tsvector_search()
--
-- The third retriever, true Okapi BM25 (search.bm25_search), runs
-- client-side via rank_bm25 and needs no schema changes — see
-- src/pipeline/bm25_index.py.
--
-- Safe to re-run (everything is IF NOT EXISTS / OR REPLACE).
-- Run in the Supabase SQL editor or via psql.

-- ── Extensions ─────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Table chunk range columns (added for split-table support) ───────────────
ALTER TABLE spec_chunks ADD COLUMN IF NOT EXISTS row_start int;
ALTER TABLE spec_chunks ADD COLUMN IF NOT EXISTS row_end   int;

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

-- ── RPC: tsvector full-text search ────────────────────────────────────────
-- Uses ts_rank_cd over the generated tsv column. Stems via 'english' config
-- (good for prose). True Okapi BM25 runs client-side; see bm25_index.py.
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

-- ── Lookup tables (loaded by scripts/load_lookup_data.py) ────────────────────
-- These replace local JSON file reads in retriever.py for deployed environments.

-- One row per unique field/register (from fields.json).
CREATE TABLE IF NOT EXISTS spec_fields (
    name          TEXT PRIMARY KEY,
    description   TEXT,
    offset        TEXT,
    figure_number TEXT,
    section_id    TEXT,
    data          JSONB NOT NULL
);

-- One row per (field_name, location) pair (from field_index.json).
-- field_index maps name → [records], so this is the flattened form.
CREATE TABLE IF NOT EXISTS spec_field_index (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    field_name    TEXT NOT NULL,
    section_id    TEXT,
    figure_number TEXT,
    data          JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS spec_field_index_name_idx ON spec_field_index (field_name);

-- One row per table (from tables.json), keyed by figure_number.
CREATE TABLE IF NOT EXISTS spec_tables (
    figure_number TEXT PRIMARY KEY,
    title         TEXT,
    section_id    TEXT,
    raw_text      TEXT,
    table_json    JSONB,
    data          JSONB NOT NULL
);
