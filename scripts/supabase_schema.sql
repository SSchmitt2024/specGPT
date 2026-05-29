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

-- ── Multi-spec discriminator (Base vs PCIe Transport, etc.) ─────────────────
-- A single tagged corpus: every row carries which spec it came from, and the
-- retrievers always filter on it so Base and PCIe results never co-mingle.
-- DEFAULT 'base' backfills the existing single-spec rows automatically.
ALTER TABLE spec_chunks ADD COLUMN IF NOT EXISTS spec text NOT NULL DEFAULT 'base';
CREATE INDEX IF NOT EXISTS spec_chunks_spec_idx ON spec_chunks (spec);

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
-- NOTE: the multi-spec revision ADDS spec/spec_document to the RETURNS TABLE.
-- Postgres refuses to change a function's return type via CREATE OR REPLACE
-- (error 42P13), so the pre-multi-spec definition must be dropped first. The
-- arg signature is stable (vector, int, jsonb).
DROP FUNCTION IF EXISTS match_spec_chunks(vector, int, jsonb);
CREATE OR REPLACE FUNCTION match_spec_chunks(
  query_embedding vector(512),
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
  spec          text,
  spec_document text,
  similarity    float
)
LANGUAGE sql STABLE AS $$
  SELECT
    c.id, c.section_id, c.section_title, c.content_type,
    c.text_raw, c.pdf_pages, c.figure_number, c.has_normative,
    c.spec, c.spec_document,
    1 - (c.embedding <=> query_embedding) AS similarity
  FROM spec_chunks c
  WHERE
        (filter->>'content_type'   IS NULL OR c.content_type   = filter->>'content_type')
    AND (filter->>'section_prefix' IS NULL OR c.section_id     LIKE (filter->>'section_prefix') || '%')
    AND (filter->>'has_normative'  IS NULL OR c.has_normative   = (filter->>'has_normative')::boolean)
    AND (filter->>'figure_number'  IS NULL OR c.figure_number   = filter->>'figure_number')
    AND (filter->>'spec_version'   IS NULL OR c.spec_version    = filter->>'spec_version')
    AND (filter->>'spec'           IS NULL OR c.spec            = filter->>'spec')
  ORDER BY c.embedding <=> query_embedding
  LIMIT match_count;
$$;

-- ── RPC: tsvector full-text search ────────────────────────────────────────
-- Uses ts_rank_cd over the generated tsv column. Stems via 'english' config
-- (good for prose). True Okapi BM25 runs client-side; see bm25_index.py.
-- Dropped first for the same return-type reason as match_spec_chunks above.
DROP FUNCTION IF EXISTS search_spec_chunks_text(text, int, jsonb);
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
  spec          text,
  spec_document text,
  rank          float
)
LANGUAGE sql STABLE AS $$
  WITH q AS (SELECT websearch_to_tsquery('english', query_text) AS tsq)
  SELECT
    c.id, c.section_id, c.section_title, c.content_type,
    c.text_raw, c.pdf_pages, c.figure_number, c.has_normative,
    c.spec, c.spec_document,
    ts_rank_cd(c.tsv, q.tsq) AS rank
  FROM spec_chunks c, q
  WHERE c.tsv @@ q.tsq
    AND (filter->>'content_type'   IS NULL OR c.content_type   = filter->>'content_type')
    AND (filter->>'section_prefix' IS NULL OR c.section_id     LIKE (filter->>'section_prefix') || '%')
    AND (filter->>'has_normative'  IS NULL OR c.has_normative   = (filter->>'has_normative')::boolean)
    AND (filter->>'figure_number'  IS NULL OR c.figure_number   = filter->>'figure_number')
    AND (filter->>'spec_version'   IS NULL OR c.spec_version    = filter->>'spec_version')
    AND (filter->>'spec'           IS NULL OR c.spec            = filter->>'spec')
  ORDER BY rank DESC
  LIMIT match_count;
$$;

-- ── Lookup tables (loaded by scripts/load_lookup_data.py) ────────────────────
-- These replace local JSON file reads in retriever.py for deployed environments.

-- One row per unique field/register (from fields.json), per spec.
CREATE TABLE IF NOT EXISTS spec_fields (
    spec          TEXT NOT NULL DEFAULT 'base',
    name          TEXT NOT NULL,
    description   TEXT,
    "offset"      TEXT,
    figure_number TEXT,
    section_id    TEXT,
    data          JSONB NOT NULL,
    PRIMARY KEY (spec, name)
);

-- One row per (field_name, location) pair (from field_index.json).
-- field_index maps name → [records], so this is the flattened form.
CREATE TABLE IF NOT EXISTS spec_field_index (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    spec          TEXT NOT NULL DEFAULT 'base',
    field_name    TEXT NOT NULL,
    section_id    TEXT,
    figure_number TEXT,
    data          JSONB NOT NULL
);
-- NB: the (spec, field_name) index is created at the end of this file, AFTER the
-- migration ALTERs add `spec` to pre-existing tables — creating it here would
-- reference a column that doesn't exist yet on an already-deployed table.

-- One row per table (from tables.json), keyed by (spec, figure_number).
CREATE TABLE IF NOT EXISTS spec_tables (
    spec          TEXT NOT NULL DEFAULT 'base',
    figure_number TEXT NOT NULL,
    title         TEXT,
    section_id    TEXT,
    raw_text      TEXT,
    table_json    JSONB,
    data          JSONB NOT NULL
);
