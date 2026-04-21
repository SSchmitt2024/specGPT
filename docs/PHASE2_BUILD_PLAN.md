# Phase 2 — Build the Demo

**Goal:** Take Phase 1's parsed output and ship a live web app where you type a question and get a cited answer. Hybrid retrieval (vector + BM25) + generation. Eval set proves it works.

---

## Data Flow

```
PHASE 1 OUTPUTS (data/)
├── cards.json              1,036 metadata cards with summaries
├── prose.json              1,036 sections, 6,275 paragraphs
├── tables.json             717 structured tables with raw_text
├── relationships_merged    7,706 relationship edges
├── definitions.json        112 term/definition pairs
├── fields.json             1,650 bit/byte field records
├── field_index.json        1,108 field name lookups
└── entity_registry.json    352 canonical entities
        │
        ▼
┌─────────────────────────────────────────────────┐
│  2.1a CHUNKING ENGINE (src/pipeline/chunker.py) │
│                                                 │
│  prose.json paragraphs → merge into ~500-token  │
│  overlapping chunks → prepend card summary      │
│  → enriched prose chunks                        │
│                                                 │
│  Output: data/chunks_prose.json (1,188 chunks)  │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────┐
│  2.1b TABLE SERIALIZER (src/pipeline/tables.py) │
│                                                 │
│  tables.json → serialize each table into        │
│  readable text (headers + rows) → prepend card  │
│  summary → one chunk per table                  │
│                                                 │
│  Output: data/chunks_tables.json (~717 chunks)  │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  2.1c EMBEDDING PIPELINE                        │
│                                                 │
│  Model: Voyage AI (free tier) or local          │
│         nomic-embed-text                        │
│                                                 │
│  Input: all enriched chunks (prose + table)     │
│  Output: one vector per chunk (~768-1024 dim)   │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  2.2 SUPABASE INDEXING                         │
│                                                 │
│  Table: spec_chunks                             │
│  Columns:                                       │
│    id, embedding (vector), text, text_raw,      │
│    content_type (prose/table), section_id,      │
│    section_title, spec_version, spec_document,  │
│    pdf_pages, chunk_index, card_id,             │
│    has_normative, figure_number (tables only),  │
│    table_json (structured data, tables only)    │
│                                                 │
│  Indexes:                                       │
│    pgvector  → vector similarity search         │
│    tsvector  → BM25 full-text search            │
│    metadata  → filtered queries                 │
└─────────────────────┬───────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   BM25 SEARCH   VECTOR SEARCH   METADATA FILTER
   (exact:       (semantic:       (spec_version,
    hex, FIDs,    conceptual       section range,
    field names)  similarity)      content_type)
        │             │             │
        └──────┬──────┘             │
               ▼                    │
┌──────────────┴────────────────────┘
│  2.3 RECIPROCAL RANK FUSION (RRF)
│
│  score = Σ 1/(k + rank_i) for each result
│  Merges BM25 + vector ranked lists into one
│  ~20 candidates
└──────────────────┬────────────────
                   ▼
┌─────────────────────────────────────────────────┐
│  2.3b CROSS-ENCODER RERANKING                   │
│                                                 │
│  Model: cross-encoder/ms-marco-MiniLM (local)   │
│  Input: (query, chunk) pairs from RRF top ~20   │
│  Output: reranked, keep top 5-7                 │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│  2.4 CONTEXT ASSEMBLY + GENERATION              │
│                                                 │
│  Assemble top 5-7 chunks into context window    │
│  (3-5k tokens max)                              │
│                                                 │
│  Large tables: pull structured JSON, filter to  │
│  relevant rows + headers only                   │
│                                                 │
│  → Claude Sonnet with strict system prompt:     │
│    - Use ONLY provided context                  │
│    - Cite section numbers for every claim       │
│    - Include exact CDW/bit/byte/hex values      │
│    - State gaps, never guess                    │
│                                                 │
│  Output: {answer, citations[], confidence,      │
│           sources[]}                            │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│  2.5 WEB APPLICATION                            │
│                                                 │
│  Backend: FastAPI                               │
│    POST /api/query → full pipeline → response   │
│    Rate limiting, error handling                │
│                                                 │
│  Frontend: React or plain HTML                  │
│    Search bar                                   │
│    Answer display with inline citations         │
│    Collapsible source panel                     │
│    Mobile-friendly                              │
│                                                 │
│  Deploy: Docker → Railway / Fly.io / AWS        │
└─────────────────────────────────────────────────┘
```

---

## Build Order

| Step | File | What It Does | Depends On | Output |
|------|------|-------------|------------|--------|
| 2.1a | `src/pipeline/chunker.py` | Chunk prose into overlapping segments with card enrichment | cards.json, prose.json | chunks_prose.json |
| 2.1b | `src/pipeline/table_serializer.py` | Serialize tables into embeddable text chunks | tables.json, cards.json | chunks_tables.json |
| 2.1c | `src/pipeline/embedder.py` | Embed all chunks via Voyage AI or nomic | 2.1a + 2.1b | vectors |
| 2.1d | `src/pipeline/indexer.py` | Load embeddings + chunks into Supabase | 2.1c | indexed DB |
| 2.2 | `src/pipeline/eval_gen.py` | Auto-generate + curate QA eval set | cards.json | eval_set.json |
| 2.3 | `src/pipeline/retriever.py` | BM25 + vector search + RRF + reranking | 2.1d | retrieval module |
| 2.4 | `src/pipeline/generator.py` | Context assembly + Sonnet generation | 2.3 | generation module |
| 2.5 | `src/pipeline/app.py` | FastAPI backend + frontend | 2.4 | deployed app |
| 2.6 | `src/pipeline/eval_run.py` | Run eval set, score by question type | 2.2 + 2.5 | accuracy report |

---

## Key Design Decisions

### Chunking Strategy
- Prose: merge paragraphs into ~500 token chunks with 50-token overlap
- Tables: one table = one chunk, never split
- All chunks get card summary prepended ("definition-enriched")

### Large Table Handling
- 18 tables exceed 1,000 words; Figure 328 (Identify Controller) is 16,400 words
- Embedding will truncate these — acceptable because BM25 still finds keywords anywhere in the table
- At context assembly time, large tables get trimmed to relevant rows + headers
- Structured JSON preserved separately for precise row extraction

### Two-Copy Design
- `text` — enriched version (summary + body), used for embedding
- `text_raw` — original body only, used for display to users

### Retrieval Funnel
1. BM25 + Vector search → ~20 candidates each (broad recall)
2. RRF merge → ~20 combined candidates (rank fusion)
3. Cross-encoder rerank → top 5-7 (precision filter)
4. Context assembly → 3-5k tokens to Sonnet (budget control)

### Cost Profile
| Component | Cost |
|-----------|------|
| Embedding (Voyage AI free tier or local) | Free |
| Supabase (free tier) | Free |
| Cross-encoder reranker (local) | Free |
| BM25 + RRF (local) | Free |
| Claude Sonnet per query | ~1-3 cents |

---

## Possible Improvements (Post-Eval)

If eval scores reveal specific failure patterns, consider these additions:

- **Query expansion** — if failures are "user said X but the spec calls it Y" (vocabulary mismatch), add a Haiku call before search that rewrites the query into 2-3 spec-aligned variants. Cheap (~$0.001/query), easy to implement, directly addresses domain terminology gaps.
- **Domain-tuned embeddings** (BUILD_PLAN_FINAL.md 3.7) — if vector search consistently ranks wrong chunks, fine-tune the embedding model on NVMe-specific pairs generated from our relationship data.
- **Parent-child retrieval** — if small chunks lack context, retrieve the small chunk for precision but return its parent section for completeness.

Don't add these preemptively. Run the eval, see what breaks, then apply the fix that matches the failure mode.

---

## Files Completed

- [x] `src/pipeline/chunker.py` — prose chunking engine (1,188 chunks produced)
- [x] `src/pipeline/table_serializer.py` — table serializer (717 table chunks produced)
- [x] `src/pipeline/embedder.py` — Voyage AI embedding pipeline (1,905 chunks → voyage-3-lite 1024-dim vectors)
