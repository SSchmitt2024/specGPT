# Phase 2 — Build the Demo

**Goal:** Take Phase 1's parsed output and ship a live web app where you type a question and get a cited answer. Hybrid retrieval (vector + BM25) with query decomposition and table-aware retrieval paths + generation. Eval set proves it works.

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
│  2.2 SUPABASE INDEXING                          │
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
                      ▼ USER QUERY
                      │
┌─────────────────────────────────────────────────┐
│  2.2a QUERY CLASSIFIER + DECOMPOSER             │
│       (src/pipeline/query_processor.py)         │
│                                                 │
│  Step 1 — Classify query type (Haiku, ~$0.001): │
│    lookup     → "What are bits 7:4 of CDW10?"   │
│    structural → "How is the SQ organized?"      │
│    relational → "How do FID 0x01 and 0x12       │
│                  interact?"                     │
│    procedural → "How do I implement SGLs?"      │
│                                                 │
│  Step 2 — Extract entities:                     │
│    field names, figure numbers, hex values,     │
│    FIDs, section refs, CDW positions            │
│                                                 │
│  Step 3 — Decompose (relational/procedural):    │
│    Complex queries → 2-3 focused sub-queries    │
│    Lookup/structural → pass through as-is       │
│                                                 │
│  Output: {type, entities, sub_queries[]}        │
└──────┬──────────────────────────────────────────┘
       │
       ├─────────────────────────────────────┐
       ▼                                     ▼
  type == "lookup"                    all other types
  AND entities found                        │
       │                                    │
       ▼                                    ▼
┌──────────────────────┐     ┌──────────────────────────────────┐
│  2.3a STRUCTURED     │     │  2.3b HYBRID RETRIEVAL           │
│  LOOKUP PATH         │     │  (per sub-query, then merged)    │
│                      │     │                                  │
│  field_index.json    │     │  For each sub-query:             │
│  → match field name  │     │   BM25 SEARCH   VECTOR SEARCH    │
│  → fields.json       │     │   (exact:        (semantic:      │
│  → pull exact rows   │     │    hex, FIDs,     conceptual     │
│    from table_json   │     │    field names)   similarity)    │
│    in Supabase       │     │        │              │          │
│                      │     │        └──────┬───────┘          │
│  Bypasses embedding  │     │               ▼                  │
│  search entirely.    │     │    RRF merge sub-query results   │
│  Returns structured  │     │    → ~20 combined candidates     │
│  rows + headers,     │     │               ▼                  │
│  not serialized text │     │    CROSS-ENCODER RERANKING       │
│                      │     │    cross-encoder/ms-marco-MiniLM │
└──────────┬───────────┘     │    (query, chunk) pairs → top 5-7│
           │                 └──────────────┬───────────────────┘
           │                                │
           └──────────────┬─────────────────┘
                          ▼
┌─────────────────────────────────────────────────┐
│  2.4 CONTEXT ASSEMBLY + GENERATION              │
│                                                 │
│  Structured lookup: exact rows + headers        │
│  Hybrid retrieval: top 5-7 reranked chunks      │
│  Total context: 3-5k tokens max                 │
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
│           sources[], query_type}                │
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
| 2.2a | `src/pipeline/query_processor.py` | Query classification, entity extraction, sub-query decomposition | — | {type, entities, sub_queries} |
| 2.3a | `src/pipeline/retriever.py` (lookup path) | Structured field/register lookup via field_index.json + table_json | 2.1d, fields.json | exact table rows |
| 2.3b | `src/pipeline/retriever.py` (hybrid path) | BM25 + vector per sub-query + RRF merge + cross-encoder reranking | 2.1d | top 5-7 chunks |
| 2.4 | `src/pipeline/generator.py` | Context assembly + Sonnet generation | 2.3a or 2.3b | generation module |
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

### Query Classification + Decomposition
- Every query is classified before retrieval: `lookup | structural | relational | procedural`
- Classification is a single cheap Haiku call (~$0.001) that also extracts entities (field names, hex values, FIDs, figure numbers, section refs)
- Relational and procedural queries are decomposed into 2-3 focused sub-queries; each runs its own BM25+vector search; results merge before RRF
- Lookup and structural queries pass through as-is (no decomposition overhead)
- The `query_type` field is passed through to the response so the eval runner can score by type

### Table-Aware Retrieval (Structured Lookup Path)
- Lookup queries with extracted field/register entities skip embedding search entirely
- Path: entity name → `field_index.json` match → `fields.json` record → fetch corresponding `table_json` from Supabase by figure number → extract exact matching rows + column headers
- Delivers the precise bit/byte/hex value the user is asking about, not a serialized text approximation
- Falls back to hybrid retrieval if no entity match is found in `field_index.json`

### Retrieval Funnel (Hybrid Path)
1. Sub-query expansion → 1-3 queries (from decomposer)
2. BM25 + Vector search per sub-query → ~20 candidates each (broad recall)
3. Merge all sub-query candidate sets → deduplicate by chunk ID
4. RRF merge → ~20 combined candidates (rank fusion)
5. Cross-encoder rerank → top 5-7 (precision filter)
6. Context assembly → 3-5k tokens to Sonnet (budget control)

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
