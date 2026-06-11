# specGPT Architecture

A code map of how raw NVMe PDF bytes become a queryable, cited answer.

The pipeline runs in two phases:

- **Phase 1** turns the PDF into a structured corpus (sections, tables, fields, definitions) and adds a semantic layer (LLM-summarized cards + a relationship graph).
- **Phase 2** chunks that corpus, embeds it, indexes it in Supabase, and (in progress) serves it through a small FastAPI backend to a React frontend.

Every JSON artifact lives in `data/` (per-corpus: root = base NVMe, plus `data/command/` and `data/pcie/`). Phase-1 parsers and the runtime live in `src/`; Phase-2 build scripts live in `scripts/`. There is **no separate `frontend/` directory** — the UI is HTML embedded in `src/pipeline/app.py` (`FRONTEND_HTML`).

---

## Phase 1A — PDF → structural artifacts (deterministic)

```mermaid
flowchart LR
    PDF([nvme_spec/*.pdf])

    PDF --> TOC[toc_rebuild.py]
    PDF --> PR[prose.py]
    PDF --> TB[tables.py]
    PDF --> FD[fields.py]
    PDF --> DS[deep_sections.py]

    TOC --> tocj[(toc.json)]
    DS  --> tocj

    PR  --> prosej[(prose.json)]
    PR  --> defsj[(definitions.json)]

    TB  --> tabj[(tables.json)]

    tabj -. printed_page lookup .-> FD
    FD  --> fldj[(fields.json)]
    FD  --> fidxj[(field_index.json)]

    tocj -. section ranges .-> DS
    tocj -. page-range join .-> RL[relationships.py]
    tabj -. figure parent_section .-> RL
    RL  --> relj[(relationships.json)]

    classDef art fill:#0f1a2a,stroke:#5b8cc7,color:#cde,stroke-width:1px;
    classDef mod fill:#1e2a14,stroke:#a0a050,color:#fec,stroke-width:1px;
    classDef ext fill:#2a141e,stroke:#c75b8c,color:#fcd,stroke-width:1px;
    class tocj,prosej,defsj,tabj,fldj,fidxj,relj art;
    class TOC,PR,TB,FD,DS,RL mod;
    class PDF ext;
```

**What each module does:**

- `toc_rebuild.py` — pulls PyMuPDF's outline, normalizes section IDs, computes `printed_page` from `pdf_page - PAGE_OFFSET`. Initial pass.
- `prose.py` — walks every content page, recovers heading hierarchy from bold-line detection, slices body paragraphs by `[start, end)` page/y bounds. Side effect: extracts a `{term: definition}` map into `definitions.json`.
- `tables.py` — uses `page.find_tables()`, captures captions above each table, merges multi-page continuations, recursively flattens nested tables, keeps `raw_text` slice.
- `fields.py` — walks tables looking for bit/byte field tables, emits `(field, abbreviation, type, spec_page, ...)` records plus an abbreviation→record index.
- `deep_sections.py` — finds subsections too deep for the PDF outline (depth-4+), assigns each to a parent by page-range, **writes back into `toc.json`**.
- `relationships.py` — deterministic edges: figure→containing-section (page-range), section→child, plus regex-driven cross-references.

---

## Phase 1B — Structural artifacts → semantic layer (LLM)

```mermaid
flowchart TD
    tocj[(toc.json)]
    prosej[(prose.json)]
    tabj[(tables.json)]
    fldj[(fields.json)]
    relj[(relationships.json)]

    Gem([Gemini API])

    tocj   --> GC[llm/generate_cards.py]
    prosej --> GC
    tabj   --> GC
    relj   --> GC
    Gem   -. summary + keywords .-> GC
    GC    --> cardsj[(cards.json)]
    GC    -. checkpoint .-> cstate[(cards_state.json)]

    prosej --> ER[llm/extract_relationships.py]
    tocj   --> ER
    fldj   --> ER
    Gem   -. implicit edges .-> ER
    ER    --> rllmj[(relationships_llm.json)]
    ER    -. checkpoint .-> rstate[(relationships_llm_state.json)]

    relj   --> RC[llm/reconcile.py]
    rllmj  --> RC
    tocj   --> RC
    fldj   --> RC
    RC    --> rmrgj[(relationships_merged.json)]
    RC    --> entj[(entity_registry.json)]

    classDef art fill:#0f1a2a,stroke:#5b8cc7,color:#cde,stroke-width:1px;
    classDef mod fill:#1e2a14,stroke:#a0a050,color:#fec,stroke-width:1px;
    classDef ext fill:#2a141e,stroke:#c75b8c,color:#fcd,stroke-width:1px;
    class tocj,prosej,tabj,fldj,relj,cardsj,cstate,rllmj,rstate,rmrgj,entj art;
    class GC,ER,RC mod;
    class Gem ext;
```

**What each module does:**

- `llm/generate_cards.py` — one card per section. Most fields are deterministic (parent/child, tables, prose blocks, normative count). Only `summary` + `keywords` go to Gemini. Sections under `MIN_PROSE_CHARS` get a synthetic skeleton built from the title + child titles + table captions.
- `llm/extract_relationships.py` — reads each section's prose + a curated entity list, asks Gemini for implicit edges that the regex pass would miss (e.g. "Set Features uses Host Memory Buffer").
- `llm/reconcile.py` — normalizes entity names (strip articles, suffixes, parentheticals), snaps to canonical IDs (`section:<num>` validated against `toc.json`, `field:<abbrev>` validated against `fields.json`), merges deterministic + LLM edges, writes the canonical alias map.
- `llm/client.py` — shared Gemini client (`generate_json`) with retries.

---

## Phase 2A — Corpus → chunks → embeddings → index

```mermaid
flowchart LR
    cardsj[(cards.json)]
    prosej[(prose.json)]
    tabj[(tables.json)]

    cardsj --> CH[scripts/chunker.py]
    prosej --> CH
    CH    --> cpj[(chunks_prose.json)]

    cardsj --> TS[pipeline/table_serializer.py]
    tabj   --> TS
    TS    --> ctj[(chunks_tables.json)]

    Voy([Voyage AI<br/>voyage-3-lite, 1024d])

    cpj --> EM[scripts/embedder.py]
    ctj --> EM
    Voy -. embed batch .-> EM
    EM  --> cej[(chunks_embedded.json)]

    SB([Supabase<br/>spec_chunks<br/>pgvector + tsvector])
    cej --> IX[scripts/indexer.py]
    IX  --> SB

    classDef art fill:#0f1a2a,stroke:#5b8cc7,color:#cde,stroke-width:1px;
    classDef mod fill:#1e2a14,stroke:#a0a050,color:#fec,stroke-width:1px;
    classDef ext fill:#2a141e,stroke:#c75b8c,color:#fcd,stroke-width:1px;
    class cardsj,prosej,tabj,cpj,ctj,cej art;
    class CH,TS,EM,IX mod;
    class Voy,SB ext;
```

**What each module does:**

- `scripts/chunker.py` — flattens prose paragraphs into ~500-word overlapping chunks, prepends the section's card summary to every chunk so the embedding sees enough context, tracks which `pdf_pages` each chunk spans.
- `src/pipeline/table_serializer.py` — renders each table to plain text (`Figure N, caption`, headers, rows joined by `|`), prepends the card summary, one chunk per table. (Shared with the runtime.)
- `scripts/embedder.py` — batches all enriched chunks into Voyage AI (`voyage-3-lite`, 1024 dims, batch=128), enforces a per-run budget cap, writes vectors back onto each chunk record.
- `scripts/indexer.py` — upserts each chunk into Supabase `spec_chunks` (vector + raw text + metadata: section_id, content_type, pdf_pages, has_normative, figure_number, spec, etc.).

---

## Phase 2B — Runtime (frontend → backend → store)

```mermaid
flowchart LR
    U((User))

    subgraph App [src/pipeline/app.py — FastAPI, single file]
        FE[embedded HTML UI<br/>FRONTEND_HTML]
        AU[auth.py<br/>session-cookie gate]
        OR[orchestrator.py<br/>orchestrate]
    end

    DB([Supabase<br/>spec_chunks])
    LL([LLM provider<br/>Anthropic / OpenAI / Gemini])

    U  --> FE
    FE -. POST /api/query<br/>+ /api/query/stream SSE .-> OR
    AU -. gate .-> OR
    OR -. vector + BM25 + RRF<br/>+ cross-encoder rerank .-> DB
    OR -. generate w/ bracket-tag citations .-> LL
    OR --> FE

    classDef ui fill:#1e2a14,stroke:#a0a050,color:#fec,stroke-width:1px;
    classDef ext fill:#2a141e,stroke:#c75b8c,color:#fcd,stroke-width:1px;
    class FE,AU,OR ui;
    class DB,LL ext;
    class U ext;
```

**Status:** implemented and serving. `src/pipeline/app.py` is a single-file FastAPI app whose UI is HTML embedded in `FRONTEND_HTML` (no Vite/React app, no `frontend/` dir). It owns the full runtime: query decomposition, structured field/table lookup, hybrid retrieval (vector + BM25/tsvector), RRF merge, cross-encoder rerank, and LLM-generated answers with bracket-tag section citations. It is **multi-corpus** (NVMe base + Command Set + PCIe, with a `spec="all"` merge mode) and supports an optional **agentic** gap-analysis loop. The runtime reads from Supabase, not from `data/`. Run with `python -m src.pipeline.app`; see `RUNNING_THE_APP.md`. The actual stage wiring lives in `orchestrator.py::orchestrate`.

---

## Data artifact reference

| File | Producer | Consumers | What's inside |
|---|---|---|---|
| `toc.json` | `toc_rebuild.py` (+ `deep_sections.py`) | `relationships.py`, `generate_cards.py`, `extract_relationships.py`, `reconcile.py`, `deep_sections.py` | Section tree: `{section_number, title, level, target_page, pdf_page}`. Enriched in-place by `deep_sections`. |
| `prose.json` | `prose.py` | `generate_cards.py`, `extract_relationships.py`, `chunker.py` | One entry per section with paragraph list. Each paragraph carries text + page metadata. |
| `definitions.json` | `prose.py` | (future retrieval) | `{term: definition}` extracted from §1.6-style definition blocks. |
| `tables.json` | `tables.py` | `fields.py`, `relationships.py`, `generate_cards.py`, `table_serializer.py` | 717 figures with `caption`, `headers`, `rows`, `raw_text`, `printed_page`, `parent_section`. |
| `fields.json` | `fields.py` | `extract_relationships.py`, `reconcile.py` | 1,650 bit/byte field records w/ `spec_page`. |
| `field_index.json` | `fields.py` | (future retrieval, lookups) | Abbreviation → field record. |
| `relationships.json` | `relationships.py` | `generate_cards.py`, `reconcile.py` | Deterministic edges (regex + page-range). |
| `cards.json` | `llm/generate_cards.py` | `chunker.py`, `table_serializer.py` | 1,036 metadata cards w/ LLM `summary` + `keywords`. **Anchors every chunk's embedding context.** |
| `cards_state.json` | `llm/generate_cards.py` | itself (resume) | Checkpoint of which sections have been summarized. |
| `relationships_llm.json` | `llm/extract_relationships.py` | `reconcile.py` | Implicit edges Gemini surfaced. |
| `relationships_llm_state.json` | `llm/extract_relationships.py` | itself (resume) | Checkpoint. |
| `relationships_merged.json` | `llm/reconcile.py` | (future graph expansion in retrieval) | 7,706 unified, normalized edges. |
| `entity_registry.json` | `llm/reconcile.py` | (inspection / future canonicalization) | `{canonical: [aliases]}`. |
| `chunks_prose.json` | `pipeline/chunker.py` | `embedder.py` | ~1,188 prose chunks (~500 words, overlapping, summary-prefixed). |
| `chunks_tables.json` | `pipeline/table_serializer.py` | `embedder.py` | ~717 table chunks (serialized + summary-prefixed). |
| `chunks_embedded.json` | `pipeline/embedder.py` | `indexer.py` | Same chunks + `embedding: float[1024]`. |

---

## Notes / current state

- **Backend is live.** `src/pipeline/app.py` + `orchestrator.py` implement the full runtime (see Phase 2B). This doc's diagrams describe the data lineage; the authoritative runtime behavior is in the code, and `RUNNING_THE_APP.md` covers how to run it.
- **Multi-corpus.** Three specs are ingested (NVMe base, Command Set, PCIe), each with its own `data/<spec>/` artifact set and Supabase rows tagged by `spec`. `spec="all"` runs a per-corpus merge. Adding a corpus: see `docs/ADDING_A_SPEC.md` (and register it in `CONCRETE_SPEC_IDS` + `AVAILABLE_SPECS`).
- **Citations** deep-link the official PDF via `#page=N`; `pdf_pages` are 0-indexed, so the human page is `+1`. Citation format is bracket tags — the generation prompt in `generator.py` and the parser (`_extract_citations`) must stay in sync.
- **`relationships_merged.json`** (graph-expanded retrieval) and **`definitions.json`** remain available as enrichment artifacts; check the retrieval code for current usage rather than trusting this note.
