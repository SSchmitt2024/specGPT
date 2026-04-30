# specGPT — Data Model Reference

What we extract from the NVMe spec PDF and why each piece exists.

---

## Data Files (all in `data/`)

### 1. `toc.json` — Section Index
**Source:** `parser.py` + `deep_sections.py`
**What it is:** Flat list of every section heading in the spec, with hierarchy.

```json
{
  "section_number": "5.17",
  "title": "Format NVM Command",
  "level": 2,
  "target_page": 320
}
```

**Why:** This is the skeleton. The parsing scripts (`prose.py`, `relationships.py`, `deep_sections.py`, `fields.py`, `generate_cards.py`, etc.) all use `section_number` to assign content to the right section. Without the TOC, we can't slice the 800-page PDF into addressable units or build the section hierarchy that cards, prose, and graph edges depend on.

---

### 2. `tables.json` — Structured Table Data
**Source:** `tables.py`
**What it is:** Every "Figure N:" table in the spec, parsed into headers + rows.

```json
{
  "figure_number": 328,
  "caption": "Identify – Identify Controller Data Structure",
  "printed_page": 322,
  "headers": ["Bytes", "I/O", "Admin", "Disc", "Description"],
  "rows": [["76", "O", "O", "R", "Controller Multi-Path ..."], ...],K
  
  "raw_text": "<fallback plain text>"
}
```

**Why:** NVMe answers live in tables — register layouts, command fields, data structures. If you can't parse the tables, you can't answer most questions. The `raw_text` field is a fallback for embedding/search when structured parsing misses something.

---

### 3. `fields.json` + `field_index.json` — Named Field Registry
**Source:** `fields.py`
**What it is:** Every named field extracted from data-structure, register, and command tables.

```json
{
  "field_name": "HMPRE",
  "full_name": "Host Memory Buffer Preferred Size",
  "parent_figure": 328,
  "parent_type": "data_structure",
  "offset": "275:272",
  "offset_type": "bytes",
  "requirements": {"I/O": "O", "Admin": "O"},
  "values": {"00h": "No HMB support"},
  "cross_refs": [{"type": "section", "id": "5.1.2"}],
  "description": "Indicates the preferred size..."
}
```

`field_index.json` is just `{field_name: [field records]}` for fast lookup.

**Why:** When someone asks "what is HMPRE?", this is the direct answer. Fields are the atomic unit of NVMe knowledge. The `values` dict captures enumeration tables inline. `cross_refs` are extracted from the description text (e.g., "refer to section 5.1.2").

---

### 4. `prose.json` — Section Prose Text
**Source:** `prose.py`
**What it is:** All non-table text per section, split into paragraphs, with normative language tagged.


```json
{
  "section_number": "5.17",
  "title": "Format NVM Command",
  "paragraphs": [
    {"text": "The Format NVM command ...", "pdf_page": 320}
  ],
  "normative": [
    {"strength": "shall", "text": "The host shall set CDW10...", "pdf_page": 321}
  ]
}
```

**Why:** Tables tell you the structure, prose tells you the rules. The `normative` tags (shall/should/may) are critical — they're the actual requirements engineers need to follow. These get embedded for vector search.

---

### 5. `definitions.json` — Term Glossary
**Source:** `prose.py` (from section 1.5.x)
**What it is:** `{term: definition}` lookup.

**Why:** Prepended to chunks before embedding so the embedding model understands NVMe jargon. Also used for direct definition lookups.

---

### 6. `relationships.json` — Structural Edges (Deterministic)
**Source:** `relationships.py`
**What it is:** Links between sections and figures, extracted by regex.

```json
{
  "source": "figure:328",
  "target": "section:5.17",
  "type": "contained_in",
  "evidence": "printed_page 322 falls inside section 5.17",
  "confidence": "deterministic"
}
```

**Edge types:**
| Type | Meaning | Example |
|------|---------|---------|
| `contained_in` | Figure X lives inside Section Y | Figure 328 is in Section 5.17 |
| `child_of` | Section X.Y is a sub-section of X | 5.17.1 -> 5.17 |
| `cross_reference` | Text in X mentions Y | Section 5.17 says "see Section 3.2" |

Cross-references also have a `strength` field:
- `strong`: preceded by "see", "refer to", "as defined in" — an intentional pointer
- `mention`: bare "Section X.Y" occurrence — could be incidental

---

### 7. `relationships_llm.json` — Semantic Edges (LLM-Extracted)
**Source:** `llm/extract_relationships.py`
**What it is:** Implicit relationships the regex missed, extracted by sending prose to an LLM.

```json
{
  "source": "command:Set Features",
  "target": "feature:Host Memory Buffer",
  "type": "configured_by",
  "evidence": "[5.1.2] the Set Features command configures the Host Memory Buffer",
  "confidence": "llm"
}
```

**Edge types:** `uses`, `returned_by`, `posts_to`, `requires`, `defined_in`, `configured_by`, `superseded_by`, `related_to`

---

### 8. `cards.json` — Section Metadata Cards
**Source:** `llm/generate_cards.py`
**What it is:** One card per section, combining everything above into a single record.

```json
{
  "section_id": "5.17",
  "title": "Format NVM Command",
  "spec_document": "NVM Express Base Specification",
  "spec_version": "2.1",
  "summary": "LLM-generated 2-4 sentence summary",
  "keywords": ["Format NVM", "LBAF", "secure erase"],
  "parent_section": "5",
  "child_sections": ["5.17.1", "5.17.2"],
  "tables": [415, 416],
  "prose_blocks": [0, 1, 2],
  "relationships": [...],
  "normative_count": 12
}
```

**Why:** The card is the retrieval unit. When a chunk matches a query, the card's summary gets prepended to enrich the embedding context. Keywords help BM25 search. The card also connects everything: which tables, which children, how many normative statements.

---

## How the Edges Get Used 

The edges are **not** a general-purpose knowledge graph for browsing. They serve one purpose: **retrieval expansion**.

Here's the query pipeline:

```
User question
    ↓
[BM25 search] → exact term matches (field names, hex values, LIDs)
[Vector search] → semantic similarity on embedded chunks
    ↓
Reciprocal Rank Fusion → merged ranked list
    ↓
★ Graph expansion ★ → look up top chunks' graph nodes,
                        walk 1-2 hops, pull in neighbors
    ↓
Cross-encoder reranking → final top 5-7 chunks
    ↓
LLM generation with citations
```

The **graph expansion** step is where edges matter. Example:

> **Question:** "How do I enable Host Memory Buffer?"
>
> Vector search finds Section 5.1.2 (HMB overview).
> Graph expansion walks edges and also pulls in:
> - Figure 313 (CDW11 bit layout) via `contained_in`
> - Section 5.1.1 (Identify Controller fields HMPRE/HMMIN) via `cross_reference`
> - Set Features command via `configured_by` (LLM edge)
>
> Now the LLM has all the pieces to give a complete procedural answer.

Without edges, you'd only get whatever vector similarity happened to surface. With edges, structurally related content that might not share the same words still gets pulled into context.

### What makes this different from Obsidian

| | Obsidian | specGPT edges |
|---|---|---|
| **Purpose** | Human browsing/navigation | Machine retrieval expansion |
| **Created by** | User manually links notes | Auto-extracted from PDF structure + regex + LLM |
| **Granularity** | Note-to-note | Section/figure/field/command-level |
| **Consumed by** | Human clicking links | NetworkX graph traversal in the retrieval pipeline |
| **Editing** | Users add/remove links freely | Static per spec version, regenerated on re-parse |

The edges are an internal retrieval optimization, not a user-facing feature. Users never see or interact with them — they just get better answers because the retrieval pipeline can follow structural connections.

---

## Data Flow Summary

```
NVMe PDF
  ├── parser.py ────────→ toc.json (section index)
  ├── tables.py ────────→ tables.json (structured tables)
  ├── fields.py ────────→ fields.json + field_index.json (named fields)
  ├── prose.py ─────────→ prose.json + definitions.json (text + normative tags)
  ├── deep_sections.py ─→ toc.json (enriched with depth 4+ sections)
  ├── relationships.py ─→ relationships.json (structural edges)
  ├── llm/extract_relationships.py → relationships_llm.json (semantic edges)
  └── llm/generate_cards.py ───────→ cards.json (per-section metadata cards)
```

Phase 2 consumes all of this to build the NetworkX graph + Supabase vector index.
Phase 3 queries both during retrieval.
