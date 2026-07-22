# Pipeline Debug UI

User-facing interface for the Phase 2 retrieval and generation pipeline. Shows every step, every choice, every result retrieved—designed for debugging answer quality and understanding why the system ranked chunks the way it did.

---

## Overview

The pipeline has 7 decision/processing stages. The UI visualizes all of them in sequence with collapsible drill-down for each stage.

```
User Query
    ↓
[1] Query Processor (classify + decompose)
    ├─ lookup query?
    │   ├─ [2a] Structured Lookup (field_index → fields → table rows)
    │   └─ found? → YES → [Jump to 6. Context Assembly]
    │                 → NO → [Continue to 2b]
    │
    └─ other type?
        └─ [2b] Hybrid Search (per sub-query)
            ├─ Vector Search (top 10)
            ├─ BM25 Search (top 10)
            ├─ RRF Merge (dedupe + rank fusion)
            └─ [3] Cross-Encoder Rerank (top 5-7)
    ↓
[4] Result Dedup (merge structured + hybrid results)
    ↓
[5] Context Assembly (3-5k tokens, trim large tables)
    ↓
[6] Generation (Sonnet with strict system prompt)
    ↓
[7] Response (answer + citations + sources)
```

---

## UI Layout

### Header
- **Query Input**: The original user query (read-only, shown for context)
- **Query Metadata**: Classification result (lookup/structural/relational/procedural), detected entities, decomposition count
- **Timeline**: mini timeline showing which paths were taken (structured vs hybrid)

### Main Panel: Collapsible Stages

#### Stage 1: Query Processor
**Title**: "Query Classification & Entity Extraction"

**Collapsed view**:
```
Type: lookup                          Entities: 3
"What is bit 7:4 of CDW10?"  [expand] Fields: [CDW10]
                                      Figures: [307]
```

**Expanded view**:
- **Raw LLM Output** (if used): Show the JSON from query_processor.py (type, entities, sub_queries)
- **Extracted Entities Table**:
  | Entity | Kind | Confidence |
  |--------|------|------------|
  | CDW10 | field | high |
  | bit 7:4 | range | auto-detected |
  | 307 | figure | auto-detected |
- **Decision Logic**: "→ Lookup query with field entity found. Routing to Structured Lookup."
- **Sub-queries** (if decomposed): List of 2-3 focused sub-queries with arrows showing how they'll be searched

#### Stage 2a: Structured Lookup (conditional)
**Title**: "Structured Lookup" (shown only if path taken)

**Collapsed view**:
```
Structured Lookup          Found: YES
"CDW10 bit 7:4" [expand]   Fields: 1 | Tables: 1
```

**Expanded view**:
- **Field Matches**:
  | Field Name | Parent Figure | Offset | Description |
  |------------|---------------|--------|-------------|
  | OACS | 307 | 7:4 | Command Abort Control Status |
- **Table Rows Retrieved**: Show the trimmed table rows (not raw text, but as a formatted table)
- **Confidence**: HIGH (exact match found)
- **Fallback Note**: If structured_lookup returned `found=False`, show "→ No entity match. Falling back to Hybrid Search."

#### Stage 2b: Hybrid Search (conditional)
**Title**: "Hybrid Retrieval" (shown only if path taken)

**Sub-stage: Per-Sub-Query Search** (if multiple sub-queries, show each in a tab or accordion)

##### Vector Search
**Collapsed**:
```
Vector Search  [expand]  Top 10 | Took 45ms
```

**Expanded**:
- **Embedding Model**: voyage-3-lite
- **Results Table**:
  | Rank | Section | Content Type | Score | Text Snippet |
  |------|---------|--------------|-------|------|
  | 1 | 5.2.1 | prose | 0.847 | "The host memory buffer stores..." |
  | 2 | Figure 312 | table | 0.823 | (table header row + 2 sample rows) |
  | ... | ... | ... | ... | ... |

##### BM25 Search
**Collapsed**:
```
BM25 Search  [expand]  Top 10 | Took 12ms
```

**Expanded**:
- Same table format as vector search, but `Score` is ts_rank_cd

##### RRF Merge
**Collapsed**:
```
RRF Merge  [expand]  Combined: 12 unique chunks | RRF(k=60)
```

**Expanded**:
- **Merge Logic Visualization**: Show a small Sankey/alluvial diagram:
  - Vector results (10) + BM25 results (10) → 12 unique chunks (8 in both, 2 only vector, 0 only BM25)
  - Hover on a chunk to see its position in each ranking
- **RRF Results Table**:
  | Rank | Section | RRF Score | Ranks | Contributing Methods |
  |------|---------|-----------|-------|----------------------|
  | 1 | 5.2.1 | 0.0456 | vec:1, bm25:2 | vector, bm25 |
  | 2 | Figure 312 | 0.0391 | vec:2, bm25:3 | vector, bm25 |
  | ... | ... | ... | ... | ... |
  - Show which chunks appear in both rankings (higher RRF scores)
  - Highlight the difference between single-method vs both-method chunks

#### Stage 3: Cross-Encoder Reranking
**Title**: "Cross-Encoder Rerank" (shown only if 2b was used)

**Collapsed**:
```
Rerank (ms-marco-MiniLM)  [expand]  Input: 12 | Output: 7 | Took 320ms
```

**Expanded**:
- **Rerank Scores Table**:
  | Rank | Section | Rerank Score | Prior Rank (RRF) | Score Δ |
  |------|---------|--------------|------------------|---------|
  | 1 | 5.2.1 | 8.234 | 1 | +0 |
  | 2 | Figure 312 | 7.891 | 2 | +0 |
  | 3 | 5.3.2 | 7.456 | 5 | ↑ moved up 2 |
  | ... | ... | ... | ... | ... |
- **Score Δ Column**: Shows if cross-encoder heavily reordered RRF's ranking (useful for detecting if RRF is missing the mark)
- **Visualization**: Before/After table or a swapping animation showing rank changes

#### Stage 4: Result Dedup
**Title**: "Final Candidate List" (always shown, even if only 1 path)

**Collapsed**:
```
Final Results  [expand]  7 chunks | Merged from: structured + rerank
```

**Expanded**:
- **Final Table**:
  | Rank | Section ID | Title | Content Type | Source | Score |
  |------|-----------|-------|--------------|--------|-------|
  | 1 | 5.2.1 | Host Memory Buffer | prose | rerank | 8.234 |
  | 2 | Figure 307 | CDW Definition | table | structured_lookup | HIGH (exact) |
  | ... | ... | ... | ... | ... | ... |
- **Merging Notes**: If both paths contributed (rare), explain how conflicts were resolved

#### Stage 5: Context Assembly
**Title**: "Context Prepared for Generation"

**Collapsed**:
```
Context Assembly  [expand]  3,847 / 5,000 tokens used
```

**Expanded**:
- **Token Budget Breakdown**:
  - System prompt: 342 tokens
  - Context (7 chunks): 3,087 tokens
  - Reserved for response: 1,571 tokens
- **Large Table Handling**: If any result was a large table, show the trimming:
  ```
  Figure 328 (Identify Controller) original: 16,400 words
  → Trimmed to: 42 rows (matched field names) + headers
  → Result: 890 tokens (displayed below)
  ```
- **Full Context Preview**: Collapsible block showing the exact text that will go to Sonnet

#### Stage 6: Generation
**Title**: "Generation (Claude Sonnet)"

**Collapsed**:
```
Generation  [expand]  Generated in 2.3s
```

**Expanded**:
- **System Prompt** (collapsible):
  ```
  You are an expert on NVMe specifications. Answer only using the provided
  context. Never guess. If context is insufficient, say so. Cite section
  numbers for every claim...
  ```
- **Generated Answer**:
  ```
  Bits 7:4 of the OACS field (CDW10) represent the Command Abort Control
  Status, as defined in Section 5.2.1 of the NVMe 2.1 specification.
  ...
  ```
- **Citations Extracted**:
  | Citation | Source |
  |----------|--------|
  | Section 5.2.1 | prose chunk from host memory buffer section |
  | CDW10 definition | Table 307 (Identify Controller) |

---

## Debugging Features

### Filtering & Search
- **Highlight Entity**: Type an entity name (e.g., "CDW10") → highlight all occurrences across all stages
- **Show Only Ranked Top-K**: Toggle to hide ranks > 7 for readability
- **Search Text Snippet**: Find a chunk that mentions specific text

### Annotations
- **Flag a Result**: "This chunk is wrong" → mark it for later review
- **Add Note**: Attach a note to a stage (e.g., "RRF ranked this too high because...")
- **Save Session**: Export the full pipeline trace as JSON for offline analysis

### Metrics
- **Latency Breakdown**: Bar chart of time spent in each stage
- **Score Analysis**: Histogram of RRF vs rerank scores; detect if reranker agrees with RRF or heavily reorders
- **Entity Match Rate**: Did we find all extracted entities? (Useful for spotting decomposition failures)

### Comparison Mode (future)
- Load two different answers for the same query
- Side-by-side pipeline comparison: see where the two diverged
- Useful for A/B testing ranking strategies, models, decomposition rules, etc.

---

## Data Model

All pipeline state is serializable as JSON. Each stage emits a result object:

```json
{
  "stage": "query_processor",
  "input": { "query": "..." },
  "output": {
    "type": "lookup",
    "entities": [...],
    "sub_queries": [...],
    "took_ms": 150
  }
}
```

The full trace is a list of these objects:
```json
[
  { "stage": "query_processor", ... },
  { "stage": "structured_lookup", ... },
  { "stage": "hybrid_search", ... },
  { "stage": "rerank", ... },
  { "stage": "context_assembly", ... },
  { "stage": "generation", ... }
]
```

Backend serializes this after each request. Frontend renders it. The JSON is also available for download (CSV export for tables, JSON for full trace).

---

## Implementation Notes

### Backend (FastAPI)
- Each pipeline module (query_processor, retriever, search, reranker, generator) emits a `PipelineStage` dataclass with `input`, `output`, `took_ms`, `metadata`
- The orchestrator collects these and returns them alongside the final answer
- New env var: `DEBUG_PIPELINE=1` to enable tracing (default off for performance)

### Frontend (React or plain HTML)
- Render stages as accordion/collapsible sections
- Use a table library for result ranking tables (sortable, filterable)
- Sankey diagram for RRF merge visualization (lightweight, client-side)
- Search/highlight via URL parameters: `?highlight=CDW10&show_stage=rerank`

### Mobile-Friendly
- Stack stages vertically
- Tables scroll horizontally
- Collapse by default (expanded on tap)

---

## Example: Debugging a Bad Answer

**User reports**: "I asked 'What is the CDW10 of a command?', and it gave me the wrong field definition."

**Steps using the UI**:
1. Enter query, run pipeline
2. Open "Query Processor" → see entities extracted correctly (CDW10 detected)
3. Open "Structured Lookup" → see exact table rows returned from Figure 307
4. **Aha**: The table rows showed a different field than expected. Check if the entity extraction got the field name wrong.
5. OR: Open "Hybrid Search" → see RRF merge rankings. Notice Figure 307 ranked 5th (low) despite being the *exact* table. Why?
   - Check Vector Search: maybe the embedding model didn't rank register definitions highly
   - Check BM25: maybe "CDW10" as exact keyword match isn't in the top 10
6. Open "Rerank" → see if cross-encoder rescued Figure 307, or if it stayed low
7. Open "Context Assembly" → see what text actually went to Sonnet. Is Figure 307 even in the context?
8. **Fix**: Adjust the RRF constant, or improve the structured lookup fallback, or boost table chunks in vector search.

---

## Future Enhancements

- **Explain Step**: For any stage, generate a textual explanation ("Why was this chunk ranked 3rd?")
- **Counterfactual**: "What if I had asked this differently?" → re-run with modified query
- **A/B Mode**: Run two ranking strategies in parallel, compare
- **Model Swap**: Switch cross-encoder model on the fly; see how it changes rankings
- **Latency Optimization**: Recommend which stage to optimize next
