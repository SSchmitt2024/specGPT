# Phase 2 Tunable Parameters for A/B Testing

Extracted from all Phase 2 pipeline scripts. Parameters marked **HIGH IMPACT** should be prioritized for tuning; **LOW IMPACT** can be left as-is for initial eval.

---

## 1. Retrieval Funnel Parameters (HIGH IMPACT)

These control the candidate pool at each stage. Most critical for balancing recall vs. precision.

| Parameter | Current | File | Impact | Test Range | Notes |
|-----------|---------|------|--------|------------|-------|
| **Vector Search Top-K** | 10 | search.py | HIGH | 5, 8, 10, 15, 20 | Per-sub-query; affects RRF merge input size |
| **BM25 Search Top-K** | 10 | search.py | HIGH | 5, 8, 10, 15, 20 | Per-sub-query; balance keyword recall |
| **RRF Merge Output** | 20 | orchestrator.py | HIGH | 10, 15, 20, 30 | Pre-rerank candidate pool size |
| **Final Rerank Top-K** | 7 | reranker.py | HIGH | 5, 7, 10, 15 | Directly controls answer context size |
| **Structured Lookup Max Fields** | 8 | retriever.py | MEDIUM | 4, 6, 8, 12 | Prevents structured path from bloating output |

### Testing Strategy

Run a grid search on the most impactful pairs:
- (vector_topk, bm25_topk): 10×10 vs. 15×15 vs. 20×20 (higher = more recall)
- (rrf_output, final_rerank_topk): 20×7 vs. 30×10 (test recall-precision frontier)

---

## 2. RRF Fusion Configuration (MEDIUM IMPACT)

Controls how vector + BM25 results are merged.

| Parameter | Current | File | Impact | Test Range | Notes |
|-----------|---------|------|--------|------------|-------|
| **RRF k (constant)** | 60 | retriever.py | MEDIUM | 30, 45, 60, 90, 120 | Higher = softer ranking; lower = sharper lead |

### What It Does

RRF score = Σ 1/(k + rank). With k=60, a result at rank #1 gets 1/61 ≈ 0.0164. Lower k makes top results more dominant; higher k gives more weight to lower ranks.

**Test scenarios:**
- k=30: sharper lead; penalizes results outside top 5-10
- k=60: balanced (default; Cormack et al. recommendation)
- k=120: flatter curve; ranks 5-20 are nearly equivalent

---

## 3. Cross-Encoder Model Selection (HIGH IMPACT)

Affects precision of final ranking.

| Parameter | Current | File | Impact | Test Range | Notes |
|-----------|---------|------|--------|------------|-------|
| **Reranker Model** | cross-encoder/ms-marco-MiniLM-L-6-v2 | reranker.py | HIGH | See below | ~80MB local; latency ~320ms for 20 candidates |

### Model Alternatives to Test

- **cross-encoder/ms-marco-MiniLM-L-6-v2** (current): Fast, accurate on passage ranking
- **cross-encoder/qnli-distilroberta-base**: Optimized for NLI; may catch semantic mismatches
- **cross-encoder/ms-marco-TinyBERT-L-2-v2**: Smaller, faster; lower precision (if speed matters)

Load different model: `python -m src.pipeline.reranker "query" results.json --model <model_name>`

---

## 4. Query Decomposition (MEDIUM IMPACT)

Controls how complex queries are split into sub-queries.

| Parameter | Current | File | Impact | Test Range | Notes |
|-----------|---------|------|--------|------------|-------|
| **Max Sub-Queries** | 3 | query_processor.py (line 274) | MEDIUM | 2, 3, 4, 5 | Higher = better coverage; lower = less search volume |
| **LLM Max Tokens** | 400 | query_processor.py (line 291) | LOW | 300, 400, 500 | Budget for classification + decomposition |
| **LLM Temperature** | 0.0 | query_processor.py (line 290) | LOW | 0.0 (fixed) | Deterministic; don't change |

### Testing Strategy

For procedural/relational queries, test:
- max_subqueries=2: fast but misses nuance
- max_subqueries=3: balanced
- max_subqueries=4: comprehensive but 4× search volume

Monitor: total pipeline latency and search API cost.

---

## 5. Embedding & Indexing (LOW IMPACT for eval; HIGH for cost)

Affects initial indexing; less relevant for runtime tuning.

| Parameter | Current | File | Impact | Test Range | Notes |
|-----------|---------|------|--------|------------|-------|
| **Embedding Model** | voyage-3-lite | embedder.py, search.py | MEDIUM | voyage-3-lite vs. voyage-3 | voyage-3 is higher quality but 10× cost |
| **Batch Size** | 128 | embedder.py | LOW | 32, 64, 128 | Only matters during re-embedding; ~300 RPM limit on free tier |
| **Sleep Between Batches** | 0.5s | embedder.py | LOW | 0.2, 0.5, 1.0s | Rate limiting to avoid quota overrun |
| **Max Budget** | $0.50 | embedder.py | LOW | Tuning this is per-organization | Hard limit to prevent surprise charges |

### Notes

- Embedding quality is **fixed once indexed**; only relevant if re-indexing with a different model
- Current indexing used **voyage-3-lite** (free tier); changing model requires re-embedding all 1,900 chunks
- For eval, assume embeddings are fixed; focus on retrieval + ranking tuning

---

## 6. Context Assembly & Generation (NOT YET TUNABLE)

These are placeholders; generator.py not yet implemented.

| Parameter | Current | File | Impact | Status |
|-----------|---------|------|--------|--------|
| **Context Token Budget** | Hardcoded 3-5k | orchestrator.py (line 347) | HIGH | TODO: implement proper token counting & trimming |
| **Max Context Chunks** | 7 | orchestrator.py (lines 325, 351) | MEDIUM | Naive fixed limit; should be dynamic based on token budget |
| **Large Table Trimming** | Not implemented | orchestrator.py | MEDIUM | Currently sends full table text; should trim rows to relevant fields |

---

## 7. Structured Lookup (MEDIUM IMPACT for lookup queries)

Only applies to lookup-type queries (e.g., "What is bit 7 of CDW10?").

| Parameter | Current | File | Impact | Test Range | Notes |
|-----------|---------|------|--------|------------|-------|
| **Max Fields** | 8 | retriever.py | MEDIUM | 4, 6, 8, 12, 16 | Prevents bloated table excerpts |
| **Followup Rows After Field Heading** | 12 | retriever.py (line 259) | LOW | 8, 10, 12, 15 | How many rows to include after a field definition |
| **Confidence Threshold** | Rule-based | retriever.py (lines 304-309) | LOW | N/A (heuristic logic) | Returns HIGH/MEDIUM/LOW; not numeric threshold |

---

## Recommended A/B Testing Plan

### Phase 1: Baseline (Week 1)
Run eval on **current defaults**:
- vector_topk=10, bm25_topk=10
- rrf_output=20, final_rerank_topk=7
- rrf_k=60
- max_subqueries=3
- cross-encoder: ms-marco-MiniLM

Expected: Establish baseline accuracy (EM, F1) on eval set.

### Phase 2: Funnel Optimization (Week 2)
Grid search on top-K values:
```
for vector_topk in [8, 10, 15]:
  for bm25_topk in [8, 10, 15]:
    for rrf_topk in [15, 20, 25]:
      for final_topk in [5, 7, 10]:
        run_eval()
```

Analyze: accuracy vs. latency trade-off. Find the Pareto frontier.

### Phase 3: RRF & Reranker Tuning (Week 3)
- Test RRF k ∈ {30, 60, 90, 120}
- Test cross-encoder models (if different models improve precision)
- Measure: impact on ranking disagreement between RRF and cross-encoder

### Phase 4: Query Decomposition (Week 4)
- Test max_subqueries ∈ {2, 3, 4}
- Measure: per-query-type accuracy (lookup vs. structural vs. procedural)
- Optimize: max_subqueries by type

---

## How to Expose These in the Frontend

Each parameter should be:
1. **Configurable via API**: `POST /api/query` with optional `config` dict
2. **Loggable in pipeline trace**: Each stage emits the parameter value used
3. **Sweepable for batch experiments**: `POST /api/batch-eval` with grid of configs

Example request:
```json
{
  "query": "What is bit 7 of CDW10?",
  "config": {
    "vector_topk": 15,
    "bm25_topk": 15,
    "rrf_k": 45,
    "final_rerank_topk": 10,
    "max_subqueries": 3,
    "cross_encoder_model": "cross-encoder/qnli-distilroberta-base"
  }
}
```

Response includes: `answer`, `citations`, `pipeline_trace` (with config values logged at each stage), `metrics` (latency, cost, confidence).

---

## Cost-Benefit Summary

### High ROI Tuning (Worth Testing)
- ✅ Vector + BM25 top-K: massive impact on recall; cheap to change
- ✅ Final rerank top-K: directly affects answer quality
- ✅ RRF k constant: tunes merge aggressiveness; cheap to test
- ✅ Cross-encoder model: precision leverage point
- ✅ Max sub-queries: coverage vs. cost trade-off

### Low ROI Tuning (Not Worth Testing)
- ❌ Batch size, sleep timers: only matter during initial indexing
- ❌ LLM temperature: already deterministic (0.0)
- ❌ Structured lookup row limits: only affects <20% of queries
- ❌ Max token budget, chunk count: awaiting generator implementation

### Blocked on Implementation
- 🔒 Context assembly token budgeting (generator.py needed)
- 🔒 Per-subquery RRF isolation (orchestrator.py todo)
- 🔒 Large table row trimming (context assembly needed)

---

## Logging Strategy for Testing

Every pipeline execution should log:
```json
{
  "config": {
    "vector_topk": 10,
    "bm25_topk": 10,
    "rrf_k": 60,
    ...
  },
  "pipeline_trace": [
    {
      "stage": "hybrid_search.vector_search_q0",
      "config_used": {"top_k": 10},
      "output": {"results": [...], "count": 10}
    },
    ...
  ],
  "metrics": {
    "latency_ms": 1245,
    "api_calls": 6,
    "estimated_cost_cents": 0.15
  }
}
```

This allows post-hoc analysis: "For queries where RRF k=30 produced X answer, what was the cost vs. k=60?"
