# Using PipelineConfig for A/B Testing

All tunable high-impact parameters are now exposed via `PipelineConfig`. The GUI can pass different values to experiment with answer quality.

## Basic Usage

```python
from src.pipeline.orchestrator import orchestrate, PipelineConfig

# Use defaults (current best settings)
result = orchestrate("What is bit 7 of CDW10?")

# Or customize for testing
config = PipelineConfig(
    vector_topk=15,          # More vector candidates
    bm25_topk=15,            # More BM25 candidates
    rrf_k=45,                # Sharper RRF ranking
    rrf_output_topk=25,      # Larger candidate pool
    final_rerank_topk=10,    # More final results
    max_subqueries=4,        # More decomposition
)
result = orchestrate("What is bit 7 of CDW10?", config=config)

# Response includes the config used
answer = result["answer"]
used_config = result["config"]  # {'vector_topk': 15, ...}
```

## Grid Search Example (for batch experiments)

```python
from src.pipeline.orchestrator import orchestrate, PipelineConfig
import json

# Test different top-K combinations
results = []
for vec_k in [8, 10, 15]:
    for bm25_k in [8, 10, 15]:
        for final_k in [5, 7, 10]:
            config = PipelineConfig(
                vector_topk=vec_k,
                bm25_topk=bm25_k,
                final_rerank_topk=final_k,
            )
            result = orchestrate(query, config=config)
            results.append({
                "config": config.to_dict(),
                "answer_length": len(result["answer"]),
                "latency_ms": result["pipeline_trace"][0]["took_ms"],
            })

# Save results for later analysis
with open("grid_search.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Web API Contract

The FastAPI endpoint can accept config as JSON:

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is bit 7 of CDW10?",
    "config": {
      "vector_topk": 15,
      "bm25_topk": 15,
      "rrf_k": 45,
      "rrf_output_topk": 25,
      "final_rerank_topk": 10,
      "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
      "max_subqueries": 4
    },
    "debug": true
  }'
```

Response:
```json
{
  "query": "What is bit 7 of CDW10?",
  "answer": "...",
  "config": {
    "vector_topk": 15,
    "bm25_topk": 15,
    ...
  },
  "pipeline_trace": [
    {
      "stage": "query_processor",
      "input": {"query": "..."},
      "output": {"type": "lookup", ...},
      "took_ms": 150
    },
    ...
  ]
}
```

## What Gets Logged in Pipeline Trace

Each stage logs the config values it used:

- **query_processor**: logs `max_subqueries` used
- **hybrid_search.vector_search_qX**: logs `vector_topk`
- **hybrid_search.bm25_search_qX**: logs `bm25_topk`
- **hybrid_search.rrf_merge**: logs `rrf_k` and `rrf_output_topk`
- **final_rerank**: logs `final_rerank_topk` and `cross_encoder_model`

This makes it easy to correlate config changes to answer quality changes.

## Default Config

If no config is passed, `PipelineConfig()` uses these defaults (current best):

```python
PipelineConfig(
    vector_topk=10,
    bm25_topk=10,
    rrf_k=60,
    rrf_output_topk=20,
    final_rerank_topk=7,
    cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_subqueries=3,
)
```

These match the Phase 2 baseline; tuning them is the next step for answer quality optimization.
