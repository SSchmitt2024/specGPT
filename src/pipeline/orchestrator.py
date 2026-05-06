"""
Phase 2 - Orchestration Layer

Wires together query_processor → (structured_lookup AND hybrid_search) →
reranker → generator. Each stage emits structured tracing data for the
debug UI.

Designed to be called by the FastAPI app (app.py). Returns full pipeline
trace + final answer + citations, making it easy to wire a frontend that
visualizes every decision and result.

All high-impact tunable parameters are exposed as config:
  - vector_search.top_k
  - bm25_search.top_k
  - rrf_merge.k
  - rrf_output.top_k
  - reranker.top_k
  - query_processor.max_subqueries
  - reranker.model_name

    config = PipelineConfig(
        vector_topk=15,
        bm25_topk=15,
        rrf_k=45,
        final_rerank_topk=10,
    )
    result = orchestrate("What is bit 7:4 of CDW10?", config=config)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.pipeline import query_processor, retriever, search, reranker, generator
from src.pipeline.query_processor import QueryDecomposition


@dataclass
class PipelineConfig:
    """Configuration for all tunable high-impact parameters."""
    # Search parameters
    vector_topk: int = 10
    bm25_topk: int = 10

    # RRF merge parameters
    rrf_k: int = 60
    rrf_output_topk: int = 20

    # Reranking parameters
    final_rerank_topk: int = 7
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Query decomposition parameters
    max_subqueries: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PipelineStage:
    """A single execution stage with input, output, and timing."""
    stage: str
    input: dict
    output: dict
    took_ms: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _entity_list_to_dict(entities: list) -> list[dict]:
    """Convert Entity objects to dicts for JSON serialization."""
    return [
        {"text": e.text, "kind": e.kind}
        if hasattr(e, "text") and hasattr(e, "kind")
        else e
        for e in entities
    ]


def _result_summary(results: list[dict], limit: int = 5) -> list[dict]:
    """Summarize results for tracing (full text_raw for display, not in trace)."""
    return [
        {
            "id": r.get("id"),
            "section_id": r.get("section_id"),
            "section_title": r.get("section_title"),
            "content_type": r.get("content_type"),
            "method": r.get("method"),
            "score": r.get("score"),
            "rrf_score": r.get("rrf_score"),
            "rerank_score": r.get("rerank_score"),
        }
        for r in results[:limit]
    ]


def hybrid_search(
    query: str,
    sub_queries: list[str] | None = None,
    *,
    config: PipelineConfig | None = None,
) -> tuple[list[dict], list[PipelineStage]]:
    """
    Orchestrate hybrid retrieval: vector + BM25 per sub-query, then RRF merge.

    Args:
        query: original user query.
        sub_queries: decomposed queries. If None, use [query].
        config: PipelineConfig with tunable parameters. Defaults to PipelineConfig().

    Returns:
        (chunks, sub_trace) where chunks is RRF-merged results and sub_trace
        is list of PipelineStage dicts.
    """
    if config is None:
        config = PipelineConfig()
    if sub_queries is None:
        sub_queries = [query]

    sub_trace: list[PipelineStage] = []
    all_results: list[dict] = []

    # Step 1: Vector + BM25 per sub-query
    for i, sq in enumerate(sub_queries):
        start = time.time()
        vec_results = search.vector_search(sq, top_k=config.vector_topk)
        took_vec = time.time() - start

        start = time.time()
        bm25_results = search.bm25_search(sq, top_k=config.bm25_topk)
        took_bm25 = time.time() - start

        sub_trace.append(
            PipelineStage(
                stage=f"hybrid_search.vector_search_q{i}",
                input={"query": sq, "top_k": config.vector_topk},
                output={"results": _result_summary(vec_results, limit=3), "count": len(vec_results)},
                took_ms=took_vec * 1000,
            )
        )
        sub_trace.append(
            PipelineStage(
                stage=f"hybrid_search.bm25_search_q{i}",
                input={"query": sq, "top_k": config.bm25_topk},
                output={"results": _result_summary(bm25_results, limit=3), "count": len(bm25_results)},
                took_ms=took_bm25 * 1000,
            )
        )

        all_results.extend(vec_results)
        all_results.extend(bm25_results)

    # Step 2: RRF merge
    start = time.time()
    # Group by sub-query for per-query RRF, or just merge all together?
    # For simplicity, merge all. A more sophisticated approach would RRF per sub-query,
    # then rank sub-query result groups.
    # TODO: Consider per-sub-query RRF for better relevance isolation.
    merged = retriever.rrf_merge([all_results], k=config.rrf_k, top_k=config.rrf_output_topk)
    took_rrf = time.time() - start

    sub_trace.append(
        PipelineStage(
            stage="hybrid_search.rrf_merge",
            input={
                "result_lists": len([all_results]),
                "total_input": len(all_results),
                "k": config.rrf_k,
            },
            output={"results": _result_summary(merged, limit=3), "count": len(merged)},
            took_ms=took_rrf * 1000,
        )
    )

    # Step 3: Optional cross-encoder reranking
    final_results = merged
    if rerank:
        start = time.time()
        final_results = reranker.rerank(
            query,
            merged,
            top_k=None,  # keep all for now; will rerank merged pool again at orchestrator level
            text_field="text_raw",
        )
        took_rerank = time.time() - start

        sub_trace.append(
            PipelineStage(
                stage="hybrid_search.rerank",
                input={"count": len(merged)},
                output={"results": _result_summary(final_results, limit=3), "count": len(final_results)},
                took_ms=took_rerank * 1000,
            )
        )

    return final_results, sub_trace


def orchestrate(
    query: str,
    *,
    config: PipelineConfig | None = None,
    debug: bool = True,
) -> dict:
    """
    Execute the full retrieval + generation pipeline.

    Args:
        query: the user's question.
        config: PipelineConfig with tunable parameters. Defaults to PipelineConfig().
        debug: if True, include full pipeline_trace in response.

    Returns:
        {
            "answer": str,
            "citations": [{"text": ..., "source": ...}, ...],
            "sources": [chunk dicts],
            "config": config used,
            "pipeline_trace": [PipelineStage dicts] if debug else [],
        }
    """
    if config is None:
        config = PipelineConfig()

    trace: list[PipelineStage] = []

    # -------------------------------------------------------------------------
    # Stage 1: Query Processor (classify + decompose + extract entities)
    # -------------------------------------------------------------------------
    start = time.time()
    decomp: QueryDecomposition = query_processor.process_query(
        query,
        use_llm=True,
        max_subqueries=config.max_subqueries,
    )
    took_qp = time.time() - start

    trace.append(
        PipelineStage(
            stage="query_processor",
            input={"query": query},
            output={
                "type": decomp.type,
                "entities": _entity_list_to_dict(decomp.entities),
                "sub_queries": decomp.sub_queries,
                "notes": decomp.notes,
            },
            took_ms=took_qp * 1000,
        )
    )

    # -------------------------------------------------------------------------
    # Stage 2a: Structured Lookup (always attempt if lookup query)
    # -------------------------------------------------------------------------
    structured_chunks: list[dict] = []
    struct_found: bool = False

    if decomp.type == "lookup" and decomp.entities:
        start = time.time()
        struct_result = retriever.structured_lookup(
            decomp,
            use_llm=False,  # already did LLM in query_processor
            max_fields=8,
        )
        took_struct = time.time() - start

        struct_found = struct_result.found
        structured_chunks = struct_result.sources if struct_result.found else []

        trace.append(
            PipelineStage(
                stage="structured_lookup",
                input={
                    "type": decomp.type,
                    "entities": _entity_list_to_dict(decomp.entities),
                },
                output={
                    "found": struct_result.found,
                    "confidence": struct_result.confidence,
                    "field_count": len(struct_result.fields),
                    "table_count": len(struct_result.tables),
                    "sources": _result_summary(struct_result.sources),
                    "notes": struct_result.notes,
                },
                took_ms=took_struct * 1000,
            )
        )
    else:
        trace.append(
            PipelineStage(
                stage="structured_lookup",
                input={
                    "type": decomp.type,
                    "entities": _entity_list_to_dict(decomp.entities),
                },
                output={
                    "found": False,
                    "skipped": True,
                    "reason": "not a lookup query or no entities extracted",
                },
                took_ms=0.0,
            )
        )

    # -------------------------------------------------------------------------
    # Stage 2b: Hybrid Search (always run; vector + BM25 + RRF, no rerank yet)
    # -------------------------------------------------------------------------
    start = time.time()
    hybrid_chunks, hybrid_trace = hybrid_search(
        query,
        sub_queries=decomp.sub_queries if decomp.sub_queries else None,
        config=config,
    )
    took_hybrid = time.time() - start

    trace.extend(hybrid_trace)

    # -------------------------------------------------------------------------
    # Stage 2c: Merge results from both paths
    # -------------------------------------------------------------------------
    start = time.time()
    all_chunks = structured_chunks + hybrid_chunks

    # Deduplicate by id, keeping first occurrence (structured has priority)
    seen_ids: set = set()
    deduplicated: list[dict] = []
    for chunk in all_chunks:
        chunk_id = chunk.get("id")
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            deduplicated.append(chunk)

    took_dedup = time.time() - start

    trace.append(
        PipelineStage(
            stage="result_dedup",
            input={
                "structured_count": len(structured_chunks),
                "hybrid_count": len(hybrid_chunks),
            },
            output={
                "deduped_count": len(deduplicated),
                "sources": [
                    {
                        "id": c.get("id"),
                        "section_id": c.get("section_id"),
                        "method": c.get("method"),
                    }
                    for c in deduplicated[:10]
                ],
            },
            took_ms=took_dedup * 1000,
        )
    )

    # -------------------------------------------------------------------------
    # Stage 3: Rerank merged results (cross-encoder on combined pool)
    # -------------------------------------------------------------------------
    start = time.time()
    retrieved_chunks = reranker.rerank(
        query,
        deduplicated,
        top_k=config.final_rerank_topk,
        model_name=config.cross_encoder_model,
        text_field="text_raw",
    )
    took_rerank = time.time() - start

    trace.append(
        PipelineStage(
            stage="final_rerank",
            input={"chunk_count": len(deduplicated), "model": config.cross_encoder_model},
            output={
                "results": _result_summary(retrieved_chunks),
                "count": len(retrieved_chunks),
            },
            took_ms=took_rerank * 1000,
        )
    )

    # -------------------------------------------------------------------------
    # Stage 4: Context Assembly + Generation (Sonnet with strict system prompt)
    # -------------------------------------------------------------------------
    start = time.time()
    try:
        answer, citations, context_used, tokens_used = generator.generate(
            query,
            retrieved_chunks,
            model="claude-3-5-sonnet-20241022",
            max_context_tokens=4000,
        )
        took_gen = time.time() - start

        trace.append(
            PipelineStage(
                stage="generation",
                input={
                    "query": query,
                    "chunk_count": len(retrieved_chunks),
                },
                output={
                    "answer_length": len(answer),
                    "citation_count": len(citations),
                    "context_used": [
                        {
                            "section_id": c.get("section_id"),
                            "section_title": c.get("section_title"),
                            "content_type": c.get("content_type"),
                        }
                        for c in context_used
                    ],
                    "tokens": tokens_used,
                },
                took_ms=took_gen * 1000,
            )
        )
        context_chunks = context_used  # For response metadata
    except Exception as e:
        answer = f"Generation failed: {type(e).__name__}: {str(e)}"
        citations = []
        context_chunks = retrieved_chunks
        took_gen = time.time() - start
        tokens_used = {"prompt": 0, "completion": 0}

        trace.append(
            PipelineStage(
                stage="generation",
                input={"query": query, "chunk_count": len(retrieved_chunks)},
                output={"error": str(e)},
                took_ms=took_gen * 1000,
            )
        )

    trace.append(
        PipelineStage(
            stage="generation",
            input={
                "query": query,
                "context_length": len(context_text),
            },
            output={
                "answer_length": len(answer),
                "citation_count": len(citations),
            },
            took_ms=took_gen * 1000,
        )
    )

    # -------------------------------------------------------------------------
    # Assemble final response
    # -------------------------------------------------------------------------
    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "sources": context_chunks,
        "config": config.to_dict(),
        "pipeline_trace": [s.to_dict() for s in trace] if debug else [],
    }


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Run the full orchestration pipeline.")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--no-debug", action="store_true", help="Suppress pipeline trace in output")
    args = parser.parse_args()

    query = " ".join(args.query)
    result = orchestrate(query, debug=not args.no_debug)
    print(json.dumps(result, indent=2, ensure_ascii=False))
