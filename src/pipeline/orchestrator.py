"""
Phase 2 - Orchestration Layer

Wires together query_processor → (structured_lookup OR hybrid_search) →
reranker → generator. Each stage emits structured tracing data for the
debug UI.

Designed to be called by the FastAPI app (app.py). Returns full pipeline
trace + final answer + citations, making it easy to wire a frontend that
visualizes every decision and result.

    result = orchestrate("What is bit 7:4 of CDW10?", debug=True)
    answer = result["answer"]
    trace = result["pipeline_trace"]  # list of PipelineStage dicts
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.pipeline import query_processor, retriever, search, reranker
from src.pipeline.query_processor import QueryDecomposition


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
    top_k_per_method: int = 10,
    top_k_final: int = 7,
) -> tuple[list[dict], list[PipelineStage]]:
    """
    Orchestrate hybrid retrieval for non-lookup queries.

    Runs BM25 + vector search per sub-query, merges via RRF, reranks via
    cross-encoder. Returns final chunks + trace of sub-stages.

    Args:
        query: original user query (for reranking context).
        sub_queries: decomposed queries. If None, use [query].
        top_k_per_method: top-K candidates per search method per sub-query.
        top_k_final: final top-K after reranking.

    Returns:
        (final_chunks, sub_trace) where final_chunks is the top-K reranked
        results and sub_trace is a list of PipelineStage dicts for each
        sub-step (vector, bm25, rrf, rerank).
    """
    if sub_queries is None:
        sub_queries = [query]

    sub_trace: list[PipelineStage] = []
    all_results: list[dict] = []

    # Step 1: Vector + BM25 per sub-query
    for i, sq in enumerate(sub_queries):
        start = time.time()
        vec_results = search.vector_search(sq, top_k=top_k_per_method)
        took_vec = time.time() - start

        start = time.time()
        bm25_results = search.bm25_search(sq, top_k=top_k_per_method)
        took_bm25 = time.time() - start

        sub_trace.append(
            PipelineStage(
                stage=f"hybrid_search.vector_search_q{i}",
                input={"query": sq, "top_k": top_k_per_method},
                output={"results": _result_summary(vec_results, limit=3), "count": len(vec_results)},
                took_ms=took_vec * 1000,
            )
        )
        sub_trace.append(
            PipelineStage(
                stage=f"hybrid_search.bm25_search_q{i}",
                input={"query": sq, "top_k": top_k_per_method},
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
    merged = retriever.rrf_merge([all_results], top_k=top_k_per_method * 2)
    took_rrf = time.time() - start

    sub_trace.append(
        PipelineStage(
            stage="hybrid_search.rrf_merge",
            input={
                "result_lists": len([all_results]),
                "total_input": len(all_results),
                "k": 60,
            },
            output={"results": _result_summary(merged, limit=3), "count": len(merged)},
            took_ms=took_rrf * 1000,
        )
    )

    # Step 3: Cross-encoder reranking
    start = time.time()
    reranked = reranker.rerank(
        query,
        merged,
        top_k=top_k_final,
        text_field="text_raw",
    )
    took_rerank = time.time() - start

    sub_trace.append(
        PipelineStage(
            stage="hybrid_search.rerank",
            input={"count": len(merged), "top_k": top_k_final},
            output={"results": _result_summary(reranked, limit=3), "count": len(reranked)},
            took_ms=took_rerank * 1000,
            metadata={
                "score_shifts": [
                    {
                        "id": r.get("id"),
                        "prior_rank": i,
                        "rerank_score": r.get("rerank_score"),
                    }
                    for i, r in enumerate(reranked[:5])
                ],
            },
        )
    )

    return reranked, sub_trace


def orchestrate(query: str, *, debug: bool = True) -> dict:
    """
    Execute the full retrieval + generation pipeline.

    Args:
        query: the user's question.
        debug: if True, include full pipeline_trace in response.

    Returns:
        {
            "answer": str,
            "citations": [{"text": ..., "source": ...}, ...],
            "sources": [chunk dicts],
            "pipeline_trace": [PipelineStage dicts] if debug else [],
        }
    """
    trace: list[PipelineStage] = []

    # -------------------------------------------------------------------------
    # Stage 1: Query Processor (classify + decompose + extract entities)
    # -------------------------------------------------------------------------
    start = time.time()
    decomp: QueryDecomposition = query_processor.process_query(query, use_llm=True)
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
    # Stage 2a/2b: Routing decision (structured vs hybrid)
    # -------------------------------------------------------------------------
    retrieved_chunks: list[dict] = []
    used_structured: bool = False

    # Try structured lookup if this is a lookup query with extracted entities
    if decomp.type == "lookup" and decomp.entities:
        start = time.time()
        struct_result = retriever.structured_lookup(
            decomp,
            use_llm=False,  # already did LLM in query_processor
            max_fields=8,
        )
        took_struct = time.time() - start

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

        if struct_result.found:
            retrieved_chunks = struct_result.sources
            used_structured = True
        # If not found, fall through to hybrid

    # If structured lookup was skipped or didn't find anything, try hybrid
    if not used_structured:
        start = time.time()
        retrieved_chunks, hybrid_trace = hybrid_search(
            query,
            sub_queries=decomp.sub_queries if decomp.sub_queries else None,
            top_k_per_method=10,
            top_k_final=7,
        )
        took_hybrid = time.time() - start

        trace.extend(hybrid_trace)
        trace.append(
            PipelineStage(
                stage="hybrid_search_complete",
                input={
                    "method": "hybrid",
                    "sub_queries": decomp.sub_queries or [query],
                },
                output={"results": _result_summary(retrieved_chunks), "count": len(retrieved_chunks)},
                took_ms=took_hybrid * 1000,
            )
        )

    # -------------------------------------------------------------------------
    # Stage 3: Context Assembly (TODO: implement full token budgeting)
    # -------------------------------------------------------------------------
    start = time.time()
    # TODO: Implement proper context assembly with:
    #   - Token budget enforcement (3-5k tokens)
    #   - Large table trimming (serialize only relevant rows)
    #   - Card summary prepending
    #   - Metadata preservation for citations
    context_chunks = retrieved_chunks[:7]  # For now, just use top 7
    context_text = "\n\n".join(
        f"[Section {c.get('section_id')}] {c.get('section_title')}\n{c.get('text_raw', '')}"
        for c in context_chunks
    )
    took_context = time.time() - start

    trace.append(
        PipelineStage(
            stage="context_assembly",
            input={"chunk_count": len(retrieved_chunks)},
            output={
                "context_length": len(context_text),
                "chunk_count": len(context_chunks),
                "chunks_included": [
                    {
                        "id": c.get("id"),
                        "section_id": c.get("section_id"),
                        "content_type": c.get("content_type"),
                    }
                    for c in context_chunks
                ],
            },
            took_ms=took_context * 1000,
        )
    )

    # -------------------------------------------------------------------------
    # Stage 4: Generation (Sonnet with strict system prompt)
    # -------------------------------------------------------------------------
    start = time.time()
    # TODO: Implement generator.generate(query, context, context_chunks)
    #   Should return (answer, citations) where citations are structured dicts
    #   with {"text": quoted text, "source": section_id or chunk_id}
    answer = "Answer not yet implemented; generator.py needed"
    citations = []
    took_gen = time.time() - start

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
