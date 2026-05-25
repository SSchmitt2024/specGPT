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
  - tsvector_search.top_k
  - bm25_search.top_k
  - rrf_merge.k
  - rrf_output.top_k
  - reranker.top_k
  - query_processor.max_subqueries
  - reranker.model_name

    config = PipelineConfig(
        vector_topk=15,
        tsvector_topk=15,
        bm25_topk=15,
        rrf_k=45,
        final_rerank_topk=10,
    )
    result = orchestrate("What is bit 7:4 of CDW10?", config=config)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field

from src.pipeline import query_processor, retriever, search, reranker, generator
from src.pipeline.query_processor import QueryDecomposition

logger = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    """Raised when context retrieval succeeded but generation failed.

    Carries the partial pipeline trace + the underlying cause so the web
    layer can decide how to surface the failure (e.g., HTTP 502 with a
    request id while still serving the retrieval trace for debugging).
    """

    def __init__(self, message: str, *, cause: Exception, trace: list[dict] | None = None,
                 retrieved_chunks: list[dict] | None = None):
        super().__init__(message)
        self.cause = cause
        self.trace = trace or []
        self.retrieved_chunks = retrieved_chunks or []


@dataclass
class PipelineConfig:
    """Configuration for all tunable high-impact parameters."""
    # Search parameters
    vector_topk: int = 10
    tsvector_topk: int = 10
    bm25_topk: int = 10

    # RRF merge parameters
    rrf_k: int = 60
    rrf_output_topk: int = 20

    # Reranking parameters
    final_rerank_topk: int = 7
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Query decomposition parameters
    max_subqueries: int = 3

    # Generation parameters
    llm_model: str = "claude-sonnet-4-5"
    llm_max_context_tokens: int = 4000
    llm_max_output_tokens: int = 1024

    # Agentic-mode parameters (only used when orchestrate(..., agentic=True))
    agentic_model: str = "claude-opus-4-7"
    agentic_max_context_tokens: int = 16000
    agentic_max_output_tokens: int = 2048
    agentic_max_followups: int = 3   # cap LLM-generated follow-up queries
    agentic_rerank_topk: int = 14    # top-k after re-rerank (~2× normal)

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
    Orchestrate hybrid retrieval: vector + tsvector + BM25 per sub-query,
    fused via Reciprocal Rank Fusion.

    Each (method, sub-query) pair is treated as an independent ranked list
    in the RRF input — that's what makes the fusion work. Flattening them
    into a single list before RRF would collapse all rank information.

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
    ranked_lists: list[list[dict]] = []
    total_input = 0

    # Step 1: vector + tsvector + bm25 per sub-query, each as its own ranked list
    for i, sq in enumerate(sub_queries):
        start = time.time()
        vec_results = search.vector_search(sq, top_k=config.vector_topk)
        took_vec = time.time() - start

        start = time.time()
        tsv_results = search.tsvector_search(sq, top_k=config.tsvector_topk)
        took_tsv = time.time() - start

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
                stage=f"hybrid_search.tsvector_search_q{i}",
                input={"query": sq, "top_k": config.tsvector_topk},
                output={"results": _result_summary(tsv_results, limit=3), "count": len(tsv_results)},
                took_ms=took_tsv * 1000,
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

        for lst in (vec_results, tsv_results, bm25_results):
            if lst:
                ranked_lists.append(lst)
                total_input += len(lst)

    # Step 2: RRF merge across all (method, sub-query) ranked lists
    start = time.time()
    merged = retriever.rrf_merge(ranked_lists, k=config.rrf_k, top_k=config.rrf_output_topk)
    took_rrf = time.time() - start

    sub_trace.append(
        PipelineStage(
            stage="hybrid_search.rrf_merge",
            input={
                "result_lists": len(ranked_lists),
                "total_input": total_input,
                "k": config.rrf_k,
            },
            output={"results": _result_summary(merged, limit=3), "count": len(merged)},
            took_ms=took_rrf * 1000,
        )
    )

    # Cross-encoder reranking is applied later at the orchestrator level
    # on the merged pool (structured + hybrid), not here.
    return merged, sub_trace


_AGENTIC_GAP_SYSTEM = """You are a quality reviewer for an NVMe specification Q&A system.

A retrieval pipeline produced an answer to a user question using the provided
retrieved sections. Your job: decide whether the answer adequately covers the
question given what was retrieved, and if not, produce 1-3 focused follow-up
search queries that would surface the missing information.

A follow-up is needed when the answer:
  - admits "the context does not contain X" or "information about Y is missing"
  - cites a section that wasn't actually retrieved (hallucinated reference)
  - leaves a specific bit/field/figure mentioned in the question undefined
  - mentions a related concept that wasn't itself retrieved

Each follow-up query must be a complete, focused question on its own — the
kind a spec engineer would phrase to look up a specific definition, field
layout, or behavior. NEVER repeat the original query verbatim.

Return JSON ONLY:
  {
    "needs_followup": true|false,
    "reason": "one short sentence",
    "queries": ["...", "..."]   // empty array if needs_followup is false
  }
"""


def _agentic_gap_analysis(
    *,
    query: str,
    answer: str,
    used_chunks: list[dict],
    citations: list[dict],
    max_followups: int,
) -> tuple[list[str], str]:
    """Ask the classifier LLM if follow-up retrieval is needed; return queries + reason.

    On any failure, returns ([], "<failure note>") so the agentic loop falls
    back to the single-pass answer.
    """
    used_titles = "\n".join(
        f"  - [{c.get('section_id','?')}] {c.get('section_title','')}"
        for c in used_chunks[:20]
    ) or "  (none)"
    halluc = [c for c in citations if c.get("hallucinated")]
    halluc_block = ""
    if halluc:
        halluc_block = "\nCitations not found in retrieved context:\n" + "\n".join(
            f"  - Section {c.get('section_id')}" for c in halluc
        )

    user_prompt = (
        f"Original question:\n  {query}\n\n"
        f"Retrieved sections fed to the answerer:\n{used_titles}\n"
        f"{halluc_block}\n"
        f"Answer (first 1500 chars):\n<<<\n{(answer or '')[:1500]}\n>>>\n"
    )

    try:
        parsed, _ = query_processor.generate_json(  # via the LLM client
            user_prompt,
            system=_AGENTIC_GAP_SYSTEM,
            temperature=0.0,
            max_output_tokens=300,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("agentic gap-analysis failed: %s", e)
        return [], f"gap-analysis failed ({type(e).__name__})"

    if not isinstance(parsed, dict) or not parsed.get("needs_followup"):
        return [], str(parsed.get("reason", "no follow-ups needed")) if isinstance(parsed, dict) else "non-dict gap response"

    raw = parsed.get("queries") or []
    if not isinstance(raw, list):
        return [], "invalid queries field"
    clean: list[str] = []
    seen: set[str] = set()
    qnorm = " ".join(query.split()).lower()
    for q in raw:
        s = " ".join(str(q).split())
        if len(s) < 4 or len(s) > 400:
            continue
        if s.lower() == qnorm:
            continue  # skip exact-duplicate of original
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        clean.append(s)
    return clean[:max_followups], str(parsed.get("reason", ""))


def orchestrate(
    query: str,
    *,
    config: PipelineConfig | None = None,
    debug: bool = True,
    agentic: bool = False,
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
                "rationale": decomp.rationale,
            },
            took_ms=took_qp * 1000,
        )
    )

    # -------------------------------------------------------------------------
    # Stage 2a: Structured Lookup (always attempt if lookup query)
    # -------------------------------------------------------------------------
    structured_chunks: list[dict] = []

    if (decomp.type or "").lower() == "lookup" and decomp.entities:
        start = time.time()
        struct_result = retriever.structured_lookup(
            decomp,
            use_llm=False,  # already did LLM in query_processor
            max_fields=8,
        )
        took_struct = time.time() - start

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
    # Always include the verbatim original query so RRF can pick up direct
    # phrase matches the LLM's reworded sub-queries miss.
    search_queries: list[str] = [query]
    for sq in decomp.sub_queries or []:
        if sq and sq.strip() and sq.strip() != query.strip() and sq not in search_queries:
            search_queries.append(sq)

    hybrid_chunks, hybrid_trace = hybrid_search(
        query,
        sub_queries=search_queries,
        config=config,
    )
    took_hybrid = time.time() - start

    trace.extend(hybrid_trace)
    trace.append(
        PipelineStage(
            stage="hybrid_search.total",
            input={"sub_queries": search_queries},
            output={"chunk_count": len(hybrid_chunks)},
            took_ms=took_hybrid * 1000,
        )
    )

    # -------------------------------------------------------------------------
    # Stage 2c: Merge results from both paths
    # -------------------------------------------------------------------------
    start = time.time()
    all_chunks = structured_chunks + hybrid_chunks

    # Deduplicate by id, keeping first occurrence (structured has priority).
    # Chunks missing an id (e.g. structured_lookup synthetic rows) must NOT
    # collide on the shared None key — fall back to chunk_id, then a
    # content-derived surrogate so each missing-id chunk is treated as unique.
    seen_ids: set = set()
    deduplicated: list[dict] = []
    for chunk in all_chunks:
        chunk_id = chunk.get("id") or chunk.get("chunk_id")
        if not chunk_id:
            surrogate = (
                chunk.get("section_id"),
                chunk.get("figure_number"),
                chunk.get("content_type"),
                (chunk.get("text_raw") or "")[:120],
            )
            chunk_id = ("__no_id__", surrogate)
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
            model=config.llm_model,
            max_context_tokens=config.llm_max_context_tokens,
            max_tokens=config.llm_max_output_tokens,
        )
        took_gen = time.time() - start

        trace.append(
            PipelineStage(
                stage="generation",
                input={
                    "query": query,
                    "chunk_count": len(retrieved_chunks),
                    "model": config.llm_model,
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
        took_gen = time.time() - start
        # Trace the failure so debug callers still see what happened, then
        # re-raise as GenerationError. The caller (app.py) converts to 5xx
        # so the user doesn't get the raw error string as their "answer".
        logger.exception("Generation failed after %.0fms: %s", took_gen * 1000, e)
        trace.append(
            PipelineStage(
                stage="generation",
                input={"query": query, "chunk_count": len(retrieved_chunks),
                       "model": config.llm_model},
                output={"error_type": type(e).__name__},
                took_ms=took_gen * 1000,
            )
        )
        raise GenerationError(
            f"Generation failed: {type(e).__name__}",
            cause=e,
            trace=[s.to_dict() for s in trace],
            retrieved_chunks=retrieved_chunks,
        ) from e

    # -------------------------------------------------------------------------
    # Stage 5 (optional): Agentic refinement
    #
    # On request, ask a small LLM whether the answer covers the question given
    # the retrieved chunks. If it identifies gaps, retrieve more (using the
    # gap-derived follow-up queries), merge into the pool, re-rerank, and
    # regenerate with the higher-tier model + larger context budget.
    # -------------------------------------------------------------------------
    if agentic:
        start = time.time()
        followups, gap_reason = _agentic_gap_analysis(
            query=query,
            answer=answer,
            used_chunks=context_chunks,
            citations=citations,
            max_followups=config.agentic_max_followups,
        )
        took_gap = time.time() - start
        trace.append(
            PipelineStage(
                stage="agentic.gap_analysis",
                input={"answer_chars": len(answer or ""),
                       "max_followups": config.agentic_max_followups},
                output={"needs_followup": bool(followups), "reason": gap_reason,
                        "queries": followups},
                took_ms=took_gap * 1000,
            )
        )

        if followups:
            # Per-followup hybrid_search (same fan-out as the main path but
            # without re-running the LLM-based decomposer; the gap analyser
            # already produced focused queries).
            extra_chunks: list[dict] = []
            for fi, fq in enumerate(followups):
                start = time.time()
                fq_chunks, fq_trace = hybrid_search(fq, sub_queries=[fq], config=config)
                took_fq = time.time() - start
                trace.append(
                    PipelineStage(
                        stage=f"agentic.followup_search_q{fi}",
                        input={"query": fq},
                        output={"chunk_count": len(fq_chunks)},
                        took_ms=took_fq * 1000,
                    )
                )
                trace.extend(fq_trace)
                extra_chunks.extend(fq_chunks)

            # Merge new chunks with the prior pool. We re-dedup against the
            # full pre-rerank pool (`deduplicated` from stage 2c) so we keep
            # the structured-lookup sources too.
            start = time.time()
            seen2: set = set()
            expanded: list[dict] = []
            for chunk in deduplicated + extra_chunks:
                cid = chunk.get("id") or chunk.get("chunk_id")
                if not cid:
                    cid = ("__no_id__", chunk.get("section_id"),
                           chunk.get("figure_number"), chunk.get("content_type"),
                           (chunk.get("text_raw") or "")[:120])
                if cid not in seen2:
                    seen2.add(cid)
                    expanded.append(chunk)
            took_merge = time.time() - start

            # Re-rerank the expanded pool against the original query, keeping
            # a wider top-k so Opus has more material to synthesize.
            start = time.time()
            reranked2 = reranker.rerank(
                query, expanded,
                top_k=config.agentic_rerank_topk,
                model_name=config.cross_encoder_model,
                text_field="text_raw",
            )
            took_rr2 = time.time() - start
            trace.append(
                PipelineStage(
                    stage="agentic.rerank",
                    input={"chunk_count": len(expanded),
                           "top_k": config.agentic_rerank_topk,
                           "added_by_followups": len(extra_chunks)},
                    output={"results": _result_summary(reranked2),
                            "count": len(reranked2),
                            "merge_ms": took_merge * 1000},
                    took_ms=took_rr2 * 1000,
                )
            )

            # Regenerate with Opus + larger context.
            start = time.time()
            try:
                answer2, citations2, used2, tokens2 = generator.generate(
                    query, reranked2,
                    model=config.agentic_model,
                    max_context_tokens=config.agentic_max_context_tokens,
                    max_tokens=config.agentic_max_output_tokens,
                )
                took_g2 = time.time() - start
                trace.append(
                    PipelineStage(
                        stage="agentic.regenerate",
                        input={"chunk_count": len(reranked2),
                               "model": config.agentic_model,
                               "max_context_tokens": config.agentic_max_context_tokens},
                        output={"answer_length": len(answer2),
                                "citation_count": len(citations2),
                                "tokens": tokens2,
                                "context_used": [
                                    {"section_id": c.get("section_id"),
                                     "section_title": c.get("section_title"),
                                     "content_type": c.get("content_type")}
                                    for c in used2
                                ]},
                        took_ms=took_g2 * 1000,
                    )
                )
                # Promote the agentic answer as the canonical result.
                answer, citations, context_chunks = answer2, citations2, used2
            except Exception as e:
                # Agentic regenerate failed — keep the first-pass answer so the
                # user still gets something. Log the cause prominently.
                took_g2 = time.time() - start
                logger.exception("Agentic regenerate failed after %.0fms: %s",
                                 took_g2 * 1000, e)
                trace.append(
                    PipelineStage(
                        stage="agentic.regenerate",
                        input={"chunk_count": len(reranked2),
                               "model": config.agentic_model},
                        output={"error_type": type(e).__name__,
                                "note": "kept first-pass answer"},
                        took_ms=took_g2 * 1000,
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
        "agentic": agentic,
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
