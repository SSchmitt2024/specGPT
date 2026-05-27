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
import re
import time
from dataclasses import asdict, dataclass, field

from src.pipeline import query_processor, retriever, search, reranker, generator, table_serializer
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
    cross_encoder_model: str = "rerank-2-lite"

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

    # When True, the agentic gap-analyser can ALSO request specific figures /
    # fields / sections by name; the orchestrator fetches those directly from
    # the structured-lookup tables and merges them into the rerank pool.
    # Much more reliable than hoping a natural-language follow-up rediscovers
    # them, but costs an extra LLM round-trip to parse the request and one
    # Supabase lookup per requested artifact.
    agentic_targeted_fetch: bool = True

    # Recursive agentic mode: when True, re-run gap-analysis on each newly
    # regenerated answer and loop until the model stops requesting more
    # context (or `agentic_max_iterations` is hit). When False (default),
    # behavior is the original single-pass refinement.
    agentic_recursive: bool = False
    agentic_max_iterations: int = 5

    # When True and agentic mode is OFF, still run a one-shot gap analysis
    # after the first-pass answer and surface the result as `gap_hint` in
    # the response. Costs one extra LLM call per non-agentic query but lets
    # the UI prompt the user to opt into agentic refinement when the model
    # is asking for more context. Has no effect when agentic mode is on
    # (the agentic loop is the gap analyser).
    auto_gap_check: bool = True

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
question given what was retrieved, and if not, identify TWO complementary
forms of follow-up:

  (1) Free-form search QUERIES — focused questions a spec engineer would
      phrase. The pipeline runs each through hybrid retrieval (vector + BM25 +
      tsvector). Use these for "I need to know more about X concept".

  (2) Targeted REQUESTED_RESOURCES — specific figures, fields, or section IDs
      the answer mentions by NAME as missing or under-defined. The pipeline
      fetches each directly from the structured lookup tables (much faster,
      guaranteed hit when the artifact exists). Use these whenever the
      answer says things like:
        - "the context does not include Figure 630"
        - "Figure N is not provided / not in the retrieved context"
        - "the exact byte/field layout of <FIELD> is missing"
        - "Section X.Y.Z is not retrieved"
        - "the encoding/format of <FIELD> within Figure N is missing"

A follow-up is needed when the answer:
  - admits "the context does not contain X" or "information about Y is missing"
  - cites a section that wasn't actually retrieved (hallucinated reference)
  - leaves a specific bit/field/figure mentioned in the question undefined
  - mentions a related concept that wasn't itself retrieved

Each free-form query must be a complete, focused question. NEVER repeat the
original query verbatim. Cap at 3 queries.

Return JSON ONLY, matching this schema exactly. Omit a field rather than
guessing if you have no candidates for it.

  {
    "needs_followup": true|false,
    "reason": "one short sentence",
    "queries": ["..."],                 // 0-3 free-form follow-ups
    "requested_resources": {
      "figures":  ["630", "631"],       // figure numbers as strings (no "Figure" prefix)
      "fields":   ["PPI", "CDP"],       // uppercase identifiers, no qualifiers
      "sections": ["8.20.1"]            // dotted section ids
    }
  }
"""


def _parse_requested_resources(parsed: object) -> dict[str, list[str]]:
    """Defensively coerce the LLM's `requested_resources` block into a
    {figures, fields, sections} dict of clean string lists.

    Accepts any of: missing keys, wrong types, non-string members, integers
    instead of strings. Strips empties and dedupes (preserving order).
    """
    out = {"figures": [], "fields": [], "sections": []}
    if not isinstance(parsed, dict):
        return out
    req = parsed.get("requested_resources")
    if not isinstance(req, dict):
        return out

    # Strip leading "Figure ", "Fig.", "Fig ", "Section ", "§", "Appendix "
    # (case-insensitive, with or without trailing punctuation).
    _PREFIX_RE = re.compile(
        r"^\s*(?:figure|fig\.?|section|sect\.?|appendix|app\.?|§)\s*\.?\s*",
        re.IGNORECASE,
    )

    def _clean_list(raw, *, upper: bool = False, cap: int = 16) -> list[str]:
        if not isinstance(raw, list):
            return []
        seen: set[str] = set()
        items: list[str] = []
        for v in raw:
            if v is None:
                continue
            s = _PREFIX_RE.sub("", str(v)).strip()
            if not s or len(s) > 60:
                continue
            if upper:
                s = s.upper()
            if s in seen:
                continue
            seen.add(s)
            items.append(s)
            if len(items) >= cap:
                break
        return items

    out["figures"]  = _clean_list(req.get("figures"),  upper=False, cap=8)
    out["fields"]   = _clean_list(req.get("fields"),   upper=True,  cap=8)
    out["sections"] = _clean_list(req.get("sections"), upper=False, cap=5)
    return out


def _agentic_gap_analysis(
    *,
    query: str,
    answer: str,
    used_chunks: list[dict],
    citations: list[dict],
    max_followups: int,
) -> tuple[list[str], str, dict[str, list[str]]]:
    """Ask the classifier LLM if follow-up retrieval is needed.

    Returns ``(followup_queries, reason, requested_resources)``. On any
    failure, returns ``([], "<failure note>", {empty})`` so the agentic loop
    falls back to the single-pass answer.
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
            max_output_tokens=400,  # slightly larger to fit requested_resources
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("agentic gap-analysis failed: %s", e)
        return [], f"gap-analysis failed ({type(e).__name__})", {"figures": [], "fields": [], "sections": []}

    if not isinstance(parsed, dict):
        return [], "non-dict gap response", {"figures": [], "fields": [], "sections": []}

    requested = _parse_requested_resources(parsed)
    has_resources = any(requested.values())

    # If the LLM explicitly says no follow-ups but named resources, treat as
    # "yes, follow up by fetching those resources" — the model often forgets
    # the boolean when the resources block is populated.
    if not parsed.get("needs_followup") and not has_resources:
        return [], str(parsed.get("reason", "no follow-ups needed")), requested

    raw = parsed.get("queries") or []
    if not isinstance(raw, list):
        return [], "invalid queries field", requested
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
    return clean[:max_followups], str(parsed.get("reason", "")), requested


def _resolve_requested_resources(
    requested: dict[str, list[str]],
    *,
    enable_section_fallback: bool = True,
) -> list[dict]:
    """Direct-fetch chunks for specific figures/fields/sections.

    Returns chunk dicts matching the standard shape (compatible with the
    dedup/rerank pool). Tags ``method="agentic_fetch_*"`` so the trace can
    distinguish them from hybrid-search hits.
    """
    chunks: list[dict] = []
    if not any(requested.values()):
        return chunks

    try:
        tables_by_fig = retriever.load_tables_by_figure()
    except Exception as e:  # noqa: BLE001
        logger.warning("agentic targeted-fetch: tables load failed: %s", e)
        tables_by_fig = {}

    def _chunk_from_table(table: dict, *, src_tag: str, source: str) -> dict | None:
        fig = table.get("figure_number") or table.get("parent_figure")
        if fig is None:
            return None
        fig_s = str(fig)
        section_id = table.get("parent_section") or ""
        try:
            text = table_serializer.serialize_table(table)
        except Exception:
            text = table.get("raw_text") or ""
        return {
            "id": f"agentic_fetch:{src_tag}",
            "chunk_id": f"agentic_fetch:{src_tag}",
            "section_id": section_id,
            "section_title": table.get("caption") or f"Figure {fig_s}",
            "content_type": "table",
            "text_raw": text,
            "pdf_pages": [table.get("pdf_page")] if table.get("pdf_page") else [],
            "figure_number": fig_s,
            "has_normative": "shall" in (table.get("raw_text") or "").lower(),
            "score": 1.0,
            "method": source,
        }

    # ── Figures: direct table lookup ────────────────────────────────────
    for fig in requested.get("figures") or []:
        keys = {str(fig).strip(),
                str(fig).strip().lstrip("0") or str(fig).strip(),
                str(fig).strip().upper()}
        table = next((tables_by_fig[k] for k in keys if k in tables_by_fig), None)
        if not table:
            continue
        ch = _chunk_from_table(table, src_tag=f"fig{fig}", source="agentic_fetch_figure")
        if ch:
            chunks.append(ch)

    # ── Fields: resolve to parent figure(s) ─────────────────────────────
    try:
        field_index = retriever.load_field_index()
    except Exception as e:  # noqa: BLE001
        logger.warning("agentic targeted-fetch: field_index load failed: %s", e)
        field_index = {}

    for name in requested.get("fields") or []:
        recs = field_index.get(name) or field_index.get(name.upper()) or []
        if not isinstance(recs, list):
            recs = [recs]
        for rec in recs[:3]:
            parent = rec.get("parent_figure") if isinstance(rec, dict) else None
            if parent is None:
                continue
            table = tables_by_fig.get(str(parent))
            if not table:
                continue
            ch = _chunk_from_table(table, src_tag=f"field:{name}@fig{parent}",
                                   source="agentic_fetch_field")
            if ch:
                chunks.append(ch)

    # ── Sections: fall back to tsvector search keyed on the section id ──
    if enable_section_fallback:
        for sid in requested.get("sections") or []:
            try:
                hits = search.tsvector_search(sid, top_k=3,
                                              filter={"section_prefix": sid})
            except Exception:
                hits = []
            if not hits:
                try:
                    hits = search.tsvector_search(f"Section {sid}", top_k=3)
                except Exception:
                    hits = []
            for h in hits:
                h = dict(h)
                h["method"] = "agentic_fetch_section"
                h["score"] = max(float(h.get("score") or 0.0), 0.9)
                chunks.append(h)

    return chunks


def _run_stage5_and_finalize(
    *,
    query: str,
    config: PipelineConfig,
    debug: bool,
    agentic: bool,
    trace: list,
    deduplicated: list[dict],
    answer: str,
    citations: list[dict],
    context_chunks: list[dict],
    tokens_used: dict | None,
) -> dict:
    """Run Stage 5 (agentic refinement loop) + Stage 6 (non-agentic gap hint)
    and assemble the response. Extracted so /api/refine can reuse it on top
    of a seeded first-pass state without redoing Stages 1–4.
    """
    if agentic:
        # Recursive mode loops up to `agentic_max_iterations`; single-pass
        # mode is iter 0 only (original behavior). When non-recursive, stage
        # names omit the `.iterN` suffix so existing trace consumers see no
        # diff.
        max_iters = max(1, config.agentic_max_iterations) if config.agentic_recursive else 1
        # Pool accumulates across iterations so we never lose chunks already
        # retrieved (each rerank sees everything fetched so far).
        expanded_pool: list[dict] = list(deduplicated)
        last_gap_reason = ""
        converged = False

        for iteration in range(max_iters):
            suffix = f".iter{iteration}" if config.agentic_recursive else ""

            start = time.time()
            followups, gap_reason, requested = _agentic_gap_analysis(
                query=query,
                answer=answer,
                used_chunks=context_chunks,
                citations=citations,
                max_followups=config.agentic_max_followups,
            )
            took_gap = time.time() - start
            last_gap_reason = gap_reason
            targeted_requested = (
                requested if config.agentic_targeted_fetch else {"figures": [], "fields": [], "sections": []}
            )
            gap_has_work = bool(followups) or any(targeted_requested.values())
            trace.append(
                PipelineStage(
                    stage=f"agentic.gap_analysis{suffix}",
                    input={"answer_chars": len(answer or ""),
                           "max_followups": config.agentic_max_followups,
                           "targeted_fetch_enabled": config.agentic_targeted_fetch,
                           "iteration": iteration},
                    output={"needs_followup": gap_has_work,
                            "reason": gap_reason,
                            "queries": followups,
                            "requested_resources": targeted_requested},
                    took_ms=took_gap * 1000,
                )
            )

            if not gap_has_work:
                converged = True
                break

            extra_chunks: list[dict] = []

            if any(targeted_requested.values()):
                start = time.time()
                try:
                    fetched = _resolve_requested_resources(targeted_requested)
                except Exception as e:  # noqa: BLE001
                    logger.exception("agentic targeted-fetch failed: %s", e)
                    fetched = []
                took_fetch = time.time() - start
                by_method: dict[str, int] = {}
                for ch in fetched:
                    by_method[ch.get("method", "?")] = by_method.get(ch.get("method", "?"), 0) + 1
                trace.append(
                    PipelineStage(
                        stage=f"agentic.targeted_fetch{suffix}",
                        input={"requested": targeted_requested,
                               "totals": {k: len(v) for k, v in targeted_requested.items()}},
                        output={"fetched_count": len(fetched),
                                "by_method": by_method,
                                "fetched": [
                                    {"id": c.get("id"),
                                     "section_id": c.get("section_id"),
                                     "section_title": c.get("section_title"),
                                     "figure_number": c.get("figure_number"),
                                     "method": c.get("method")}
                                    for c in fetched[:10]
                                ]},
                        took_ms=took_fetch * 1000,
                    )
                )
                extra_chunks.extend(fetched)

            for fi, fq in enumerate(followups):
                # Mirror Stage 1 for each follow-up: classify + decompose +
                # entity-extract before retrieval. Previously we passed the
                # follow-up verbatim into hybrid_search with sub_queries=[fq],
                # skipping decomposition entirely — multi-part follow-ups got
                # the same shallow retrieval as a one-shot keyword search.
                start = time.time()
                try:
                    fq_decomp = query_processor.process_query(
                        fq, use_llm=True, max_subqueries=config.max_subqueries,
                    )
                    fq_subs = fq_decomp.sub_queries or [fq]
                    fq_decomp_summary = {
                        "type": fq_decomp.type,
                        "entities": _entity_list_to_dict(fq_decomp.entities),
                        "sub_queries": fq_subs,
                        "rationale": fq_decomp.rationale,
                    }
                except Exception as e:  # noqa: BLE001
                    logger.warning("agentic follow-up decomp failed (%s): falling back to verbatim", e)
                    fq_subs = [fq]
                    fq_decomp_summary = {"error": type(e).__name__, "sub_queries": fq_subs}
                took_decomp = time.time() - start
                trace.append(
                    PipelineStage(
                        stage=f"agentic.followup_decomp_q{fi}{suffix}",
                        input={"query": fq},
                        output=fq_decomp_summary,
                        took_ms=took_decomp * 1000,
                    )
                )

                start = time.time()
                fq_chunks, fq_trace = hybrid_search(fq, sub_queries=fq_subs, config=config)
                took_fq = time.time() - start
                trace.append(
                    PipelineStage(
                        stage=f"agentic.followup_search_q{fi}{suffix}",
                        input={"query": fq, "sub_queries": fq_subs},
                        output={"chunk_count": len(fq_chunks)},
                        took_ms=took_fq * 1000,
                    )
                )
                # Namespace the follow-up's hybrid_search sub-stages so they
                # don't collide with the MAIN query's hybrid_search.* stages
                # (both emit names like `hybrid_search.vector_search_q0`).
                # Without this, the viz's stage-dict lookup is last-wins and
                # the main query's retrieval nodes get silently overwritten
                # by follow-up data.
                ns_prefix = f"agentic.followup_q{fi}{suffix}"
                for sub in fq_trace:
                    trace.append(
                        PipelineStage(
                            stage=f"{ns_prefix}.{sub.stage}",
                            input=sub.input,
                            output=sub.output,
                            took_ms=sub.took_ms,
                        )
                    )
                extra_chunks.extend(fq_chunks)

            start = time.time()
            seen2: set = set()
            merged_pool: list[dict] = []
            for chunk in expanded_pool + extra_chunks:
                cid = chunk.get("id") or chunk.get("chunk_id")
                if not cid:
                    cid = ("__no_id__", chunk.get("section_id"),
                           chunk.get("figure_number"), chunk.get("content_type"),
                           (chunk.get("text_raw") or "")[:120])
                if cid not in seen2:
                    seen2.add(cid)
                    merged_pool.append(chunk)
            expanded_pool = merged_pool
            took_merge = time.time() - start

            start = time.time()
            reranked2 = reranker.rerank(
                query, expanded_pool,
                top_k=config.agentic_rerank_topk,
                model_name=config.cross_encoder_model,
                text_field="text_raw",
            )
            took_rr2 = time.time() - start
            trace.append(
                PipelineStage(
                    stage=f"agentic.rerank{suffix}",
                    input={"chunk_count": len(expanded_pool),
                           "top_k": config.agentic_rerank_topk,
                           "added_by_followups": len(extra_chunks)},
                    output={"results": _result_summary(reranked2),
                            "count": len(reranked2),
                            "merge_ms": took_merge * 1000},
                    took_ms=took_rr2 * 1000,
                )
            )

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
                        stage=f"agentic.regenerate{suffix}",
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
                answer, citations, context_chunks, tokens_used = answer2, citations2, used2, tokens2
                # Keep the cached pool in sync so the refine cache (or a later
                # iteration) sees the post-merge pool, not just first-pass.
                deduplicated = expanded_pool
            except Exception as e:
                took_g2 = time.time() - start
                logger.exception("Agentic regenerate failed after %.0fms: %s",
                                 took_g2 * 1000, e)
                trace.append(
                    PipelineStage(
                        stage=f"agentic.regenerate{suffix}",
                        input={"chunk_count": len(reranked2),
                               "model": config.agentic_model},
                        output={"error_type": type(e).__name__,
                                "note": "kept prior answer"},
                        took_ms=took_g2 * 1000,
                    )
                )
                break

        if config.agentic_recursive and not converged:
            trace.append(
                PipelineStage(
                    stage="agentic.cap_reached",
                    input={"max_iterations": max_iters},
                    output={"iterations_run": max_iters,
                            "last_gap_reason": last_gap_reason,
                            "note": "stopped at max_iterations before model declared done"},
                    took_ms=0.0,
                )
            )

    # Stage 6: non-agentic gap hint
    gap_hint: dict | None = None
    if not agentic and config.auto_gap_check:
        start = time.time()
        try:
            gh_followups, gh_reason, gh_requested = _agentic_gap_analysis(
                query=query,
                answer=answer,
                used_chunks=context_chunks,
                citations=citations,
                max_followups=config.agentic_max_followups,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("auto_gap_check failed: %s", e)
            gh_followups, gh_reason, gh_requested = [], "", {"figures": [], "fields": [], "sections": []}
        took_gh = time.time() - start
        needs = bool(gh_followups) or any(gh_requested.values())
        gap_hint = {
            "needs_followup": needs,
            "reason": gh_reason,
            "queries": gh_followups,
            "requested_resources": gh_requested,
        }
        trace.append(
            PipelineStage(
                stage="gap_hint",
                input={"agentic": False, "auto_gap_check": True},
                output=gap_hint,
                took_ms=took_gh * 1000,
            )
        )

    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "sources": context_chunks,
        "deduplicated": deduplicated,
        "config": config.to_dict(),
        "agentic": agentic,
        "gap_hint": gap_hint,
        "tokens_used": tokens_used,
        "pipeline_trace": [s.to_dict() for s in trace] if debug else [],
    }


def orchestrate(
    query: str,
    *,
    config: PipelineConfig | None = None,
    debug: bool = True,
    agentic: bool = False,
    refine_seed: dict | None = None,
) -> dict:
    """
    Execute the full retrieval + generation pipeline.

    Args:
        query: the user's question.
        config: PipelineConfig with tunable parameters. Defaults to PipelineConfig().
        debug: if True, include full pipeline_trace in response.
        refine_seed: when provided, skip Stages 1–4 and seed Stage 5 with the
            prior first-pass state. Shape:
                {"deduplicated": [chunk...],
                 "answer": str,
                 "citations": [...],
                 "context_chunks": [...],
                 "tokens_used": dict | None}
            Forces ``agentic=True`` regardless of the flag.

    Returns:
        {
            "answer": str,
            "citations": [...],
            "sources": [chunk dicts],
            "deduplicated": [chunk dicts],    # pre-rerank pool, for refine cache
            "config": ...,
            "agentic": bool,
            "gap_hint": dict | None,
            "tokens_used": dict | None,
            "pipeline_trace": [...] if debug else [],
        }
    """
    if config is None:
        config = PipelineConfig()

    trace: list[PipelineStage] = []
    tokens_used: dict | None = None

    # -------------------------------------------------------------------------
    # Refine fast-path: skip Stages 1–4 and jump straight to Stage 5 with the
    # prior first-pass state. Used by /api/refine when the user clicks
    # "Run agentic refinement" from the sidebar — avoids re-doing the work
    # we already did for the initial query.
    # -------------------------------------------------------------------------
    if refine_seed is not None:
        agentic = True
        deduplicated = list(refine_seed.get("deduplicated") or [])
        answer = refine_seed.get("answer") or ""
        citations = list(refine_seed.get("citations") or [])
        context_chunks = list(refine_seed.get("context_chunks") or [])
        tokens_used = refine_seed.get("tokens_used")
        trace.append(
            PipelineStage(
                stage="refine.seed",
                input={"reason": "skipping Stages 1–4 with cached first-pass state"},
                output={"deduplicated_count": len(deduplicated),
                        "context_chunk_count": len(context_chunks),
                        "answer_chars": len(answer),
                        "citation_count": len(citations)},
                took_ms=0.0,
            )
        )
        # Jump straight to Stage 5 — see below.
        return _run_stage5_and_finalize(
            query=query, config=config, debug=debug,
            agentic=agentic, trace=trace,
            deduplicated=deduplicated, answer=answer,
            citations=citations, context_chunks=context_chunks,
            tokens_used=tokens_used,
        )

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
    # Stage 5 (agentic loop) + Stage 6 (non-agentic gap hint) + assembly.
    # Extracted so /api/refine can call this same helper with a seeded
    # first-pass state and skip Stages 1–4 entirely.
    return _run_stage5_and_finalize(
        query=query, config=config, debug=debug, agentic=agentic,
        trace=trace, deduplicated=deduplicated,
        answer=answer, citations=citations, context_chunks=context_chunks,
        tokens_used=tokens_used,
    )


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
