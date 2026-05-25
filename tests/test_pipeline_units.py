"""
Unit-level smoke tests for pipeline pure-Python helpers.

These exercise the behavioural fixes from the audit sweep without hitting
Supabase / Voyage / Anthropic. Run from the project root:

    venv/bin/python3 -m pytest tests/test_pipeline_units.py

Or, with no pytest installed, the asserts in __main__ at the bottom of the
file run the same checks via `venv/bin/python3 tests/test_pipeline_units.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# table_serializer.row alignment

def test_normalize_row_pads_short_rows():
    from src.pipeline.table_serializer import _normalize_row
    assert _normalize_row(["a", "b"], 4) == ["a", "b", "", ""]


def test_normalize_row_joins_overflow_into_last_cell():
    from src.pipeline.table_serializer import _normalize_row
    assert _normalize_row(["a", "b", "c", "d", "e"], 3) == ["a", "b", "c d e"]


def test_normalize_row_handles_none_row():
    from src.pipeline.table_serializer import _normalize_row
    assert _normalize_row(None, 3) == ["", "", ""]


def test_serialize_table_aligns_rows_with_header():
    from src.pipeline.table_serializer import serialize_table
    out = serialize_table({
        "caption": "Cap",
        "figure_number": "42",
        "headers": ["H1", "H2", "H3"],
        "rows": [["x"], ["a", "b", "c", "d"], ["p", "q", "r"]],
    })
    assert out.splitlines() == [
        "Figure 42 — Cap",
        "H1 | H2 | H3",
        "---",
        "x |  | ",
        "a | b | c d",
        "p | q | r",
    ]


# ---------------------------------------------------------------------------
# generator helpers

def test_table_header_line_count_detects_caption_header_separator():
    from src.pipeline.generator import _table_header_line_count
    assert _table_header_line_count(["Figure 1 — Cap", "col1 | col2", "---", "row1"]) == 3


def test_table_header_line_count_no_caption():
    from src.pipeline.generator import _table_header_line_count
    assert _table_header_line_count(["col1 | col2", "---", "row1"]) == 2


def test_table_header_line_count_empty():
    from src.pipeline.generator import _table_header_line_count
    assert _table_header_line_count([]) == 0


def test_extract_citations_strips_trailing_dot():
    from src.pipeline.generator import _extract_citations
    cits = _extract_citations(
        "See Section 5.2.1. for details.",
        [{"section_id": "5.2.1", "section_title": "X", "content_type": "prose"}],
    )
    assert cits == [{
        "section_id": "5.2.1", "section_title": "X",
        "content_type": "prose", "hallucinated": False,
    }]


def test_extract_citations_flags_hallucinated_sections():
    from src.pipeline.generator import _extract_citations
    cits = _extract_citations(
        "Per Section 9.9.9 the spec says...",
        [{"section_id": "5.2.1", "section_title": "X", "content_type": "prose"}],
    )
    assert cits[0]["hallucinated"] is True


def test_extract_citations_handles_alphabetic_appendix_sections():
    """Bug found during local boot: appendix sections (A.1, B.3, ...) were
    silently dropped because the regex only matched purely numeric IDs."""
    from src.pipeline.generator import _extract_citations
    ctx = [
        {"section_id": "A.1", "section_title": "Appendix A1", "content_type": "prose"},
        {"section_id": "B.3", "section_title": "Appendix B3", "content_type": "prose"},
    ]
    cits = _extract_citations(
        "See Section A.1 and per Section B.3 the spec defines this.", ctx
    )
    ids = sorted(c["section_id"] for c in cits)
    assert ids == ["A.1", "B.3"], ids
    assert all(c["hallucinated"] is False for c in cits)


def test_extract_citations_matches_plural_sections_keyword():
    """Bug found during local boot: the LLM writes 'Sections X, Y' (plural)
    when citing multiple — previously the regex required singular only."""
    from src.pipeline.generator import _extract_citations
    ctx = [
        {"section_id": "5.2.13.2.11", "section_title": "X", "content_type": "prose"},
    ]
    cits = _extract_citations(
        "See Sections 5.2.13.2.11 for details.", ctx
    )
    assert [c["section_id"] for c in cits] == ["5.2.13.2.11"]


def test_extract_citations_matches_appendix_prefix():
    """The LLM sometimes writes 'Appendix A.1' instead of 'Section A.1'."""
    from src.pipeline.generator import _extract_citations
    cits = _extract_citations(
        "Per Appendix B.3 this defines... and Appendices A.1 contain...",
        [
            {"section_id": "B.3", "section_title": "X", "content_type": "prose"},
            {"section_id": "A.1", "section_title": "Y", "content_type": "prose"},
        ],
    )
    ids = sorted(c["section_id"] for c in cits)
    assert ids == ["A.1", "B.3"]


def test_extract_citations_skips_bare_single_letter():
    """`Section B` (no sub-section) would generate too many false positives
    mid-prose, so the regex requires at least one dotted sub-segment after
    the alphabetic prefix."""
    from src.pipeline.generator import _extract_citations
    cits = _extract_citations(
        "Section B is the appendix; per Section B.1 the rule is...",
        [{"section_id": "B.1", "section_title": "B1", "content_type": "prose"}],
    )
    ids = [c["section_id"] for c in cits]
    assert ids == ["B.1"]


def test_assemble_context_continues_past_oversized_chunks():
    """Audit item: budget loop used to `break` on first oversized chunk; now
    it `continue`s so a smaller later chunk still gets included."""
    from src.pipeline.generator import assemble_context
    chunks = [
        # ~400 tokens, oversized for a 100-token budget
        {"id": "a", "text_raw": "word " * 400, "content_type": "prose",
         "section_id": "1", "section_title": "A"},
        # ~2 tokens, fits
        {"id": "b", "text_raw": "small bit", "content_type": "prose",
         "section_id": "2", "section_title": "B"},
    ]
    _ctx, used = assemble_context("q", chunks, max_context_tokens=100)
    assert len(used) == 1 and used[0]["section_id"] == "2"


def test_assemble_context_wraps_chunks_in_fences():
    from src.pipeline.generator import assemble_context, _CHUNK_FENCE
    chunks = [{"id": "a", "text_raw": "hello", "content_type": "prose",
               "section_id": "1.1", "section_title": "Intro"}]
    ctx, _ = assemble_context("q", chunks, max_context_tokens=4000)
    assert (_CHUNK_FENCE % 1) in ctx and (_CHUNK_FENCE % "END 1") in ctx


def test_extract_text_concatenates_text_blocks_and_skips_tool_use():
    from src.pipeline.generator import _extract_text

    class _Block:
        def __init__(self, type, text=None):
            self.type = type
            self.text = text

    class _Resp:
        def __init__(self, blocks):
            self.content = blocks

    r = _Resp([_Block("tool_use"), _Block("text", "hello "), _Block("text", "world")])
    assert _extract_text(r) == "hello world"


# ---------------------------------------------------------------------------
# bm25_index

def test_tokenize_lowercases_and_strips_stopwords():
    from src.pipeline.bm25_index import tokenize
    toks = tokenize("CDW10 FUSE the and")
    assert "cdw10" in toks and "fuse" in toks
    assert "the" not in toks and "and" not in toks


def test_stopwords_aligned_with_postgres_english():
    from src.pipeline.bm25_index import _STOPWORDS
    # spot check Snowball english additions that the old list missed
    for w in ["because", "should", "between", "above", "below", "whom", "until"]:
        assert w in _STOPWORDS, f"missing english stopword: {w}"


# ---------------------------------------------------------------------------
# query_processor

def test_heuristic_type_lookup_for_single_field_entity():
    from src.pipeline.query_processor import _heuristic_type, Entity
    assert _heuristic_type("What are bits 7:4 of CDW10?", [Entity("CDW10", "cdw")]) == "lookup"


def test_heuristic_type_relational_for_multi_entity_interaction():
    from src.pipeline.query_processor import _heuristic_type, Entity
    out = _heuristic_type(
        "How do FID 0x01 and FID 0x12 interact?",
        [Entity("FID 0x01", "fid"), Entity("FID 0x12", "fid")],
    )
    # Either is acceptable — both indicate the old "always lookup" heuristic
    # has been replaced with a more nuanced classifier.
    assert out in ("relational", "procedural")


def test_heuristic_type_procedural_for_initialization_question():
    from src.pipeline.query_processor import _heuristic_type
    assert _heuristic_type("How do I initialize the controller?", []) == "procedural"


def test_heuristic_type_structural_for_describe_question():
    from src.pipeline.query_processor import _heuristic_type
    assert _heuristic_type("Describe the Identify Controller data structure.", []) == "structural"


def test_normalize_llm_output_dedupes_and_caps_subqueries():
    from src.pipeline.query_processor import _normalize_llm_output
    typ, subs, _ = _normalize_llm_output(
        {
            "type": "relational",
            "sub_queries": ["  ", "a", "How do X work?", "How do X work?", "x" * 500],
            "rationale": "r",
        },
        "orig",
        max_subqueries=3,
    )
    assert typ == "relational"
    assert subs == ["How do X work?"]


def test_normalize_llm_output_lookup_collapses_to_original():
    from src.pipeline.query_processor import _normalize_llm_output
    typ, subs, _ = _normalize_llm_output(
        {"type": "lookup", "sub_queries": ["a", "b", "c"], "rationale": ""},
        "orig query",
    )
    assert typ == "lookup" and subs == ["orig query"]


# ---------------------------------------------------------------------------
# retriever.rrf_merge

def test_rrf_merge_does_not_collapse_idless_chunks():
    """Audit item: dedup keyed by None collapsed all id-less chunks."""
    from src.pipeline.retriever import rrf_merge
    lists = [
        [
            {"id": None, "section_id": "s1", "figure_number": "f1",
             "content_type": "table", "text_raw": "aaa", "method": "structured_lookup"},
            {"id": None, "section_id": "s2", "figure_number": "f2",
             "content_type": "table", "text_raw": "bbb", "method": "structured_lookup"},
        ],
        [{"id": "X1", "method": "vector", "text_raw": "ccc"}],
    ]
    merged = rrf_merge(lists, top_k=10)
    assert len(merged) == 3


def test_rrf_merge_tags_missing_method_loudly():
    from src.pipeline.retriever import rrf_merge
    merged = rrf_merge([[{"id": "X", "text_raw": "t"}]], top_k=5)
    assert "missing_method" in merged[0]["contributing_methods"]


# ---------------------------------------------------------------------------
# reranker shape stability

def test_rerank_empty_query_preserves_shape():
    from src.pipeline.reranker import rerank
    out = rerank("", [{"method": "rrf", "text_raw": "hello"}])
    assert out[0]["method"] == "rerank"
    assert "rerank_score" in out[0] and out[0]["rerank_score"] is None
    assert out[0]["prior_method"] == "rrf"


def test_rerank_empty_results_returns_empty():
    from src.pipeline.reranker import rerank
    assert rerank("q", []) == []


# ---------------------------------------------------------------------------
# search guards

def test_is_empty_query_catches_punctuation_and_whitespace():
    from src.pipeline.search import _is_empty_query
    for bad in [None, "", "   ", "?!@#$%", "...!!!"]:
        assert _is_empty_query(bad), f"should be empty: {bad!r}"
    for good in ["hi", "CDW10", "a "]:
        assert not _is_empty_query(good), f"should be non-empty: {good!r}"


# ---------------------------------------------------------------------------
# orchestrator GenerationError carries cause + trace

def test_generation_error_carries_cause_and_trace():
    from src.pipeline.orchestrator import GenerationError
    cause = RuntimeError("boom")
    err = GenerationError("test", cause=cause, trace=[{"stage": "x"}],
                          retrieved_chunks=[{"id": "1"}])
    assert err.cause is cause
    assert err.trace == [{"stage": "x"}]
    assert err.retrieved_chunks == [{"id": "1"}]


# ---------------------------------------------------------------------------
# Agentic targeted-fetch: requested-resources parsing

def test_parse_requested_resources_strips_prefixes_and_dedupes():
    from src.pipeline.orchestrator import _parse_requested_resources
    out = _parse_requested_resources({"requested_resources": {
        "figures": ["630", 631, "Figure 634", "fig.635", "FIG. 636", "Fig 637",
                    "  ", None, "Figure 630"],  # dedupe last one
        "fields": ["PPI", "cdp", "PPI", "  ", "MQES"],
        "sections": ["Section 8.20.1", "§5.2", "SECTION 6.7", "appendix A.3", None],
    }})
    assert out["figures"] == ["630", "631", "634", "635", "636", "637"]
    assert out["fields"] == ["PPI", "CDP", "MQES"]
    assert out["sections"] == ["8.20.1", "5.2", "6.7", "A.3"]


def test_parse_requested_resources_defensive_on_garbage():
    from src.pipeline.orchestrator import _parse_requested_resources
    empty = {"figures": [], "fields": [], "sections": []}
    assert _parse_requested_resources(None) == empty
    assert _parse_requested_resources({}) == empty
    assert _parse_requested_resources({"requested_resources": "string"}) == empty
    assert _parse_requested_resources({"requested_resources": {"figures": "not-a-list"}}) == empty
    # Caps respected
    big = {"requested_resources": {"figures": [str(i) for i in range(50)]}}
    assert len(_parse_requested_resources(big)["figures"]) == 8


def test_pipeline_config_has_agentic_targeted_fetch_default():
    from src.pipeline.orchestrator import PipelineConfig
    cfg = PipelineConfig()
    assert cfg.agentic_targeted_fetch is True
    # Other agentic defaults still in place
    assert cfg.agentic_model == "claude-opus-4-7"
    assert cfg.agentic_max_context_tokens == 16000


# ---------------------------------------------------------------------------
# Allow `python tests/test_pipeline_units.py` (no pytest) to validate fast.

if __name__ == "__main__":
    import inspect

    failures: list[str] = []
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and inspect.isfunction(obj)
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failures.append(f"{name}: AssertionError: {e}")
            print(f"FAIL {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: {type(e).__name__}: {e}")
            print(f"ERROR {name}: {type(e).__name__}: {e}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(0 if not failures else 1)
