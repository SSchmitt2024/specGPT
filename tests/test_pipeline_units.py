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
        "content_type": "prose",
        # Provenance fields default to None/[] when the context chunk omits them.
        "spec": None, "spec_document": None, "pdf_pages": [],
        "snippet": "",
        "hallucinated": False,
    }]


def test_extract_citations_picks_up_block_attribution_line():
    """Tables/code blocks can't carry an end-of-sentence tag, so the model is
    instructed (rule 2b) to cite them on a trailing `Source: [§…]` line. That
    line is plain answer text, so the bracket parser must resolve it like any
    other tag — keeping the block's citation in the sidebar + chips."""
    from src.pipeline.generator import _extract_citations
    answer = (
        "Here is the CDW10 layout:\n\n"
        "| Bits | Field | Description |\n"
        "| --- | --- | --- |\n"
        "| 7:0 | OPC | Opcode |\n\n"
        "Source: [§5.2.1]\n"
    )
    cits = _extract_citations(
        answer,
        [{"section_id": "5.2.1", "section_title": "X", "content_type": "table"}],
    )
    ids = [c["section_id"] for c in cits]
    assert ids == ["5.2.1"], ids
    assert cits[0]["hallucinated"] is False


def test_extract_citations_flags_hallucinated_sections():
    from src.pipeline.generator import _extract_citations
    cits = _extract_citations(
        "Per Section 9.9.9 the spec says...",
        [{"section_id": "5.2.1", "section_title": "X", "content_type": "prose"}],
    )
    assert cits[0]["hallucinated"] is True


def test_extract_citations_resolves_title_only_pages():
    """Spec 'pages' often carry no numeric section_id (e.g. "Persistent Event
    Log Page"); the context header renders them as "[i] § <title>" so the model
    cites them by title. Those must resolve to the chunk, not be flagged
    hallucinated (the amber-dot bug)."""
    from src.pipeline.generator import _extract_citations
    ctx = [{
        "section_id": "",
        "section_title": "Persistent Event Log Page",
        "content_type": "prose",
        "spec": "base",
        "pdf_pages": [42],
    }]
    cits = _extract_citations(
        "LREV shall be set to 03h [§Persistent Event Log Page]. It sits at "
        "byte 16 [§Persistent Event Log Page].",
        ctx,
    )
    assert len(cits) == 1, cits
    assert cits[0]["hallucinated"] is False
    assert cits[0]["section_title"] == "Persistent Event Log Page"


def test_extract_citations_title_with_internal_comma_is_one_citation():
    """A §-title that contains a comma (e.g. the Identify Controller table title)
    must NOT be split on the comma into two hallucinated halves - it resolves as
    a single citation with the chunk's page."""
    from src.pipeline.generator import _extract_citations
    title = "Identify – Identify Controller Data Structure, I/O Command Set Independent"
    ctx = [{
        "section_id": "",
        "section_title": title,
        "content_type": "table",
        "spec": "base",
        "pdf_pages": [344],
    }]
    cits = _extract_citations(f"OACS lives here [§{title}].", ctx)
    assert len(cits) == 1, cits
    assert cits[0]["hallucinated"] is False
    assert cits[0]["section_id"] == title
    assert cits[0]["pdf_pages"] == [344]


def test_extract_citations_mixed_bracket_id_and_comma_title():
    """A bracket mixing a section id with a comma-containing title (split on §,
    not comma) must yield two clean citations, not the title torn in half."""
    from src.pipeline.generator import _extract_citations
    title = "Identify – Identify Controller Data Structure, I/O Command Set Independent"
    ctx = [
        {"section_id": "", "section_title": title, "content_type": "table",
         "spec": "base", "pdf_pages": [344]},
        {"section_id": "8.1.16", "section_title": "NM", "content_type": "prose",
         "pdf_pages": [604]},
    ]
    cits = _extract_citations(f"Support indicated [§8.1.16, §{title}].", ctx)
    assert len(cits) == 2, cits
    assert all(c["hallucinated"] is False for c in cits)
    assert {c["section_id"] for c in cits} == {"8.1.16", title}


def test_extract_citations_figure_token_not_a_section_citation():
    """A '[§Figure 328]' bracket must not produce a hallucinated section
    citation - figures are surfaced via the figures payload instead."""
    from src.pipeline.generator import _extract_citations
    ctx = [{"section_id": "5.2", "section_title": "X", "content_type": "prose",
            "pdf_pages": [1]}]
    cits = _extract_citations("See [§Figure 328] and [§5.2].", ctx)
    assert [c["section_id"] for c in cits] == ["5.2"]
    assert cits[0]["hallucinated"] is False


def test_extract_citations_comma_separated_ids_still_split():
    """Regression guard for the comma-title fix: a real comma-separated id list
    must still produce one citation per id."""
    from src.pipeline.generator import _extract_citations
    ctx = [
        {"section_id": "8.1.5", "section_title": "A", "content_type": "prose", "pdf_pages": [1]},
        {"section_id": "8.2.6", "section_title": "B", "content_type": "prose", "pdf_pages": [2]},
    ]
    cits = _extract_citations("See [§8.1.5, §8.2.6] for details.", ctx)
    assert sorted(c["section_id"] for c in cits) == ["8.1.5", "8.2.6"]
    assert all(c["hallucinated"] is False for c in cits)


def test_extract_citations_salvages_id_from_prose_bracket():
    """Flagged-answer bug: the model wrote prose inside the bracket
    ("[§5.2.12.1 is not in context, but ...]") and the whole sentence became
    the sidebar 'citation'. The parser must salvage just the leading id."""
    from src.pipeline.generator import _extract_citations
    cits = _extract_citations(
        "[§5.2.12.1 is not in context, but the log page identity is defined "
        "in the support requirements table below]",
        [{"section_id": "2.4.1", "section_title": "X", "content_type": "prose"}],
    )
    assert len(cits) == 1, cits
    assert cits[0]["section_id"] == "5.2.12.1"
    assert cits[0]["hallucinated"] is True


def test_extract_citations_drops_long_prose_bracket_without_id():
    """A §-bracket holding a long sentence with no leading section id is prose,
    not a citation - it must not reach the sidebar at all."""
    from src.pipeline.generator import _extract_citations
    cits = _extract_citations(
        "[§this detail is not present in the provided context so no precise "
        "definition or table can be given for the requested field here]",
        [{"section_id": "2.4.1", "section_title": "X", "content_type": "prose"}],
    )
    assert cits == [], cits


def test_extract_citations_records_cited_as_on_near_miss_resolution():
    """When 'section 5.3' in prose resolves to the in-context descendant
    5.3.2.1, the citation must carry cited_as='5.3' so the UI can tie the
    inline text the user sees to the sidebar source."""
    from src.pipeline.generator import _extract_citations
    cits = _extract_citations(
        "That requirement applies only in contexts defined in section 5.3, "
        "which is not fully detailed here.",
        [{"section_id": "5.3.2.1", "section_title": "PI and Write Commands",
          "content_type": "prose", "pdf_pages": [139]}],
    )
    assert len(cits) == 1, cits
    assert cits[0]["section_id"] == "5.3.2.1"
    assert cits[0]["cited_as"] == "5.3"
    assert cits[0]["hallucinated"] is False


def test_extract_citations_exact_match_has_no_cited_as():
    from src.pipeline.generator import _extract_citations
    cits = _extract_citations(
        "Defined in [§5.2.1].",
        [{"section_id": "5.2.1", "section_title": "X", "content_type": "prose"}],
    )
    assert len(cits) == 1
    assert "cited_as" not in cits[0]


def test_extract_citations_title_match_is_case_and_space_insensitive():
    from src.pipeline.generator import _extract_citations
    ctx = [{"section_id": "5.2", "section_title": "Get Log Page",
            "content_type": "prose"}]
    # Cited with different casing/whitespace; resolves to the real numeric id.
    cits = _extract_citations("Details here [§get  log page].", ctx)
    assert len(cits) == 1 and cits[0]["hallucinated"] is False
    assert cits[0]["section_id"] == "5.2"


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
# retriever value-keyed enumeration lookup (FID/opcode/log-page/CNS/status)

def test_value_tokens_parses_fid_and_hex_entities():
    from src.pipeline.retriever import _value_tokens
    ents = [{"text": "FID 17h", "kind": "fid"}, {"text": "0x0D", "kind": "hex"}]
    assert _value_tokens(ents) == {0x17, 0x0D}


def test_cell_value_only_parses_bare_hex_tokens():
    from src.pipeline.retriever import _cell_value
    assert _cell_value("17h") == 0x17
    assert _cell_value("0x17") == 0x17
    assert _cell_value("5.2.26.1.15") is None  # section ref, not a value
    assert _cell_value("Sanitize Config") is None


def test_enum_value_match_resolves_fid_to_feature_name():
    """FID 17h must resolve to its Feature Identifiers row without an LLM."""
    from src.pipeline.retriever import _enum_value_matches
    tables = {
        "198": {
            "figure_number": "198",
            "caption": "Get Features – Feature Identifiers",
            "rows": [
                ["Power Management", "02h", "5.2.26.1.2"],
                ["Sanitize Config", "17h", "5.2.26.1.15"],
            ],
        },
        # Same value lives in an unrelated table; the concept gate must exclude it.
        "206": {
            "figure_number": "206",
            "caption": "Get Log Page – Log Page Identifiers",
            "rows": [["17h", "N", "Controller", "Some Log", "5.x"]],
        },
    }
    ents = [{"text": "FID 17h", "kind": "fid"}]
    matches = _enum_value_matches(ents, "Which feature corresponds to FID 17h?", tables)
    figs = {m["parent_figure"] for m in matches}
    names = {m["field_name"] for m in matches}
    assert figs == {"198"}                 # log-page table excluded by concept gate
    assert "Sanitize Config" in names


def test_enum_value_match_noop_without_value_entity():
    from src.pipeline.retriever import _enum_value_matches
    tables = {"198": {"figure_number": "198",
                      "caption": "Get Features – Feature Identifiers",
                      "rows": [["Sanitize Config", "17h", "5.2.26.1.15"]]}}
    # bit-range style query carries no fid/hex entity → must not match.
    assert _enum_value_matches([], "What are bits 7:4 of CDW10?", tables) == []


# ---------------------------------------------------------------------------
# enum_tables extractor + deterministic keyed enum index

_FID_TABLES_FIXTURE = [
    {
        "figure_number": 198,
        "caption": "Get Features – Feature Identifiers",
        "rows": [
            ["Arbitration", "01h", "5.2.26.1.1"],
            ["Configurable Device Personality", "22h", "5.2.26.1.24"],
            ["Attributes Returned"],  # broken PDF row, no value → skipped
        ],
    },
    {
        "figure_number": 403,
        "caption": "Set Features – Feature Identifiers",
        "rows": [
            ["01h", "No", "No", "Arbitration", "Controller"],
            ["22h", "Yes", "Yes", "Configurable Device Personality", "NVM subsystem"],
        ],
    },
    {
        "figure_number": 206,
        "caption": "Get Log Page – Log Page Identifiers",
        "rows": [["22h", "N", "Controller", "Endurance Group", "5.2.12.1.31"]],
    },
    # Near-miss caption must be excluded by the concept's exclude_re.
    {"figure_number": 999, "caption": "Feature Identifiers Effects Log Page",
     "rows": [["22h", "Bogus Feature"]]},
]


def test_enum_tables_extracts_fid_and_merges_figures():
    from src.enum_tables import build_enum_index
    index = build_enum_index(_FID_TABLES_FIXTURE)
    fid_entries = {e["value"]: e for e in index["fid"]["entries"]}
    cdp = fid_entries[0x22]
    assert cdp["name"] == "Configurable Device Personality"
    assert cdp["value_hex"] == "22h"
    # Same FID listed in both Get and Set tables → figures merged, not duplicated.
    assert set(cdp["figures"]) == {"198", "403"}
    assert cdp["sections"] == ["5.2.26.1.24"]
    # The "Effects Log Page" near-miss must not pollute the fid concept.
    assert all("Bogus" not in e["name"] for e in index["fid"]["entries"])


def test_enum_tables_extracts_lid_separately():
    from src.enum_tables import build_enum_index
    index = build_enum_index(_FID_TABLES_FIXTURE)
    lid_entries = {e["value"]: e for e in index["lid"]["entries"]}
    assert lid_entries[0x22]["name"] == "Endurance Group"


def test_enum_index_hits_fid_22_always_hex():
    """'FID 22' and 'FID 22h' must BOTH resolve to 0x22 (always hex)."""
    from src.enum_tables import build_enum_index
    from src.pipeline.retriever import _enum_index_hits
    from src.pipeline.query_processor import extract_entities

    index = build_enum_index(_FID_TABLES_FIXTURE)
    for query in ["what is FID 22", "what is FID 22h", "what feature is FID 0x22"]:
        ents = [{"text": e.text, "kind": e.kind} for e in extract_entities(query)]
        hits = _enum_index_hits(ents, query, index)
        names = {h["name"] for h in hits if h["concept"] == "fid"}
        assert "Configurable Device Personality" in names, query
        # Must be the hex 0x22 entry, never decimal 22 (= 0x16).
        assert all(h["value"] == 0x22 for h in hits if h["concept"] == "fid"), query


def test_enum_index_hits_decimal_is_hex_not_literal():
    """'FID 22' resolves to 0x22, distinct from the entry at decimal-looking 16h."""
    from src.enum_tables import build_enum_index
    from src.pipeline.retriever import _enum_index_hits
    tables = [{
        "figure_number": 198,
        "caption": "Get Features – Feature Identifiers",
        "rows": [["Host Behavior Support", "16h", "5.2.26.1.22"],
                 ["Configurable Device Personality", "22h", "5.2.26.1.24"]],
    }]
    index = build_enum_index(tables)
    hits = _enum_index_hits([{"text": "FID 22", "kind": "fid"}], "FID 22", index)
    names = {h["name"] for h in hits}
    assert names == {"Configurable Device Personality"}  # 0x22, not "the 22nd / 16h"


def test_enum_index_lid_value_resolves():
    from src.enum_tables import build_enum_index
    from src.pipeline.retriever import _enum_index_hits
    index = build_enum_index(_FID_TABLES_FIXTURE)
    hits = _enum_index_hits([{"text": "LID 22h", "kind": "lid"}],
                            "what log page is LID 22h", index)
    names = {h["name"] for h in hits if h["concept"] == "lid"}
    assert "Endurance Group" in names


def test_enum_index_hits_empty_without_index():
    from src.pipeline.retriever import _enum_index_hits
    assert _enum_index_hits([{"text": "FID 22", "kind": "fid"}], "FID 22", {}) == []


def test_lid_entity_extracted_with_and_without_h():
    from src.pipeline.query_processor import extract_entities
    for q in ["what is LID 22", "what is LID 22h", "Log Page Identifier 0x02"]:
        kinds = {e.kind for e in extract_entities(q)}
        assert "lid" in kinds, q


def test_value_tokens_parses_lid_entity_as_hex():
    from src.pipeline.retriever import _value_tokens
    assert _value_tokens([{"text": "LID 22", "kind": "lid"}]) == {0x22}
    assert _value_tokens([{"text": "LID 22h", "kind": "lid"}]) == {0x22}


# --- value-keyed lookup must depend on the hex value, not how it is typed -----
# Regression for: "FID 2" / "opcode 2" / "status code 6" silently failing while
# the "0x.." / "..h" / leading-zero spellings worked. Every form of a value must
# resolve to the same hexadecimal magnitude across all enum concepts.

def test_fid_extracted_for_every_value_spelling():
    from src.pipeline.query_processor import extract_entities
    for q in ["what is FID 2", "what is FID 02", "what is FID 2h", "what is FID 0x2"]:
        kinds = {e.kind for e in extract_entities(q)}
        assert "fid" in kinds, q


def test_value_tokens_single_digit_fid_is_hex():
    from src.pipeline.retriever import _value_tokens
    # "FID 2", "FID 02", "FID 2h" must all be the same value (0x02) — not decimal,
    # not dependent on padding or the trailing 'h'.
    assert _value_tokens([{"text": "FID 2", "kind": "fid"}]) == {0x02}
    assert _value_tokens([{"text": "FID 02", "kind": "fid"}]) == {0x02}
    assert _value_tokens([{"text": "FID 2h", "kind": "fid"}]) == {0x02}


def test_value_tokens_letter_leading_fid_lid_resolve_by_hex():
    """A FID/LID whose value leads with a hex letter (a-f), e.g. the vendor-
    specific 0xC0 range or 0x0B, must resolve regardless of spelling — not be
    dropped or truncated to a digit suffix."""
    from src.pipeline.retriever import _value_tokens
    for text, value in [
        ("FID b", 0x0B), ("FID 0b", 0x0B), ("FID 0Bh", 0x0B),
        ("FID c0", 0xC0), ("FID C0", 0xC0), ("FID c0h", 0xC0), ("FID 0xc0", 0xC0),
        ("FID ff", 0xFF),
    ]:
        assert _value_tokens([{"text": text, "kind": "fid"}]) == {value}, text
    for text, value in [("LID a", 0x0A), ("LID d0", 0xD0), ("LID 0xd0", 0xD0)]:
        assert _value_tokens([{"text": text, "kind": "lid"}]) == {value}, text


def test_value_tokens_no_space_keyword_does_not_bleed_into_value():
    """A keyword whose final letter is a hex digit (FID→D, LID→D) must not merge
    into a no-space value: "FID2" is 0x02, never "D2" (0xD2)."""
    from src.pipeline.retriever import _value_tokens
    assert _value_tokens([{"text": "FID2", "kind": "fid"}]) == {0x02}
    assert _value_tokens([{"text": "FIDc0", "kind": "fid"}]) == {0xC0}
    assert _value_tokens([{"text": "LIDd", "kind": "lid"}]) == {0x0D}


def test_opcode_cns_status_bare_values_extracted_as_hex():
    """opcode / CNS / status code values must resolve from a bare number, not
    only the 0x-prefixed or ..h spellings the generic hex pattern caught."""
    from src.pipeline.query_processor import extract_entities
    from src.pipeline.retriever import _value_tokens
    cases = [
        ("what is opcode 2", "opcode", 0x02),
        ("what is opcode 02", "opcode", 0x02),
        ("what is opcode 2h", "opcode", 0x02),
        ("what command is opcode 0Dh", "opcode", 0x0D),
        ("what is CNS 1", "cns", 0x01),
        ("what is status code 6", "status", 0x06),
        ("what is status 6", "status", 0x06),
    ]
    for query, kind, value in cases:
        ents = [{"text": e.text, "kind": e.kind} for e in extract_entities(query)]
        assert any(e["kind"] == kind for e in ents), query
        assert _value_tokens(ents) == {value}, query


def test_enum_concept_keywords_do_not_false_match_english():
    """The opcode/CNS/status keywords are ordinary words; a value entity must
    not be invented when no actual value follows them."""
    from src.pipeline.query_processor import extract_entities
    from src.pipeline.retriever import _value_tokens
    for q in ["status of the controller", "opcode for the read command",
              "what does the status field mean"]:
        ents = [{"text": e.text, "kind": e.kind} for e in extract_entities(q)]
        assert _value_tokens(ents) == set(), q


def test_enum_index_hits_opcode_bare_value_resolves():
    from src.enum_tables import build_enum_index
    from src.pipeline.retriever import _enum_index_hits
    from src.pipeline.query_processor import extract_entities
    tables = [{
        "figure_number": 140,
        "caption": "Opcodes for Admin Commands",
        "rows": [["02h", "Get Log Page", "5.x"], ["09h", "Set Features", "5.y"]],
    }]
    index = build_enum_index(tables)
    for query in ["what is opcode 2", "what is opcode 02", "what is opcode 2h"]:
        ents = [{"text": e.text, "kind": e.kind} for e in extract_entities(query)]
        hits = _enum_index_hits(ents, query, index)
        names = {h["name"] for h in hits if h["concept"] == "opcode"}
        assert "Get Log Page" in names, query
        assert all(h["value"] == 0x02 for h in hits if h["concept"] == "opcode"), query


def test_enum_hit_to_source_is_self_contained():
    from src.pipeline.retriever import _enum_hit_to_source
    hit = {"concept": "fid", "label": "Feature Identifier", "value": 0x22,
           "value_hex": "22h", "name": "Configurable Device Personality",
           "figures": ["198", "403"], "sections": ["5.2.26.1.24"]}
    src = _enum_hit_to_source(hit)
    assert "Configurable Device Personality" in src["text_raw"]
    assert "22h" in src["text_raw"]
    assert src["method"] == "structured_lookup"


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
# Structured-lookup pinning survives reranking (regression: "what feature is
# fid 2" → Power Management got dropped at the top_k cut)

def test_pin_structured_hits_keeps_low_scoring_structured_hit():
    from src.pipeline.orchestrator import _pin_structured_hits
    # Pre-rerank pool: one authoritative structured hit + three hybrid hits.
    pre = [
        {"chunk_id": "enum:fid:02h:Power_Management", "method": "structured_lookup"},
        {"chunk_id": "fig266", "method": "rrf"},
        {"chunk_id": "fig267", "method": "rrf"},
        {"chunk_id": "fig401", "method": "rrf"},
    ]
    # Reranker scored the structured hit LOWEST and ordered it last.
    ranked = [
        {"chunk_id": "fig266", "prior_method": "rrf", "rerank_score": 0.62},
        {"chunk_id": "fig267", "prior_method": "rrf", "rerank_score": 0.60},
        {"chunk_id": "fig401", "prior_method": "rrf", "rerank_score": 0.55},
        {"chunk_id": "enum:fid:02h:Power_Management",
         "prior_method": "structured_lookup", "rerank_score": 0.10},
    ]
    out = _pin_structured_hits(ranked, pre, budget=2)
    # Structured hit is pinned first and never truncated, even with budget=2.
    assert out[0]["chunk_id"] == "enum:fid:02h:Power_Management"
    assert len(out) == 2  # pinned + top-1 semantic
    assert out[1]["chunk_id"] == "fig266"


def test_pin_structured_hits_preserves_structured_order():
    from src.pipeline.orchestrator import _pin_structured_hits
    pre = [
        {"chunk_id": "s1", "method": "structured_lookup"},
        {"chunk_id": "s2", "method": "structured_lookup"},
        {"chunk_id": "h1", "method": "rrf"},
    ]
    # Reranker shuffled the two structured hits (s2 above s1).
    ranked = [
        {"chunk_id": "h1", "prior_method": "rrf", "rerank_score": 0.9},
        {"chunk_id": "s2", "prior_method": "structured_lookup", "rerank_score": 0.4},
        {"chunk_id": "s1", "prior_method": "structured_lookup", "rerank_score": 0.2},
    ]
    out = _pin_structured_hits(ranked, pre, budget=5)
    # Original structured order (s1 before s2) is restored ahead of hybrid.
    assert [c["chunk_id"] for c in out] == ["s1", "s2", "h1"]


def test_pin_structured_hits_noop_without_structured():
    from src.pipeline.orchestrator import _pin_structured_hits
    pre = [{"chunk_id": "h1", "method": "rrf"}, {"chunk_id": "h2", "method": "rrf"}]
    ranked = [
        {"chunk_id": "h1", "prior_method": "rrf", "rerank_score": 0.9},
        {"chunk_id": "h2", "prior_method": "rrf", "rerank_score": 0.5},
    ]
    out = _pin_structured_hits(ranked, pre, budget=1)
    assert [c["chunk_id"] for c in out] == ["h1"]


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


def test_model_supports_temperature_excludes_newer_opus():
    from src.pipeline.generator import _model_supports_temperature
    # Newer Opus reasoning models reject the `temperature` param (400).
    assert _model_supports_temperature("claude-opus-4-7") is False
    assert _model_supports_temperature("claude-opus-4-8") is False
    # Sonnet/Haiku still accept it.
    assert _model_supports_temperature("claude-sonnet-4-6") is True
    assert _model_supports_temperature("claude-haiku-4-5-20251001") is True


def test_call_with_retry_omits_temperature_for_deprecated_models():
    from src.pipeline import generator

    class _FakeClient:
        def __init__(self):
            self.messages = self
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return object()  # opaque stub; _call_with_retry doesn't inspect it

    # Opus: temperature must NOT be sent, or the API 400s.
    opus = _FakeClient()
    generator._call_with_retry(
        opus, model="claude-opus-4-7", system="sys",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=16, timeout=1.0, max_retries=1,
    )
    assert "temperature" not in opus.calls[0]

    # Sonnet: temperature is still pinned to 0.0 for determinism.
    sonnet = _FakeClient()
    generator._call_with_retry(
        sonnet, model="claude-sonnet-4-6", system="sys",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=16, timeout=1.0, max_retries=1,
    )
    assert sonnet.calls[0].get("temperature") == 0.0


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
# retriever fuzzy full-name fallback (runs *after* the exact lookup tables)

# Synthetic field_index: acronym -> [record]. Two names share words so we can
# prove acronyms never cross over and a descriptive phrase resolves cleanly.
_FUZZY_FIELD_INDEX = {
    "MPTR": [{"field_name": "MPTR", "full_name": "Metadata Pointer", "parent_figure": 22}],
    "DPTR": [{"field_name": "DPTR", "full_name": "Data Pointer", "parent_figure": 23}],
    "CRTO": [{"field_name": "CRTO", "full_name": "Controller Ready Timeout", "parent_figure": 30}],
    # Single-word name: must be excluded from the fuzzy index entirely.
    "NSID": [{"field_name": "NSID", "full_name": "Namespace", "parent_figure": 40}],
}


def _fuzzy_name_index():
    from src.pipeline.retriever import _RE_WORD
    by_name: dict[str, set[str]] = {}
    for records in _FUZZY_FIELD_INDEX.values():
        for rec in records:
            norm = " ".join(_RE_WORD.findall(rec["full_name"].lower()))
            if " " not in norm:
                continue
            by_name.setdefault(norm, set()).add(rec["field_name"])
    return tuple((n, tuple(sorted(a))) for n, a in by_name.items())


def test_fuzzy_full_name_resolves_descriptive_phrase():
    """A paraphrased multi-word name reaches its field; hits are tagged."""
    from src.pipeline.retriever import _fuzzy_full_name_matches
    recs, notes = _fuzzy_full_name_matches(
        "what is the controller ready timeout",
        _FUZZY_FIELD_INDEX, _fuzzy_name_index(), cutoff=0.86, max_hits=8,
    )
    assert "CRTO" in {r["field_name"] for r in recs}
    assert all(r["source"] == "fuzzy_full_name" for r in recs)
    assert all("fuzzy_score" in r for r in recs)
    assert notes


def test_fuzzy_full_name_never_crosses_acronyms():
    """An acronym near-miss must never resolve to a *different* acronym.

    CRATT is not a known field; it must NOT fuzzily collapse to CRTO (or any
    acronym). Acronyms stay exact — only descriptive full names are fuzzed.
    """
    from src.pipeline.retriever import _fuzzy_full_name_matches
    for token in ("CRATT", "MPTRR", "DPTRX"):
        recs, _ = _fuzzy_full_name_matches(
            token, _FUZZY_FIELD_INDEX, _fuzzy_name_index(), cutoff=0.86, max_hits=8,
        )
        assert recs == [], f"{token} must not fuzzy-match any acronym"


def test_fuzzy_full_name_excludes_single_word_names():
    """Single-word names (e.g. 'Namespace') are not eligible for fuzzy matching,
    so a one-token query can never collapse onto them."""
    from src.pipeline.retriever import _fuzzy_full_name_matches
    recs, _ = _fuzzy_full_name_matches(
        "namespac", _FUZZY_FIELD_INDEX, _fuzzy_name_index(), cutoff=0.86, max_hits=8,
    )
    assert "NSID" not in {r["field_name"] for r in recs}


def test_fuzzy_full_name_empty_query_is_noop():
    from src.pipeline.retriever import _fuzzy_full_name_matches
    recs, notes = _fuzzy_full_name_matches(
        "what is the of", _FUZZY_FIELD_INDEX, _fuzzy_name_index(), cutoff=0.86, max_hits=8,
    )
    assert recs == [] and notes == []


# ---------------------------------------------------------------------------
# qa_log row mapping (every Q&A is recorded, not just flagged ones)

def test_qa_log_row_maps_response_and_denormalizes_config():
    """Every answered query is logged to qa_log. The row builder is pure, so we
    can assert the field mapping without touching Supabase, and confirm spec /
    llm_model are pulled out of config for easy querying."""
    import os
    # app.py bootstraps auth at import; give it the minimum so import succeeds.
    os.environ.setdefault("APP_PASSWORD", "test")
    os.environ.setdefault("SESSION_SECRET", "x" * 32)
    from src.pipeline.app import _qa_log_row, QueryResponse

    resp = QueryResponse(
        query="What is LBA?",
        answer="A logical block address.",
        citations=[{"section_id": "5.2", "hallucinated": False}],
        config={"spec": "base", "llm_model": "claude-opus-4-8"},
        latency_ms=1234.5,
        tokens_used={"input": 10, "output": 20},
        agentic=True,
    )
    row = _qa_log_row(resp, "req123abc")

    assert row["request_id"] == "req123abc"
    assert row["query"] == "What is LBA?"
    assert row["answer"] == "A logical block address."
    assert row["citations"] == [{"section_id": "5.2", "hallucinated": False}]
    assert row["spec"] == "base"
    assert row["llm_model"] == "claude-opus-4-8"
    assert row["agentic"] is True
    assert row["latency_ms"] == 1234.5
    assert row["tokens_used"] == {"input": 10, "output": 20}


def test_qa_log_row_tolerates_missing_config_keys():
    import os
    os.environ.setdefault("APP_PASSWORD", "test")
    os.environ.setdefault("SESSION_SECRET", "x" * 32)
    from src.pipeline.app import _qa_log_row, QueryResponse

    resp = QueryResponse(
        query="q", answer="a", citations=[], config={}, latency_ms=1.0,
    )
    row = _qa_log_row(resp, "rid")
    assert row["spec"] is None and row["llm_model"] is None
    assert row["agentic"] is False and row["tokens_used"] is None


# ---------------------------------------------------------------------------
# Completeness verdict: the agentic loop's stop signal must parse cleanly and,
# above all, never leak the sentinel into the user-facing answer.

def test_split_verdict_parses_and_strips_sentinel():
    from src.pipeline.generator import _split_verdict
    ans, verdict = _split_verdict(
        'The PRACT bit is ignored. [§3.3.5]\n'
        '@@VERDICT@@{"answered": true, "context_has_answer": true, "missing": ""}'
    )
    assert ans == "The PRACT bit is ignored. [§3.3.5]"  # sentinel stripped
    assert "@@VERDICT@@" not in ans
    assert verdict == {"answered": True, "context_has_answer": True, "missing": ""}


def test_split_verdict_absent_marker_returns_answer_unchanged():
    from src.pipeline.generator import _split_verdict
    text = "A normal answer with no verdict line. [§5.2]"
    ans, verdict = _split_verdict(text)
    assert ans == text and verdict is None


def test_split_verdict_malformed_json_never_corrupts_answer():
    """A truncated/garbled verdict must degrade to (clean_answer, None) — the
    sentinel is still stripped so the user never sees it, but no exception and
    no partial answer loss."""
    from src.pipeline.generator import _split_verdict
    ans, verdict = _split_verdict("Body of the answer.\n@@VERDICT@@ {oops not json")
    assert ans == "Body of the answer."
    assert "@@VERDICT@@" not in ans
    assert verdict is None


def test_split_verdict_defaults_context_has_answer_to_answered():
    from src.pipeline.generator import _split_verdict
    _, verdict = _split_verdict('x\n@@VERDICT@@{"answered": false, "missing": "Figure 11"}')
    assert verdict["answered"] is False
    assert verdict["context_has_answer"] is False  # defaulted from answered
    assert verdict["missing"] == "Figure 11"


# ---------------------------------------------------------------------------
# Context header: section-less chunks must not emit an empty "[Section ]"

def test_assemble_context_section_less_chunk_uses_title_not_empty():
    """Figure/table chunks from structured lookups have no section_id. The
    header must fall back to the title so the model cites "[§<title>]" instead
    of an empty "[§]" that breaks the sidebar and poisons citation brackets."""
    from src.pipeline.generator import assemble_context
    ctx = [{
        "section_id": "",
        "section_title": "Optional Admin Command Support",
        "content_type": "table",
        "text_raw": "Bit 0 SSRS: Security Send/Receive Supported.",
        "pdf_pages": [256],
    }]
    formatted, _ = assemble_context("what is oacs", ctx)
    assert "[Section ]" not in formatted          # the bug
    assert "[Section Optional Admin Command Support]" in formatted


def test_assemble_context_numberless_figure_chunk_uses_figure_header():
    """A figure/table chunk with no section number must present a stable
    "[Figure N]" header so the model cites [Figure N] rather than inventing a
    section number from memory (which lands as a hallucinated citation)."""
    from src.pipeline.generator import assemble_context
    ctx = [{
        "section_id": "",
        "section_title": "Identify – Identify Controller Data Structure, I/O Command Set Independent",
        "figure_number": "328",
        "content_type": "table",
        "text_raw": "Bytes 257:256 Optional Admin Command Support (OACS).",
        "pdf_pages": [344],
    }]
    formatted, _ = assemble_context("what is oacs", ctx)
    assert "[Figure 328]" in formatted
    assert "[Section ]" not in formatted


def test_assemble_context_numbered_chunk_keeps_normal_header():
    from src.pipeline.generator import assemble_context
    ctx = [{
        "section_id": "8.1.5",
        "section_title": "Command and Feature Lockdown",
        "content_type": "prose",
        "text_raw": "The Lockdown command ...",
    }]
    formatted, _ = assemble_context("lockdown", ctx)
    assert "[Section 8.1.5] Command and Feature Lockdown" in formatted


# ---------------------------------------------------------------------------
# Citation deep-link backfill (pages/spec for structured-lookup citations)

def test_backfill_citation_pages_sets_spec_without_db_when_pages_present():
    """When every live citation already has pdf_pages, no DB lookup is needed;
    the backfill only stamps the (single-spec) spec on those missing it, and
    never touches hallucinated citations. Runs offline (no Supabase)."""
    from src.pipeline.orchestrator import _backfill_citation_pages
    cits = [
        {"section_id": "5.2", "pdf_pages": [10], "spec": None, "hallucinated": False},
        {"section_id": "5.3", "pdf_pages": [11], "spec": "base", "hallucinated": False},
        {"section_id": "9.9", "pdf_pages": [], "spec": None, "hallucinated": True},
    ]
    _backfill_citation_pages(cits, "base")
    assert cits[0]["spec"] == "base" and cits[0]["pdf_pages"] == [10]
    assert cits[1]["spec"] == "base"          # already set, unchanged
    assert cits[2]["spec"] is None and cits[2]["pdf_pages"] == []  # hallucinated untouched


def test_backfill_citation_pages_noop_on_empty_or_no_spec():
    from src.pipeline.orchestrator import _backfill_citation_pages
    _backfill_citation_pages([], "base")  # must not raise
    # No spec → can't look up pages, but also must not raise or query.
    cits = [{"section_id": "5.2", "pdf_pages": [], "spec": None, "hallucinated": False}]
    _backfill_citation_pages(cits, "")
    assert cits[0]["pdf_pages"] == []


# ---------------------------------------------------------------------------
# All-specs mode ("all" sentinel searches every corpus at once)

def test_backfill_citation_pages_all_specs_never_stamps_sentinel():
    """In all-specs mode a citation's spec must come from per-chunk provenance
    (or the section lookup), never the "all" sentinel, which has no PDF URL
    and would break the frontend deep link. Runs offline: every live citation
    already has pages so no DB lookup happens."""
    from src.pipeline.orchestrator import ALL_SPECS, _backfill_citation_pages
    cits = [
        {"section_id": "5.2", "pdf_pages": [10], "spec": None, "hallucinated": False},
        {"section_id": "2.1", "pdf_pages": [44], "spec": "pcie", "hallucinated": False},
    ]
    _backfill_citation_pages(cits, ALL_SPECS)
    assert cits[0]["spec"] is None      # left unset, not "all"
    assert cits[1]["spec"] == "pcie"    # per-chunk provenance preserved


def test_resolve_requested_resources_all_mode_probes_each_corpus(monkeypatch):
    """All-specs targeted fetch probes every corpus's figure index and stamps
    each chunk with its source spec. Figure numbers collide across specs, so
    the two "Figure 11"s must come back as distinct chunks (distinct ids) with
    distinct spec provenance."""
    from src.pipeline import orchestrator as orch

    tables = {
        "base":    {"11": {"figure_number": "11", "parent_section": "1.1",
                           "raw_text": "base table", "caption": "Base Fig 11"}},
        "pcie":    {"11": {"figure_number": "11", "parent_section": "2.2",
                           "raw_text": "pcie table", "caption": "PCIe Fig 11"}},
        "command": {},
    }
    monkeypatch.setattr(orch.retriever, "load_tables_by_figure", lambda spec: tables[spec])
    monkeypatch.setattr(orch.retriever, "load_field_index", lambda spec: {})

    out = orch._resolve_requested_resources(
        {"figures": ["11"], "fields": [], "sections": []},
        spec=orch.ALL_SPECS,
    )
    assert sorted(c["spec"] for c in out) == ["base", "pcie"]
    assert len({c["id"] for c in out}) == 2  # spec-prefixed ids stay distinct


def test_structured_lookup_all_specs_merges_and_stamps_spec(monkeypatch):
    """The all-specs structured lookup merges per-corpus results: found/
    confidence aggregate across specs, sources get spec provenance plus
    spec-prefixed ids (so dedup can't collapse colliding figure numbers), and
    notes say which corpus they came from."""
    from src.pipeline import orchestrator as orch
    from src.pipeline.retriever import StructuredLookupResult

    def fake_lookup(decomp, *, use_llm, max_fields, spec, enable_fuzzy, fuzzy_cutoff):
        if spec == "pcie":
            return StructuredLookupResult(
                query="q", found=True, confidence="HIGH",
                fields=[{"name": "X"}], tables=[{"figure_number": "11"}],
                sources=[{"chunk_id": "table:11", "score": 1.0,
                          "method": "structured_lookup"}],
                notes=["hit"],
            )
        return StructuredLookupResult(query="q", found=False, confidence="LOW",
                                      notes=["miss"])

    monkeypatch.setattr(orch.retriever, "structured_lookup", fake_lookup)
    res = orch._structured_lookup_all_specs("ignored-decomp")
    assert res.found and res.confidence == "HIGH"
    assert res.fields == [{"name": "X", "spec": "pcie"}]
    assert res.sources[0]["spec"] == "pcie"
    assert res.sources[0]["chunk_id"] == "pcie:table:11"
    assert "[base] miss" in res.notes and "[pcie] hit" in res.notes


def test_all_specs_option_registered():
    """The "all" sentinel must be selectable (validated spec id) and must stay
    LAST in AVAILABLE_SPECS so the frontend's _specData[0] fallback for
    spec-less citations remains the base spec."""
    from src.pipeline import app as app_mod
    from src.pipeline.orchestrator import ALL_SPECS
    assert ALL_SPECS in app_mod._VALID_SPEC_IDS
    assert app_mod.AVAILABLE_SPECS[-1]["id"] == ALL_SPECS
    assert app_mod.AVAILABLE_SPECS[0]["id"] == "base"
    assert app_mod.AVAILABLE_SPECS[-1]["url"] is None


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


def test_hallucinated_section_ids_returns_only_fetchable_ids():
    """Only id-like hallucinated cites are fetchable by section; figure/title
    misses and resolved citations must be excluded (and order preserved)."""
    from src.pipeline.orchestrator import _hallucinated_section_ids
    cits = [
        {"section_id": "8.1.6.3.2", "hallucinated": True},
        {"section_id": "8.1.6.3.1.1", "hallucinated": True},
        {"section_id": "8.1.6.2", "hallucinated": False},      # resolved
        {"section_id": "Some Title Text", "hallucinated": True},  # not an id
        {"section_id": "A.2.1", "hallucinated": True},          # appendix ok
        {"section_id": "8.1.6.3.2", "hallucinated": True},      # dup
    ]
    assert _hallucinated_section_ids(cits) == ["8.1.6.3.2", "8.1.6.3.1.1", "A.2.1"]
    assert _hallucinated_section_ids(None) == []
