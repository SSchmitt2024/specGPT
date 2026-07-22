"""Unit tests for the UNH-IOL test plan feature: parser helpers, the
system-prompt renderer, and the one-test-per-chat lock. No network.

    venv/bin/python3 -m pytest tests/test_testplans.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


# ---------------------------------------------------------------------------
# parser helpers (scripts/ingest_iol_testplans.py)

def test_parse_numbered_basic_and_multiline():
    from ingest_iol_testplans import _parse_numbered
    items = _parse_numbered("1. First step\ncontinued text\n2. Second step\n")
    assert [i["n"] for i in items] == [1, 2]
    assert items[0]["text"] == "First step continued text"


def test_parse_numbered_number_alone_on_line():
    from ingest_iol_testplans import _parse_numbered
    items = _parse_numbered("1. \nConfigure the host\n2. \nCheck the bit\n")
    assert [i["text"] for i in items] == ["Configure the host", "Check the bit"]


def test_parse_numbered_continuation_numbering():
    # Some cases continue the previous case's numbering (starts at 32).
    from ingest_iol_testplans import _parse_numbered
    items = _parse_numbered("32. Check version\n33. Check MDS bit\n")
    assert [i["n"] for i in items] == [32, 33]


def test_parse_numbered_unnumbered_prose_becomes_single_item():
    from ingest_iol_testplans import _parse_numbered
    items = _parse_numbered("Verify the command completes with status 02h.")
    assert len(items) == 1 and items[0]["n"] == 1


def test_split_fields_handles_missing_colon_and_leading_dot():
    from ingest_iol_testplans import _split_fields
    seg = ("Case 13: Something\nTest Procedure\n1. Do a thing\n"
           ".Observable Results: \n1. Verify the thing\n")
    f = _split_fields(seg, ["Test Procedure", "Observable Results"])
    assert "Test Procedure" in f and "Observable Results" in f


def test_resolve_ref_cross_test_and_local():
    from ingest_iol_testplans import _resolve_ref
    by_id = {"1.26/3": {"id": "1.26/3"}, "2.1/4": {"id": "2.1/4"}}
    assert _resolve_ref("1.26.3", "2.1", by_id)["id"] == "1.26/3"
    assert _resolve_ref("4", "2.1", by_id)["id"] == "2.1/4"


def test_materialize_replace_step():
    from ingest_iol_testplans import _materialize
    base = {"id": "1.26/3", "test_id": "1.26",
            "steps": [{"n": 1, "text": "step one"}, {"n": 2, "text": "step two"},
                      {"n": 3, "text": "step three"}],
            "observables": [{"n": 1, "text": "verify it"}], "raw_text": "",
            "materialized_from": None}
    sub = {"id": "1.26/3/2", "test_id": "1.26", "steps": [], "observables": [],
           "_parent_id": "1.26/3",
           "raw_text": ("2. Sanitize Operation\n"
                        "1. Replace step 2 from the test case with the following steps:\n"
                        "2. Check SANICAP field\n3. Start a sanitize operation\n"),
           "materialized_from": None}
    parent_preamble = "follow the test procedure of test case 3 except for the modified steps"
    base["_subcases_preamble"] = parent_preamble
    _materialize([base, sub])
    assert sub["materialized_from"] == "1.26/3"
    texts = [s["text"] for s in sub["steps"]]
    assert "step one" in texts and "step three" in texts
    assert "step two" not in texts
    assert any("SANICAP" in t for t in texts)
    assert sub["observables"] == base["observables"]


# ---------------------------------------------------------------------------
# generator.format_test_context

def _row(**over):
    row = {"id": "1.1/3", "test_id": "1.1", "case_num": "3", "subcase_num": None,
           "title": "Case 3: CNS=02h", "test_title": "Test 1.1 – Identify Command",
           "group_name": "Group 1: Admin Command Set", "purpose": "Verify identify",
           "setup": "Default setup", "references_text": None, "discussion": None,
           "steps": [{"n": 1, "text": "Send Identify with CNS=02h"}],
           "observables": [{"n": 1, "text": "Verify namespace list returned"}],
           "possible_problems": None, "materialized_from": None, "raw_text": "x"}
    row.update(over)
    return row


def test_format_test_context_renders_plan_block():
    from src.pipeline.generator import format_test_context
    out = format_test_context(_row())
    assert "<test_plan" in out and "</test_plan>" in out
    assert "1.1" in out and "CNS=02h" in out
    assert "Send Identify with CNS=02h" in out
    assert "Verify namespace list returned" in out
    assert "NEVER cite it" in out


def test_format_test_context_brace_safe():
    # Row text with braces must not blow up (block is appended AFTER
    # system_prompt.format(context=...) and never .format()ed itself).
    from src.pipeline.generator import format_test_context
    out = format_test_context(_row(purpose="uses {braces} and {context}"))
    assert "{braces}" in out


def test_citations_ignore_test_plan_text():
    # The plan block lives in the system prompt, not the context chunks, so
    # _extract_citations can never produce a citation pointing at it.
    from src.pipeline.generator import _extract_citations
    answer = "The controller shall do X [1]."
    chunks = [{"id": "c1", "section_id": "5.17", "section_title": "Identify",
               "content_type": "prose", "text_raw": "...", "pdf_pages": [10],
               "figure_number": None, "has_normative": True}]
    citations = _extract_citations(answer, chunks)
    assert all(c.get("section_id") != "test_plan" for c in citations)


# ---------------------------------------------------------------------------
# one-test-per-chat lock (app._resolve_test_context)

def _reset_bindings():
    from src.pipeline import app as appmod
    appmod._CONVO_TESTS.clear()


def test_lock_binds_then_allows_same_test_other_case():
    from src.pipeline.app import _resolve_test_context
    _reset_bindings()
    with patch("src.pipeline.search.fetch_test_plan",
               side_effect=lambda pid: _row(id=pid, test_id=pid.split("/")[0])):
        row = _resolve_test_context("conv1", "1.1/3")
        assert row["test_id"] == "1.1"
        # different case of the SAME test: allowed
        row2 = _resolve_test_context("conv1", "1.1/16/3")
        assert row2["id"] == "1.1/16/3"


def test_lock_rejects_other_test_and_missing_test():
    from src.pipeline.app import _resolve_test_context
    _reset_bindings()
    with patch("src.pipeline.search.fetch_test_plan",
               side_effect=lambda pid: _row(id=pid, test_id=pid.split("/")[0])):
        _resolve_test_context("conv2", "1.1/3")
        with pytest.raises(HTTPException) as e:
            _resolve_test_context("conv2", "2.1/4")
        assert e.value.status_code == 409
        with pytest.raises(HTTPException) as e2:
            _resolve_test_context("conv2", None)
        assert e2.value.status_code == 409


def test_new_conversation_resets_binding():
    from src.pipeline.app import _resolve_test_context
    _reset_bindings()
    with patch("src.pipeline.search.fetch_test_plan",
               side_effect=lambda pid: _row(id=pid, test_id=pid.split("/")[0])):
        _resolve_test_context("conv3", "1.1/3")
        # a NEW conversation_id is a fresh binding
        row = _resolve_test_context("conv4", "2.1/4")
        assert row["test_id"] == "2.1"


def test_no_test_no_binding():
    from src.pipeline.app import _resolve_test_context
    _reset_bindings()
    assert _resolve_test_context("conv5", None) is None
    # selecting a test later in an unbound conversation is allowed (binds then)
    with patch("src.pipeline.search.fetch_test_plan",
               side_effect=lambda pid: _row(id=pid, test_id=pid.split("/")[0])):
        assert _resolve_test_context("conv5", "1.1/3")["test_id"] == "1.1"


def test_unknown_test_case_400():
    from src.pipeline.app import _resolve_test_context
    _reset_bindings()
    with patch("src.pipeline.search.fetch_test_plan", return_value=None):
        with pytest.raises(HTTPException) as e:
            _resolve_test_context("conv6", "9.9/99")
        assert e.value.status_code == 400


# ---------------------------------------------------------------------------
# test-plan priming (orchestrator.prime_test_plan + app pin cache)

def _chunk(sid, **kw):
    return {"id": f"c-{sid}", "section_id": sid, "section_title": f"S {sid}",
            "spec": "command", "chunk_index": 0, "text_raw": "...", **kw}


def test_prime_test_plan_loops_until_understood_and_tags():
    from src.pipeline import orchestrator as om

    class _Res:
        model = "m"; prompt_tokens = 10; output_tokens = 5
    responses = [
        ({"understood": False, "queries": ["identify command", "CNS values"],
          "important_sections": []}, _Res()),
        ({"understood": True, "queries": [],
          "important_sections": ["5.17"]}, _Res()),
    ]
    with patch.object(om.query_processor, "generate_json",
                      side_effect=responses) as gj, \
         patch.object(om, "hybrid_search",
                      return_value=([_chunk("5.17"), _chunk("1.2")], [])):
        out = om.prime_test_plan(_row(id="1.1/3", test_id="1.1"),
                                 config=om.PipelineConfig())
    assert gj.call_count == 2
    assert [c["section_id"] for c in out["chunks"]] == ["5.17"]
    assert all(c["method"] == "testplan_prime" for c in out["chunks"])
    assert len(out["llm_calls"]) == 2
    assert out["llm_calls"][0]["stage"] == "testplan_prime"


def test_prime_test_plan_llm_failure_returns_empty():
    from src.pipeline import orchestrator as om
    with patch.object(om.query_processor, "generate_json",
                      side_effect=RuntimeError("boom")):
        out = om.prime_test_plan(_row(id="1.1/3", test_id="1.1"),
                                 config=om.PipelineConfig())
    assert out["chunks"] == [] and out["llm_calls"] == []


def test_test_pins_cached_per_conversation():
    from src.pipeline import app as appmod
    appmod._CONVO_TEST_PINS.clear()
    primed = {"chunks": [_chunk("5.17")], "llm_calls": [{"stage": "testplan_prime"}],
              "trace": []}
    with patch.object(appmod, "prime_test_plan", return_value=primed) as pr:
        pins, meta = appmod._test_pins("convP", _row(id="1.1/3", test_id="1.1"),
                                       None)
        assert pins and meta is primed
        pins2, meta2 = appmod._test_pins("convP", _row(id="1.1/3", test_id="1.1"),
                                         None)
        assert pins2 == pins and meta2 is None   # cache hit, tokens not re-spent
    assert pr.call_count == 1


def test_history_with_pins_merges_and_dedups():
    from src.pipeline.app import _history_with_pins
    pins = [_chunk("5.17")]
    # no history → minimal history dict carrying the pins
    h = _history_with_pins(None, pins)
    assert h["turns"] == [] and h["pinned_chunks"] == pins
    # pins go first and duplicates collapse
    h2 = _history_with_pins(
        {"turns": [{"query": "q", "answer": "a"}],
         "pinned_chunks": [_chunk("5.17"), _chunk("2.1")]}, pins)
    assert [c["section_id"] for c in h2["pinned_chunks"]] == ["5.17", "2.1"]
    # no pins → history untouched
    assert _history_with_pins(None, []) is None


def test_priming_disabled_via_config():
    from src.pipeline import app as appmod
    from src.pipeline.orchestrator import PipelineConfig
    with patch.object(appmod, "prime_test_plan") as pr, \
         patch.object(appmod, "orchestrate", return_value={}) as orch:
        appmod._orchestrate_query(
            "q", config=PipelineConfig(testplan_priming=False), debug=False,
            agentic=False, history=None,
            test_context=_row(id="1.1/3", test_id="1.1"),
            conversation_id="convD")
    pr.assert_not_called()
    assert orch.call_args.kwargs["history"] is None


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
