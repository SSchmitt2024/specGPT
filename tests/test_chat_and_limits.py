"""
Unit tests for the chat/multi-turn backend and the request limits added with
it: RateLimiter, QueryRequest length cap, conversation turn cap, pinned-chunk
extraction, history formatting, and the pinned reserve in assemble_context.

Run:  venv/bin/python3 -m pytest tests/test_chat_and_limits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.auth import RateLimiter  # noqa: E402


# ---------------------------------------------------------------------------
# RateLimiter

def test_rate_limiter_allows_then_blocks():
    rl = RateLimiter(max_requests=3, window_seconds=60.0)
    assert rl.retry_after("k") == 0.0
    assert rl.retry_after("k") == 0.0
    assert rl.retry_after("k") == 0.0
    wait = rl.retry_after("k")
    assert 0.0 < wait <= 60.0


def test_rate_limiter_keys_are_independent():
    rl = RateLimiter(max_requests=1, window_seconds=60.0)
    assert rl.retry_after("a") == 0.0
    assert rl.retry_after("a") > 0.0
    assert rl.retry_after("b") == 0.0


def test_rate_limiter_window_expiry():
    rl = RateLimiter(max_requests=1, window_seconds=0.01)
    assert rl.retry_after("k") == 0.0
    import time
    time.sleep(0.02)
    assert rl.retry_after("k") == 0.0


# ---------------------------------------------------------------------------
# QueryRequest length cap

def test_query_request_rejects_oversized_query():
    from src.pipeline.app import QueryRequest
    QueryRequest(query="x" * 4000)  # at the cap: fine
    with pytest.raises(Exception):
        QueryRequest(query="x" * 4001)


# ---------------------------------------------------------------------------
# Conversation store

def _fake_result(query="q", answer="a [§1.1]", cited=("1.1",), halluc=()):
    citations = [{"section_id": s, "hallucinated": False} for s in cited]
    citations += [{"section_id": s, "hallucinated": True} for s in halluc]
    dedup = [
        {"section_id": s, "spec": "base", "chunk_index": 0,
         "text_raw": f"text for {s}", "id": i}
        for i, s in enumerate(dict.fromkeys(list(cited) + list(halluc)))
    ]
    return {"query": query, "answer": answer, "citations": citations,
            "deduplicated": dedup}


def test_cited_chunks_skips_hallucinated_and_dedupes():
    from src.pipeline.app import _cited_chunks
    result = _fake_result(cited=("1.1",), halluc=("9.9",))
    chunks = _cited_chunks(result)
    assert [c["section_id"] for c in chunks] == ["1.1"]


def test_conversation_roundtrip_and_turn_cap():
    from src.pipeline import app as app_mod
    from fastapi import HTTPException

    cid = "test-convo-cap"
    app_mod._CONVERSATIONS.pop(cid, None)

    assert app_mod._conversation_history(cid) is None  # first turn: no history
    idx = app_mod._conversation_append(cid, _fake_result(query="q0"))
    assert idx == 0

    hist = app_mod._conversation_history(cid)
    assert hist["turns"] == [{"query": "q0", "answer": "a [§1.1]"}]
    assert hist["pinned_chunks"][0]["section_id"] == "1.1"

    with app_mod._CONVERSATIONS_LOCK:
        app_mod._CONVERSATIONS[cid] = [
            {"query": f"q{i}", "answer": "a", "pinned_chunks": []}
            for i in range(app_mod.MAX_CONVERSATION_TURNS)
        ]
    with pytest.raises(HTTPException) as exc:
        app_mod._conversation_history(cid)
    assert exc.value.status_code == 409

    # Refine (exclude_last) re-answers the latest turn: never trips the cap.
    assert app_mod._conversation_history(cid, exclude_last=True) is not None

    app_mod._CONVERSATIONS.pop(cid, None)


def test_conversation_replace_last():
    from src.pipeline import app as app_mod
    cid = "test-convo-replace"
    app_mod._CONVERSATIONS.pop(cid, None)
    app_mod._conversation_append(cid, _fake_result(query="q0", answer="first"))
    app_mod._conversation_replace_last(cid, _fake_result(query="q0", answer="refined"))
    hist = app_mod._conversation_history(cid)
    assert hist["turns"][-1]["answer"] == "refined"
    app_mod._CONVERSATIONS.pop(cid, None)


# ---------------------------------------------------------------------------
# History formatting + pinned reserve in assemble_context

def test_format_history_budget_keeps_newest():
    from src.pipeline.generator import _format_history
    turns = [{"query": f"q{i}", "answer": "a" * 400} for i in range(6)]
    out = _format_history(turns, max_tokens=150)
    assert "q5" in out and "q0" not in out
    full = _format_history(turns, max_tokens=100_000)
    assert full.index("q0") < full.index("q5")  # oldest first


def test_assemble_context_pins_prior_citations():
    from src.pipeline.generator import assemble_context

    filler = [
        {"section_id": f"5.{i}", "section_title": f"S{i}", "text_raw": "w " * 3000,
         "spec": "base", "chunk_index": 0}
        for i in range(6)
    ]
    pinned = [{"section_id": "1.1", "section_title": "Pinned", "text_raw": "pinned text",
               "spec": "base", "chunk_index": 0}]

    # Main budget saturated by filler; the pinned chunk must still be present.
    ctx, used = assemble_context("q", filler, max_context_tokens=2000,
                                 figure_reserve_tokens=0, pinned_chunks=pinned)
    assert "pinned text" in ctx
    assert any(c.get("section_id") == "1.1" for c in used)

    # Reserve of 0 disables pinning.
    ctx2, used2 = assemble_context("q", filler, max_context_tokens=2000,
                                   figure_reserve_tokens=0, pinned_chunks=pinned,
                                   pinned_reserve_tokens=0)
    assert not any(c.get("section_id") == "1.1" for c in used2)

    # A chunk both pinned and retrieved appears only once.
    ctx3, used3 = assemble_context("q", pinned + filler, max_context_tokens=2000,
                                   figure_reserve_tokens=0, pinned_chunks=pinned)
    assert sum(1 for c in used3 if c.get("section_id") == "1.1") == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
