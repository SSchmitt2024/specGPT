"""Three-tier gap analysis for batch context assembly (gap_mode="tiered").

Replaces the LLM-verdict loop for batch runs with:

  Tier 1  expand_section_refs      deterministic "see section X.Y" expansion
                                   (regex from src.relationships, no LLM).
  Tier 2  score_gap_check          sufficiency from rerank/RRF/structured
                                   scores (pure function, no I/O).
  Tier 3  constrained_gap_check    small model constrained to a tiny JSON
                                   object, consulted only when Tier 2 can't
                                   decide (or to name WHAT to fetch).

Consumed by orchestrator._run_tiered_context_loop. The live agentic loop
(_run_stage5_and_finalize) does not import this module.
"""

from __future__ import annotations

import json
import logging
import os
import re

from src.relationships import extract_cross_refs_from_text
from src.pipeline import search
from src.pipeline.generator import (
    DEEPTHOUGHT_BASE_URL,
    DeepThoughtUnreachableError,
    resolve_deepthought_model,
)

logger = logging.getLogger(__name__)

# Must match orchestrator.ALL_SPECS (imported there; importing back would be
# circular).
_ALL_SPECS = "all"


# -----------------------------------------------------------------------------
# Tier 1: deterministic cross-reference expansion
# -----------------------------------------------------------------------------

def expand_section_refs(
    pool: list[dict],
    *,
    spec: str,
    cap: int = 4,
    attempted: set[tuple[str, str]] | None = None,
) -> list[dict]:
    """Fetch sections that the pool's chunks strongly cross-reference.

    Pattern-matched, no LLM: reuses the ingest-time extractor
    (relationships.extract_cross_refs_from_text) over ``text_raw``, keeps only
    "strong" section targets ("see section 5.2.1", "as defined in ..."), ranks
    by reference count, and direct-fetches via search.fetch_section_chunks.
    Figures are NOT handled here; orchestrator._expand_referenced_figures
    already covers them deterministically.

    ``attempted`` is the caller's cross-iteration fetch blacklist; entries
    ``("xref_section", sid)`` are added for every section tried so a repeated
    reference can't be re-fetched.
    """
    attempted = attempted if attempted is not None else set()
    present = {str(c.get("section_id") or "") for c in pool}
    seen_ids = {c.get("id") or c.get("chunk_id") for c in pool}

    counts: dict[str, int] = {}
    for chunk in pool:
        text = chunk.get("text_raw") or ""
        if not text:
            continue
        source = f"section:{chunk.get('section_id') or chunk.get('id') or '?'}"
        for edge in extract_cross_refs_from_text(text, source):
            if edge.get("strength") != "strong":
                continue
            target = str(edge.get("target") or "")
            if not target.startswith("section:"):
                continue
            sid = target.split(":", 1)[1]
            if sid in present or ("xref_section", sid) in attempted:
                continue
            counts[sid] = counts.get(sid, 0) + 1

    wanted = sorted(counts, key=lambda s: (-counts[s], s))[: max(0, cap)]
    fetch_spec = None if spec == _ALL_SPECS else spec

    out: list[dict] = []
    for sid in wanted:
        attempted.add(("xref_section", sid))
        try:
            hits = search.fetch_section_chunks(sid, top_k=3, spec=fetch_spec)
        except Exception as e:  # noqa: BLE001
            logger.warning("xref section fetch failed (%s): %s", sid, e)
            hits = []
        for h in hits:
            cid = h.get("id") or h.get("chunk_id")
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            h = dict(h)
            h["method"] = "xref_section_fetch"
            out.append(h)
    return out


# -----------------------------------------------------------------------------
# Tier 2: score-based sufficiency
# -----------------------------------------------------------------------------

def score_gap_check(
    pool: list[dict],
    *,
    structured_found: bool = False,
    structured_confidence: str | None = None,
    sufficient_threshold: float = 0.6,
    insufficient_threshold: float = 0.3,
    min_strong: int = 2,
) -> dict:
    """Decide sufficiency from scores already on the chunks. Pure, no I/O.

    Signals: ``rerank_score`` (Voyage rerank, normalized 0-1, set on the whole
    pool), ``contributing_methods`` (chunks found by 2+ retrieval methods get
    a small threshold discount), and structured-lookup confidence.

    Returns {"decision": "sufficient"|"ambiguous"|"insufficient",
             "top1": float|None, "strong_n": int, "scored_n": int}.
    A pool with no usable scores (rerank API failure) is "ambiguous" so the
    caller escalates instead of trusting a blind pass.
    """
    scores: list[float] = []
    strong_n = 0
    for c in pool:
        s = c.get("rerank_score")
        if not isinstance(s, (int, float)) or s == float("-inf"):
            continue
        s = float(s)
        scores.append(s)
        threshold = sufficient_threshold
        if len(set(c.get("contributing_methods") or [])) >= 2:
            threshold -= 0.05
        if s >= threshold:
            strong_n += 1

    top1 = max(scores) if scores else None
    if top1 is None:
        decision = "ambiguous"
    elif (structured_found and structured_confidence == "HIGH"
          and top1 >= insufficient_threshold):
        decision = "sufficient"
    elif top1 >= sufficient_threshold and strong_n >= min_strong:
        decision = "sufficient"
    elif top1 < insufficient_threshold:
        decision = "insufficient"
    else:
        decision = "ambiguous"
    return {"decision": decision, "top1": top1,
            "strong_n": strong_n, "scored_n": len(scores)}


# -----------------------------------------------------------------------------
# Tier 3: constrained-JSON judgment from a small model
# -----------------------------------------------------------------------------

_GAP_JUDGE_SYSTEM = (
    "You judge whether retrieved hardware-spec excerpts contain enough "
    "information to answer a question. Respond with ONLY a JSON object, no "
    "prose, no markdown fences:\n"
    '{"sufficient": true|false, "missing": ["<item>", ...]}\n'
    'Each "missing" item must be fetchable: a dotted section id (e.g. '
    '"5.2.1"), "Figure N", or a short search phrase (under 8 words). '
    "Use an empty list when sufficient. At most 5 items."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_INVENTORY_LINES = 15
_SNIPPET_CHARS = 150


def _parse_judge_json(text: str) -> dict | None:
    """Tolerant parse of the judge's output. None on anything unusable."""
    text = _THINK_RE.sub("", text or "")
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or "sufficient" not in parsed:
        return None
    missing_raw = parsed.get("missing")
    missing: list[str] = []
    if isinstance(missing_raw, list):
        for v in missing_raw:
            s = str(v).strip()
            if s and len(s) <= 80 and s not in missing:
                missing.append(s)
    return {"sufficient": bool(parsed.get("sufficient")), "missing": missing[:5]}


def _judge_call(user_prompt: str, *, model: str, max_tokens: int = 300) -> tuple[str, dict]:
    """One chat call to DeepThought. Tries vLLM-style constrained decoding
    (response_format json_object) first; falls back to a plain call if the
    gateway rejects it. Returns (text, tokens_used)."""
    api_key = os.environ.get("DEEPTHOUGHT_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPTHOUGHT_API_KEY environment variable is not set.")

    from openai import (  # lazy import, same pattern as generator._call_deepthought
        APIConnectionError,
        APITimeoutError,
        BadRequestError,
        OpenAI,
    )

    client = OpenAI(api_key=api_key, base_url=DEEPTHOUGHT_BASE_URL)
    kwargs = dict(
        model=resolve_deepthought_model(model),
        max_tokens=max_tokens,
        # The DeepThought HF backend rejects temperature=0.0 ("must be
        # strictly positive"); 0.01 is effectively greedy.
        temperature=0.01,
        messages=[
            {"role": "system", "content": _GAP_JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    )
    try:
        try:
            resp = client.chat.completions.create(
                response_format={"type": "json_object"}, **kwargs
            )
        except BadRequestError:
            # Gateway/model doesn't support constrained decoding; the tolerant
            # parser below still guards the output.
            resp = client.chat.completions.create(**kwargs)
    except (APIConnectionError, APITimeoutError) as e:
        raise DeepThoughtUnreachableError(
            "Can't reach DeepThought at dtcontroller.sr.unh.edu — connect to "
            "the USNH GlobalProtect VPN (or use campus Wi-Fi) and try again."
        ) from e

    text = (resp.choices[0].message.content or "") if resp.choices else ""
    usage = getattr(resp, "usage", None)
    tokens = {
        "prompt": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion": int(getattr(usage, "completion_tokens", 0) or 0),
    }
    return text, tokens


def constrained_gap_check(
    query: str,
    pool: list[dict],
    *,
    model: str = "deepthought-qwen3-30b",
) -> tuple[dict, dict | None]:
    """Tier 3: ask a small model for a constrained JSON sufficiency verdict.

    Prompt is a compact inventory (section id + title + snippet), NOT full
    chunk text — output is ~100 tokens of JSON, so hallucination surface and
    cost stay minimal.

    Returns (verdict, llm_call) where verdict is always well-formed:
    {"sufficient": bool, "missing": [str, ...]}. Parse failure after one
    retry fails toward {"sufficient": False, "missing": []} — one more
    retrieval round, never a hallucinated "sufficient". llm_call is a
    token-breakdown dict for the cost panel, or None if the call failed.
    """
    lines = []
    for c in pool[:_INVENTORY_LINES]:
        sid = c.get("section_id") or (
            f"Figure {c.get('figure_number')}" if c.get("figure_number") else "?"
        )
        title = c.get("section_title") or ""
        snippet = (c.get("text_raw") or "")[:_SNIPPET_CHARS].replace("\n", " ")
        lines.append(f"[{sid}] {title} :: {snippet}")
    user_prompt = (
        f"Question:\n{query}\n\nRetrieved excerpts:\n" + "\n".join(lines)
    )

    llm_call: dict | None = None
    for attempt in range(2):
        try:
            text, tokens = _judge_call(user_prompt, model=model)
        except DeepThoughtUnreachableError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("tiered judge call failed (attempt %d): %s", attempt + 1, e)
            continue
        llm_call = {"stage": "tiered_judge", "model": model, **tokens}
        verdict = _parse_judge_json(text)
        if verdict is not None:
            return verdict, llm_call
        logger.warning("tiered judge returned unparseable output (attempt %d)", attempt + 1)
    return {"sufficient": False, "missing": []}, llm_call
