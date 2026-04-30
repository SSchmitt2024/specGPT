"""
Phase 2 — Step 2.2a: Query Classifier + Decomposer

Takes a raw user question and produces:

    {
      "query":      "<original>",
      "type":       "lookup" | "structural" | "relational" | "procedural",
      "entities":   [{"text": "...", "kind": "field"|"figure"|"hex"|"fid"|"cdw"|"section"}],
      "sub_queries":["...", "..."],     # 1–3 focused queries to feed retrieval
      "rationale":  "<one short sentence from the LLM>"
    }

Pipeline:

  1. Deterministic entity extraction (free, no LLM).
       Regex for hex values, FIDs, CDW positions, figure numbers, section refs.
       Token-level lookup against `data/field_index.json` for canonical field names
       (e.g. NSSES, HMPRE, CAP, MQES).

  2. Single LLM call (Gemini Flash via src/llm/client.generate_json, ~$0.001).
       Classifies the query AND decomposes if relational/procedural.
       Receives the deterministic entities so it doesn't re-extract them.
       JSON-mode → strict output schema.

  3. Validation + assembly into QueryDecomposition.

Lookup / structural queries pass through with sub_queries == [original_query].
Relational / procedural queries get split into 2–3 focused sub-queries.

Run:
  python -m src.pipeline.query_processor "What are bits 7:4 of CDW10 in Set Features?"
  python -m src.pipeline.query_processor --no-llm "any query"   # entities only, skip LLM
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path

from src.llm.client import generate_json


# ---------------------------------------------------------------------------
# Paths

FIELD_INDEX_PATH = Path("data/field_index.json")


# ---------------------------------------------------------------------------
# Types

VALID_TYPES = ("lookup", "structural", "relational", "procedural")


@dataclass
class Entity:
    text: str
    kind: str  # "field" | "figure" | "hex" | "fid" | "cdw" | "section"


@dataclass
class QueryDecomposition:
    query: str
    type: str
    entities: list[Entity] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Deterministic entity extraction
#
# All patterns are intentionally narrow — false negatives are fine (the LLM
# step still sees the raw query), but false positives leak garbage into the
# structured-lookup path and waste retrieval budget.

# Hex literals: 0x1A, 0x1Ah, 0x1A_3F, or trailing-h form like 1Ah / FFh.
# Require ≥2 hex chars on the trailing-h form to avoid catching plain words.
_RE_HEX = re.compile(r"\b(?:0[xX][0-9A-Fa-f]+|[0-9A-Fa-f]{2,}h)\b")

# Feature Identifier: "FID 0x01", "FID 01h", "Feature Identifier 0x12".
_RE_FID = re.compile(
    r"\b(?:FID|Feature\s+Identifier)\s*[:=]?\s*((?:0[xX])?[0-9A-Fa-f]+h?)\b",
    re.IGNORECASE,
)

# Command Dword: CDW10, CDW0[7:4], CDW 10.
_RE_CDW = re.compile(r"\bCDW\s*\d{1,2}(?:\s*\[\s*\d+\s*:\s*\d+\s*\])?", re.IGNORECASE)

# Figure references: "Figure 312", "Fig 312", "figure 41".
_RE_FIGURE = re.compile(r"\b(?:Figure|Fig\.?)\s*(\d{1,4})\b", re.IGNORECASE)

# Section refs: "1.4.2", "5.1.2.3", with at least one dot to avoid catching version numbers.
# Anchored by an optional "section" / "§" prefix or sentence boundary.
_RE_SECTION = re.compile(
    r"(?:(?<=\s)|(?<=^)|(?<=section\s)|(?<=§\s))(\d+(?:\.\d+){1,5})\b",
    re.IGNORECASE,
)

# Field-name candidates: ALL-CAPS tokens (and CAP.MQES style), 2–10 chars, possibly with dot.
_RE_FIELD_CANDIDATE = re.compile(r"\b([A-Z][A-Z0-9]{1,9}(?:\.[A-Z][A-Z0-9]{1,9})?)\b")


@lru_cache(maxsize=1)
def _load_field_index() -> set[str]:
    """Lazy-load the field-name set from data/field_index.json. Cached for process lifetime."""
    if not FIELD_INDEX_PATH.exists():
        return set()
    with open(FIELD_INDEX_PATH, encoding="utf-8") as f:
        idx = json.load(f)
    return set(idx.keys())


def _dedup_keep_order(items: list[Entity]) -> list[Entity]:
    seen = set()
    out: list[Entity] = []
    for e in items:
        key = (e.kind, e.text.upper())
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def extract_entities(query: str) -> list[Entity]:
    """Pull spec-shaped tokens out of the query. No LLM, no network."""
    found: list[Entity] = []

    # Order matters: extract more specific patterns first so generic ones don't
    # eat their substrings (e.g. FID 0x01 should be captured as fid, not as raw hex).
    for m in _RE_FID.finditer(query):
        found.append(Entity(text=m.group(0), kind="fid"))

    for m in _RE_CDW.finditer(query):
        found.append(Entity(text=m.group(0), kind="cdw"))

    for m in _RE_FIGURE.finditer(query):
        found.append(Entity(text=m.group(0), kind="figure"))

    for m in _RE_SECTION.finditer(query):
        found.append(Entity(text=m.group(1), kind="section"))

    # Hex last — skip any hex token already absorbed by FID/CDW above.
    consumed_spans = []
    for ent in found:
        # rough span tracking: find the first occurrence of each captured text
        idx = query.find(ent.text)
        if idx >= 0:
            consumed_spans.append((idx, idx + len(ent.text)))

    def _overlaps(start: int, end: int) -> bool:
        return any(not (end <= s or start >= e) for s, e in consumed_spans)

    for m in _RE_HEX.finditer(query):
        if not _overlaps(m.start(), m.end()):
            found.append(Entity(text=m.group(0), kind="hex"))

    # Field-name candidates: must be a known canonical field in field_index.json.
    # Skip tokens already captured as a more specific kind (cdw/fid/figure/hex)
    # to avoid e.g. CDW10 showing up twice (once as cdw, once as field).
    already_captured = {e.text.upper() for e in found}
    field_set = _load_field_index()
    if field_set:
        for m in _RE_FIELD_CANDIDATE.finditer(query):
            tok = m.group(1)
            if tok.upper() in already_captured:
                continue
            # Match either the full token or its first dotted segment (e.g. CAP.MQES → MQES).
            if tok in field_set:
                found.append(Entity(text=tok, kind="field"))
            elif "." in tok:
                head, tail = tok.split(".", 1)
                if head in field_set and head.upper() not in already_captured:
                    found.append(Entity(text=head, kind="field"))
                if tail in field_set and tail.upper() not in already_captured:
                    found.append(Entity(text=tail, kind="field"))

    return _dedup_keep_order(found)


# ---------------------------------------------------------------------------
# LLM classification + decomposition

_SYSTEM_PROMPT = """You are a query analyzer for an NVMe specification Q&A system.

You receive a user question and a list of pre-extracted entities (field names,
figure numbers, hex values, etc.). Your job:

1. CLASSIFY the question into exactly one type:
   - lookup     : asks for a specific value, field, bit, or table cell.
                  Examples: "What are bits 7:4 of CDW10 in Set Features?"
                            "What is HMPRE?"
                            "What's the offset of CAP register?"
   - structural : asks how a structure/concept is organized or what it is.
                  Examples: "How is the Submission Queue organized?"
                            "What is a namespace?"
                            "Describe the Identify Controller data structure."
   - relational : asks how multiple entities interact, depend on, or differ.
                  Examples: "How do FID 0x01 and FID 0x12 interact?"
                            "What's the difference between SQ and CQ?"
                            "Which commands use the Host Memory Buffer?"
   - procedural : asks how to perform a multi-step operation or implement something.
                  Examples: "How do I implement SGLs?"
                            "What's the controller initialization sequence?"
                            "How does a host issue a Get Log Page command?"

2. DECOMPOSE (only for relational and procedural):
   Produce 2–3 focused sub-queries that, when answered together, would let
   another agent compose a full answer. Each sub-query must be a complete
   question on its own, mention specific entities by name, and be answerable
   from a single passage of the spec.

   For lookup and structural, return sub_queries as a single-element list
   containing the original query verbatim (no decomposition).

3. Return JSON ONLY, matching this schema exactly:

   {
     "type": "lookup" | "structural" | "relational" | "procedural",
     "sub_queries": ["...", "..."],
     "rationale": "one short sentence explaining the classification"
   }

RULES:
- Do NOT echo the entities list back; the caller already has it.
- Do NOT invent entities not present in the question or the entities list.
- Sub-queries must be in English, declarative or interrogative, no bullet markers.
- If unsure between two types, prefer the simpler one (lookup > structural > relational > procedural).
"""


def _build_user_prompt(query: str, entities: list[Entity]) -> str:
    if entities:
        entity_lines = "\n".join(f"  - {e.text} ({e.kind})" for e in entities)
        entity_block = f"Pre-extracted entities:\n{entity_lines}\n\n"
    else:
        entity_block = "Pre-extracted entities: (none)\n\n"
    return f"{entity_block}User question:\n{query}\n"


def _normalize_llm_output(parsed: object, query: str) -> tuple[str, list[str], str]:
    """Coerce the LLM's JSON into (type, sub_queries, rationale). Raises on hard failures."""
    if not isinstance(parsed, dict):
        raise ValueError(f"LLM returned non-object: {type(parsed).__name__}")

    qtype = parsed.get("type")
    if qtype not in VALID_TYPES:
        raise ValueError(f"LLM returned invalid type: {qtype!r}")

    raw_subs = parsed.get("sub_queries", [])
    if not isinstance(raw_subs, list) or not raw_subs:
        sub_queries = [query]
    else:
        sub_queries = [str(s).strip() for s in raw_subs if str(s).strip()]
        if not sub_queries:
            sub_queries = [query]

    # Decomposition policy enforcement: lookup/structural never get expanded.
    if qtype in ("lookup", "structural"):
        sub_queries = [query]
    else:
        # Cap at 3, never duplicate the original verbatim alongside reworded variants.
        sub_queries = sub_queries[:3]

    rationale = str(parsed.get("rationale", "")).strip()
    return qtype, sub_queries, rationale


def classify_and_decompose(
    query: str,
    entities: list[Entity],
    *,
    model: str | None = None,
) -> tuple[str, list[str], str]:
    """Single LLM call. Returns (type, sub_queries, rationale)."""
    user_prompt = _build_user_prompt(query, entities)
    kwargs = {
        "system": _SYSTEM_PROMPT,
        "temperature": 0.0,
        "max_output_tokens": 400,
    }
    if model:
        kwargs["model"] = model

    parsed, _result = generate_json(user_prompt, **kwargs)
    return _normalize_llm_output(parsed, query)


# ---------------------------------------------------------------------------
# Public API

def _heuristic_result(query: str, entities: list[Entity], note: str) -> QueryDecomposition:
    qtype = "lookup" if entities else "structural"
    return QueryDecomposition(
        query=query,
        type=qtype,
        entities=entities,
        sub_queries=[query],
        rationale=f"heuristic: {note}",
    )


def process_query(
    query: str,
    *,
    use_llm: bool = True,
    model: str | None = None,
    strict: bool = False,
) -> QueryDecomposition:
    """
    Full pipeline: deterministic entities → optional LLM classify+decompose.

    With use_llm=False, returns the heuristic-only result.
    With use_llm=True (default), attempts the LLM call and falls back to the
    heuristic on any failure (quota, network, malformed JSON). Pass strict=True
    to surface the underlying exception instead of falling back — useful when
    debugging the LLM prompt itself.
    """
    query = query.strip()
    if not query:
        raise ValueError("query is empty")

    entities = extract_entities(query)

    if not use_llm:
        return _heuristic_result(query, entities, "no-llm mode")

    try:
        qtype, sub_queries, rationale = classify_and_decompose(query, entities, model=model)
    except Exception as e:  # noqa: BLE001 — broad on purpose; this is a fallback boundary
        if strict:
            raise
        return _heuristic_result(query, entities, f"LLM call failed ({type(e).__name__})")

    return QueryDecomposition(
        query=query,
        type=qtype,
        entities=entities,
        sub_queries=sub_queries,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# CLI

def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Classify + decompose a user query.")
    parser.add_argument("query", nargs="+", help="The user question (quote it).")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the LLM call and return entities-only heuristic result.",
    )
    parser.add_argument("--model", default=None, help="Override LLM model.")
    parser.add_argument("--strict", action="store_true", help="Re-raise LLM errors instead of falling back to heuristic.")
    args = parser.parse_args(argv)

    query = " ".join(args.query)
    result = process_query(query, use_llm=not args.no_llm, model=args.model, strict=args.strict)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
