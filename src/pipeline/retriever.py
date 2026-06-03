"""
Phase 2 - Step 2.3: Retriever orchestration

Houses the structured lookup path (2.3a) and the merge primitives for the
hybrid retrieval path (2.3b).

  - `structured_lookup()` — deterministic field/register/table lookup
        query -> query_processor entities -> field_index.json -> fields.json
             -> parent table from tables.json -> compact source bundle
  - `rrf_merge()` — Reciprocal Rank Fusion across vector / tsvector / BM25
        result lists (and across sub-queries when query_processor
        decomposes a query)

The web backend should call `structured_lookup()` first. If it returns
`found=False`, route the query to the hybrid path that runs
`search.vector_search` + `search.tsvector_search` + `search.bm25_search`
per sub-query and combines the ranked lists via `rrf_merge()`.

CLI:
  python -m src.pipeline.retriever structured "What does bit 3 of OACS mean?"
  python -m src.pipeline.retriever structured --no-llm "What is NSSES?"
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.pipeline.query_processor import Entity, QueryDecomposition, process_query
from src.pipeline.table_serializer import serialize_table

logger = logging.getLogger(__name__)


DATA_DIR = Path("data")

DEFAULT_SPEC = "base"


def _norm_spec(spec: str | None) -> str:
    return (spec or DEFAULT_SPEC).strip().lower() or DEFAULT_SPEC


def _spec_data_dir(spec: str) -> Path:
    """Local-JSON fallback dir for a spec. Base lives at data/; others under
    data/<spec>/ (mirrors scripts/rerun_pipeline.sh's SPEC_DATA_DIR)."""
    spec = _norm_spec(spec)
    return DATA_DIR if spec == DEFAULT_SPEC else DATA_DIR / spec

# Page size for paginated Supabase reads. PostgREST may impose a server-side
# `db-max-rows` cap below this, so loaders advance by the actual rows
# returned and terminate on the first empty batch.
_LOOKUP_PAGE_SIZE = 1000


def _paginate(table: str, columns: str, *, order_col: str, spec: str | None = None) -> list[dict]:
    """Page through `table` selecting `columns`, ordered by `order_col`.

    When `spec` is given, restrict to rows tagged with that spec so Base and
    PCIe lookups stay isolated. Robust against PostgREST max-rows truncation:
    advance by actual batch size and stop only on empty batch.
    """
    from src.pipeline.search import supabase_client

    client = supabase_client()
    rows: list[dict] = []
    start = 0
    while True:
        builder = client.table(table).select(columns)
        if spec is not None:
            builder = builder.eq("spec", spec)
        resp = (
            builder
            .order(order_col)
            .range(start, start + _LOOKUP_PAGE_SIZE - 1)
            .execute()
        )
        batch = resp.data or []
        if not batch:
            break
        rows.extend(batch)
        start += len(batch)
        if start > 10_000_000:
            break
    return rows


_RE_BIT_RANGE = re.compile(
    r"\b(?:bit|bits)\s*(\d{1,3})(?:\s*(?::|-|to)\s*(\d{1,3}))?\b",
    re.IGNORECASE,
)
_RE_BYTE_RANGE = re.compile(
    r"\b(?:byte|bytes)\s*(\d{1,3})(?:\s*(?::|-|to)\s*(\d{1,3}))?\b",
    re.IGNORECASE,
)


@dataclass
class StructuredLookupResult:
    query: str
    found: bool
    confidence: str
    route: str = "structured_lookup"
    entities: list[dict] = field(default_factory=list)
    fields: list[dict] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _supabase_available() -> bool:
    """True if SUPABASE_URL is configured (env or .env)."""
    import os
    url = os.environ.get("SUPABASE_URL")
    if url:
        return True
    env = Path(".env")
    if env.exists():
        return any(line.startswith("SUPABASE_URL=") for line in env.read_text().splitlines())
    return False


@lru_cache(maxsize=None)
def load_field_index(spec: str = DEFAULT_SPEC) -> dict[str, list[dict]]:
    """
    Map field_name → list of field records, scoped to `spec`.

    When Supabase is configured, the read is paginated (and filtered by spec)
    so PostgREST's server-side row cap can't silently truncate the field
    index. Falls back to the spec's local JSON snapshot when Supabase isn't
    configured. Cached per spec.
    """
    spec = _norm_spec(spec)
    if _supabase_available():
        try:
            rows = _paginate("spec_field_index", "field_name, data", order_col="id", spec=spec)
            result: dict[str, list[dict]] = {}
            for row in rows:
                result.setdefault(row["field_name"], []).append(row["data"])
            if result:
                return result
            # Empty table → fall through to JSON snapshot rather than
            # serving an empty index for the rest of the process lifetime.
        except Exception as e:  # noqa: BLE001
            logger.warning("Supabase lookup load failed (%s); falling back to local JSON snapshot", e)
    with open(_spec_data_dir(spec) / "field_index.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def _full_name_index(spec: str = DEFAULT_SPEC) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Normalized multi-word ``full_name`` → the field acronyms that carry it.

    Drives the fuzzy full-name fallback in ``structured_lookup``. Single-word
    names (and acronyms) are deliberately excluded: fuzzy matching is only ever
    allowed against descriptive multi-word names, so a near-miss can never
    collapse one acronym into another (CRATT must never resolve to CRAT).

    Returned as a tuple-of-tuples so the cached value is immutable. Cached per
    spec; cleared by ``reload_lookup_caches()``.
    """
    by_name: dict[str, set[str]] = {}
    for records in load_field_index(spec).values():
        for rec in records:
            full = str(rec.get("full_name") or "").strip()
            fname = str(rec.get("field_name") or "").strip()
            if not full or not fname:
                continue
            norm = " ".join(_RE_WORD.findall(full.lower()))
            if " " not in norm:  # require >= 2 words; never fuzz single tokens
                continue
            by_name.setdefault(norm, set()).add(fname)
    return tuple((name, tuple(sorted(acrs))) for name, acrs in by_name.items())


@lru_cache(maxsize=None)
def load_fields(spec: str = DEFAULT_SPEC) -> list[dict]:
    spec = _norm_spec(spec)
    if _supabase_available():
        try:
            rows = _paginate("spec_fields", "data", order_col="name", spec=spec)
            result = [row["data"] for row in rows]
            if result:
                return result
        except Exception as e:  # noqa: BLE001
            logger.warning("Supabase lookup load failed (%s); falling back to local JSON snapshot", e)
    with open(_spec_data_dir(spec) / "fields.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def load_tables_by_figure(spec: str = DEFAULT_SPEC) -> dict[str, dict]:
    spec = _norm_spec(spec)
    if _supabase_available():
        try:
            rows = _paginate("spec_tables", "figure_number, data", order_col="figure_number", spec=spec)
            result = {str(r["figure_number"]): r["data"] for r in rows if r.get("figure_number")}
            if result:
                return result
        except Exception as e:  # noqa: BLE001
            logger.warning("Supabase lookup load failed (%s); falling back to local JSON snapshot", e)
    with open(_spec_data_dir(spec) / "tables.json", encoding="utf-8") as f:
        tables = json.load(f)
    return {str(t.get("figure_number")): t for t in tables if t.get("figure_number") is not None}


@lru_cache(maxsize=None)
def load_enum_index(spec: str = DEFAULT_SPEC) -> dict[str, dict]:
    """Map concept (`fid`/`lid`/`cns`/`opcode`/`status`) → index block, scoped to `spec`.

    Each block is ``{"concept", "label", "entries": [...]}`` as produced by
    ``src.enum_tables``. Prefers Supabase ``spec_enum_index`` (one row per
    concept), falling back to the spec's local ``enum_index.json`` snapshot.
    Returns ``{}`` when neither source has data — callers then fall back to the
    live table scan (``_enum_value_matches``), so a missing index never breaks
    value lookups. Cached per spec.
    """
    spec = _norm_spec(spec)
    if _supabase_available():
        try:
            # One row per entry; regroup into {concept: {concept,label,entries}}
            # so the in-memory shape matches the local-JSON snapshot exactly.
            rows = _paginate("spec_enum_index", "concept, label, data", order_col="concept", spec=spec)
            result: dict[str, dict] = {}
            for row in rows:
                concept = row.get("concept")
                if not concept:
                    continue
                block = result.setdefault(
                    concept, {"concept": concept, "label": row.get("label"), "entries": []}
                )
                block["entries"].append(row["data"])
            if result:
                return result
        except Exception as e:  # noqa: BLE001
            logger.warning("Supabase enum-index load failed (%s); falling back to local JSON snapshot", e)
    path = _spec_data_dir(spec) / "enum_index.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:  # noqa: BLE001
            logger.warning("local enum_index.json load failed (%s)", e)
    return {}


def reload_lookup_caches() -> None:
    """Drop cached field/table loaders. Call after a re-ingest run to pick up new data."""
    load_field_index.cache_clear()
    _full_name_index.cache_clear()
    load_fields.cache_clear()
    load_tables_by_figure.cache_clear()
    load_enum_index.cache_clear()


def _entity_to_dict(entity: Entity | dict) -> dict:
    if isinstance(entity, Entity):
        return {"text": entity.text, "kind": entity.kind}
    return {"text": str(entity.get("text", "")), "kind": str(entity.get("kind", ""))}


def _normalize_key(text: str) -> str:
    return text.strip().upper()


def _field_keys_from_entities(entities: list[Entity | dict]) -> list[str]:
    keys: list[str] = []
    for raw in entities:
        ent = _entity_to_dict(raw)
        if ent["kind"] != "field":
            continue

        text = _normalize_key(ent["text"])
        candidates = [text]
        if "." in text:
            candidates.extend(part for part in text.split(".") if part)

        for candidate in candidates:
            if candidate not in keys:
                keys.append(candidate)
    return keys


def _figures_from_entities(entities: list[Entity | dict]) -> list[str]:
    figures: list[str] = []
    for raw in entities:
        ent = _entity_to_dict(raw)
        if ent["kind"] != "figure":
            continue
        match = re.search(r"\d{1,4}", ent["text"])
        if match and match.group(0) not in figures:
            figures.append(match.group(0))
    return figures


def _parse_range(match: re.Match) -> tuple[int, int]:
    first = int(match.group(1))
    second = int(match.group(2)) if match.group(2) is not None else first
    return (min(first, second), max(first, second))


def _query_bit_ranges(query: str) -> list[tuple[int, int]]:
    ranges = [_parse_range(m) for m in _RE_BIT_RANGE.finditer(query)]

    # Byte N maps to bits (N*8+7):(N*8). This covers common register questions
    # without needing another field kind from query_processor.
    for m in _RE_BYTE_RANGE.finditer(query):
        lo_byte, hi_byte = _parse_range(m)
        ranges.append((lo_byte * 8, hi_byte * 8 + 7))

    return ranges


def _offset_range(offset: Any) -> tuple[int, int] | None:
    if offset is None:
        return None

    text = str(offset).strip()
    if not text or text.lower() in {"reserved", "variable"}:
        return None

    # Common shapes: "3", "7:4", "15 to 08", "00h", "03h:00h".
    nums = re.findall(r"(?:0x)?[0-9A-Fa-f]+h?|\d+", text)
    if not nums:
        return None

    def to_int(token: str) -> int:
        token = token.lower().replace("0x", "")
        if token.endswith("h"):
            return int(token[:-1], 16)
        return int(token, 10)

    values = [to_int(n) for n in nums[:2]]
    if len(values) == 1:
        return (values[0], values[0])
    return (min(values), max(values))


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _field_matches_bit_ranges(field_record: dict, bit_ranges: list[tuple[int, int]]) -> bool:
    if not bit_ranges:
        return True

    offset_type = str(field_record.get("offset_type") or "").lower()

    # For byte-addressed data-structure fields, "bit N of FIELD" usually means
    # a bit inside that field's value, not absolute byte offset N in the
    # structure. Keep the field and let row trimming/source context handle the
    # specific bit.
    if offset_type == "bytes":
        return True

    offset = _offset_range(field_record.get("offset"))
    if offset is None:
        return False

    return any(_ranges_overlap(offset, requested) for requested in bit_ranges)


def _dedupe_fields(fields: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for item in fields:
        key = (
            item.get("field_name"),
            item.get("parent_figure"),
            item.get("offset"),
            item.get("description"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _row_matches_bit(row: list[Any], bit_ranges: list[tuple[int, int]]) -> bool:
    if not bit_ranges or not row:
        return False
    row_range = _offset_range(row[0])
    if row_range is None:
        return False
    return any(_ranges_overlap(row_range, requested) for requested in bit_ranges)


def _trim_table_rows(
    table: dict,
    field_records: list[dict],
    bit_ranges: list[tuple[int, int]],
) -> list[list[Any]]:
    rows = table.get("rows") or []
    if not field_records:
        return rows

    field_names = {str(record.get("field_name")).upper() for record in field_records if record.get("field_name")}
    full_names = {str(record.get("full_name")).upper() for record in field_records if record.get("full_name")}

    selected: list[list[Any]] = []
    in_matching_field = False
    include_followups = False
    followup_rows = 0

    for row in rows:
        row_text = " ".join(str(cell) for cell in row).upper()
        is_field_row = any(name in row_text for name in full_names) or any(
            f"({name})" in row_text or f"{name}:" in row_text
            for name in field_names
        )

        if is_field_row:
            selected.append(row)
            in_matching_field = True
            include_followups = "BITS | DESCRIPTION" in row_text or "VALUE |" in row_text
            followup_rows = 0
            continue

        if in_matching_field and bit_ranges and _row_matches_bit(row, bit_ranges):
            selected.append(row)
            continue

        # If the caller asked for the whole field, include the first few rows
        # after the field heading because nested bit/value rows often follow it.
        if in_matching_field and include_followups and not bit_ranges and followup_rows < 12:
            selected.append(row)
            followup_rows += 1
            continue

        if in_matching_field and len(row) >= 4:
            in_matching_field = False
            include_followups = False

    return selected or rows


def _table_summary(table: dict, field_records: list[dict], bit_ranges: list[tuple[int, int]]) -> dict:
    return {
        "figure_number": str(table.get("figure_number")),
        "caption": table.get("caption"),
        "parent_section": table.get("parent_section"),
        "pdf_page": table.get("pdf_page"),
        "headers": table.get("headers") or [],
        "rows": _trim_table_rows(table, field_records, bit_ranges),
    }


def _source_from_table(
    table: dict,
    field_records: list[dict],
    bit_ranges: list[tuple[int, int]],
) -> dict:
    figure = str(table.get("figure_number"))
    section_id = table.get("parent_section")
    trimmed = dict(table)
    trimmed["rows"] = _trim_table_rows(table, field_records, bit_ranges)
    return {
        "chunk_id": f"fig{figure}__{section_id}" if section_id else f"fig{figure}",
        "section_id": section_id,
        "section_title": table.get("caption"),
        "content_type": "table",
        "text_raw": serialize_table(trimmed),
        "pdf_pages": [table.get("pdf_page")] if table.get("pdf_page") else [],
        "figure_number": figure,
        "has_normative": "shall" in (table.get("raw_text") or "").lower(),
        "method": "structured_lookup",
    }


def _confidence(field_count: int, table_count: int, bit_ranges: list[tuple[int, int]]) -> str:
    if field_count == 1 and (not bit_ranges or table_count > 0):
        return "HIGH"
    if field_count > 0 or table_count > 0:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Value-keyed enumeration lookup
#
# Some questions key off a *value* inside an enumeration table rather than a
# field name or figure number — e.g. "which feature corresponds to FID 17h?"
# (Feature Identifiers), "what command is opcode 0Dh?" (Opcodes), "what log
# page is LID 02h?" (Log Page Identifiers), CNS values, status codes. The
# field-name / figure paths never reach those rows, so without this the query
# falls through to fuzzy hybrid retrieval and returns a partial answer. Here we
# map a value entity + a concept (entity kind or query keyword) to the relevant
# enumeration table(s) and pull the exact matching row. Each hit becomes a
# synthetic field record whose name lets the normal _table_summary /
# _source_from_table machinery below trim the table down to just that row.
# ---------------------------------------------------------------------------

# (concept id, entity kinds that imply this concept, query-keyword regex,
#  table-caption regex). The concept id keys the pre-computed enum index
# (src.enum_tables); the caption regex drives the live-scan fallback. Keep the
# caption patterns in sync with src.enum_tables.CONCEPTS.
_ENUM_CONCEPT_DEFS: list[tuple[str, set[str], "re.Pattern[str]", "re.Pattern[str]"]] = [
    ("fid",    {"fid"},    re.compile(r"\bfeature(?:s)?\b|\bFID\b", re.I),
                           re.compile(r"feature identifiers", re.I)),
    ("opcode", {"opcode"}, re.compile(r"\bopcodes?\b|\bcommand\b", re.I),
                           re.compile(r"opcodes for", re.I)),
    ("lid",    {"lid"},    re.compile(r"\blog\s+page\b|\blog\s+identifier|\bLID\b", re.I),
                           re.compile(r"log page identifiers", re.I)),
    ("cns",    {"cns"},    re.compile(r"\bCNS\b", re.I),
                           re.compile(r"CNS values", re.I)),
    ("status", {"status"}, re.compile(r"\bstatus\s+(?:code|value)", re.I),
                           re.compile(r"status code", re.I)),
]

# Scan-path view: (trigger_kinds, kw_re, caption_re) — unchanged shape.
_ENUM_CONCEPTS: list[tuple[set[str], "re.Pattern[str]", "re.Pattern[str]"]] = [
    (kinds, kw, cap) for (_c, kinds, kw, cap) in _ENUM_CONCEPT_DEFS
]
# Index-path view: concept id → (trigger_kinds, kw_re).
_ENUM_CONCEPT_GATES: dict[str, tuple[set[str], "re.Pattern[str]"]] = {
    c: (kinds, kw) for (c, kinds, kw, _cap) in _ENUM_CONCEPT_DEFS
}

# A bare hex token used as an enumeration value: "17h", "0x17" (1–4 hex digits).
_RE_HEX_CELL = re.compile(r"^(?:0[xX][0-9A-Fa-f]{1,4}|[0-9A-Fa-f]{1,4}h)$")


def _hex_value(token: str) -> int | None:
    """Parse "17h" / "0x17" / "17" → int. Returns None for non-hex tokens."""
    t = str(token).strip().lower()
    if t.startswith("0x"):
        t = t[2:]
    if t.endswith("h"):
        t = t[:-1]
    if t and re.fullmatch(r"[0-9a-f]+", t):
        try:
            return int(t, 16)
        except ValueError:
            return None
    return None


# Re-derive the value from an already-extracted enum entity by consuming its
# keyword and capturing the value that follows. One pattern per kind, mirroring
# the extractor in query_processor. We anchor on the keyword (rather than just
# grabbing the trailing hex run) for two reasons:
#   * a keyword whose final letter is itself a hex digit must not bleed into a
#     no-space value — "FID2" is FID 2, never "D2" (0xD2); and
#   * the value can lead with a hex letter ("FID c0", "LID ff"), so it can't be
#     required to start with a digit / 0x / trailing-h.
# Once the keyword has pinned the entity the value is parsed permissively and
# always as hex (see _hex_value): "2" == "02" == "2h" == "0x2" == 0x02.
_VALUE = r"((?:0[xX])?[0-9A-Fa-f]+h?)"
_EMBEDDED_VALUE_RE: dict[str, "re.Pattern[str]"] = {
    "fid":    re.compile(rf"(?:FID|Feature\s+Identifier)\s*[:=]?\s*{_VALUE}", re.I),
    "lid":    re.compile(rf"(?:LID|Log\s+Page\s+Identifier)\s*[:=]?\s*{_VALUE}", re.I),
    "opcode": re.compile(rf"(?:opcode|op\s*code)\s*[:=]?\s*{_VALUE}", re.I),
    "cns":    re.compile(rf"CNS(?:\s+values?)?\s*[:=]?\s*{_VALUE}", re.I),
    "status": re.compile(rf"status(?:\s+(?:code|value)s?)?\s*[:=]?\s*{_VALUE}", re.I),
}


def _value_tokens(entities: list[Entity | dict]) -> set[int]:
    """Integer values requested by value-keyed enum entities (`fid`/`lid`/
    `opcode`/`cns`/`status`/`hex`).

    Always hex: "FID 17h", "FID 17", "LID 22", "opcode 2", and "0x17" all parse
    to their hexadecimal value (FID 22 → 0x22 → 34), never decimal. Independent
    of spelling: "FID c0" / "FID 0xc0" / "FID c0h" all resolve to 0xC0.
    """
    vals: set[int] = set()
    for raw in entities:
        ent = _entity_to_dict(raw)
        kind = ent["kind"]
        if kind == "hex":
            v = _hex_value(ent["text"])
        elif kind in _EMBEDDED_VALUE_RE:
            m = _EMBEDDED_VALUE_RE[kind].search(ent["text"])
            # Keyword-anchored capture covers the normal "FID c0" / "opcode 0Dh"
            # entities; the fallback parses a bare value should an entity ever
            # arrive without its keyword (e.g. constructed directly).
            v = _hex_value(m.group(1)) if m else _hex_value(ent["text"])
        else:
            continue
        if v is not None:
            vals.add(v)
    return vals


def _cell_value(cell: Any) -> int | None:
    """Parse a table cell to an int only if it is a bare hex token like '17h'."""
    text = str(cell).strip()
    return _hex_value(text) if _RE_HEX_CELL.match(text) else None


def _row_label(row: list[Any]) -> str:
    """The most name-like cell of a row (the one with the most letters)."""
    best, best_letters = "", 0
    for cell in row:
        s = str(cell).strip()
        letters = sum(c.isalpha() for c in s)
        if letters > best_letters:
            best, best_letters = s, letters
    return best


def _enum_value_matches(
    entities: list[Entity | dict],
    query: str,
    tables_by_figure: dict[str, dict],
) -> list[dict]:
    """Synthetic field records for value-in-enumeration-table hits (see above)."""
    values = _value_tokens(entities)
    if not values:
        return []

    kinds = {_entity_to_dict(e)["kind"] for e in entities}
    caption_res = [
        cap_re for trigger_kinds, kw_re, cap_re in _ENUM_CONCEPTS
        if (kinds & trigger_kinds) or kw_re.search(query)
    ]
    if not caption_res:
        return []

    matches: list[dict] = []
    for fig, table in tables_by_figure.items():
        caption = str(table.get("caption") or "")
        if not any(cap_re.search(caption) for cap_re in caption_res):
            continue
        for row in (table.get("rows") or []):
            if not isinstance(row, (list, tuple)):
                continue
            row_vals = {v for v in (_cell_value(c) for c in row) if v is not None}
            hit = values & row_vals
            if not hit:
                continue
            label = _row_label(list(row))
            if not label:
                continue
            value_hex = f"{min(hit):02X}h"
            matches.append({
                "field_name": label,
                "full_name": label,
                "parent_figure": str(fig),
                "offset": value_hex,
                "offset_type": "value",
                "value": value_hex,
                "description": f"{label} ({caption}, value {value_hex})",
                "source": "enum_lookup",
            })
    return matches


def _enum_index_hits(
    entities: list[Entity | dict],
    query: str,
    enum_index: dict[str, dict],
) -> list[dict]:
    """Deterministic value→name hits from the pre-computed enum index.

    For each requested value (always hex — "FID 22" and "FID 22h" both → 0x22),
    return the matching entries from every concept the query/entities trigger
    (e.g. a `fid` entity or the word "feature" → the `fid` concept). Each hit is
    the raw index entry plus its `concept`/`label`. Returns ``[]`` when the index
    is empty or nothing matches, so the caller falls back to the live scan.
    """
    if not enum_index:
        return []
    values = _value_tokens(entities)
    if not values:
        return []

    kinds = {_entity_to_dict(e)["kind"] for e in entities}
    triggered = {
        concept
        for concept, (trigger_kinds, kw_re) in _ENUM_CONCEPT_GATES.items()
        if (kinds & trigger_kinds) or kw_re.search(query)
    }
    if not triggered:
        return []

    hits: list[dict] = []
    for concept in triggered:
        block = enum_index.get(concept)
        if not block:
            continue
        label = block.get("label") or concept.upper()
        for entry in block.get("entries") or []:
            if entry.get("value") in values:
                hits.append({**entry, "concept": concept, "label": label})
    return hits


def _enum_hit_to_field(hit: dict, figure: str) -> dict:
    """Synthetic field record (one per parent figure) so the hit flows through
    the normal _table_summary / _source_from_table row-trimming machinery."""
    value_hex = hit.get("value_hex") or ""
    name = hit.get("name") or ""
    label = hit.get("label") or ""
    return {
        "field_name": name,
        "full_name": name,
        "parent_figure": str(figure),
        "offset": value_hex,
        "offset_type": "value",
        "value": value_hex,
        "description": f"{name} ({label} {value_hex})",
        "source": "enum_index",
    }


def _enum_hit_to_source(hit: dict) -> dict:
    """Self-contained source chunk for an enum hit — carries the value→name
    answer directly, so it surfaces even when the parent figure table isn't
    loaded into the retriever (the guaranteed-hit path)."""
    value_hex = hit.get("value_hex") or ""
    name = hit.get("name") or ""
    label = hit.get("label") or ""
    figures = hit.get("figures") or []
    sections = hit.get("sections") or []
    parts = [f"{label} {value_hex}: {name}."]
    if sections:
        parts.append(f"Defined in section {', '.join(sections)}.")
    if figures:
        parts.append(f"Listed in Figure {', '.join(figures)}.")
    text = " ".join(parts)
    section_id = sections[0] if sections else None
    return {
        "chunk_id": f"enum:{hit.get('concept')}:{value_hex}:{name}".replace(" ", "_"),
        "section_id": section_id,
        "section_title": f"{label} {value_hex} — {name}",
        "content_type": "table",
        "text_raw": text,
        "pdf_pages": [],
        "figure_number": figures[0] if figures else None,
        "has_normative": False,
        "score": 1.0,
        "method": "structured_lookup",
    }


# ---------------------------------------------------------------------------
# Fuzzy full-name fallback
#
# The exact field-index lookup above is the *specific* path: an acronym only
# ever resolves to its own exact record. This is the *wide* path, tried only
# after the exact tables come up empty (specific-then-wide). It matches the
# query's descriptive wording against field *full names* — never acronyms — so
# a misspelled or paraphrased name ("controller ready timeout") can still reach
# its field, while an acronym near-miss can never cross over (CRATT ↛ CRAT).
# ---------------------------------------------------------------------------

# A descriptive word: letters first, then alphanumerics/hyphens. Used to tokenize
# both the query and full names into a comparable bag of words.
_RE_WORD = re.compile(r"[A-Za-z][A-Za-z0-9-]*")

# Generic words too common to anchor a full-name match. Kept intentionally small
# so real name words (e.g. "data", "pointer") survive.
_FUZZY_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "is", "are",
    "what", "whats", "which", "how", "does", "do", "mean", "means", "this",
    "that", "with", "when", "value", "field", "register", "offset", "bit",
    "bits", "byte", "bytes",
})


def _fuzzy_full_name_matches(
    query: str,
    field_index: dict[str, list[dict]],
    name_index: tuple[tuple[str, tuple[str, ...]], ...],
    *,
    cutoff: float,
    max_hits: int,
) -> tuple[list[dict], list[str]]:
    """Fuzzy-match the query's wording against field full names. See section note.

    For each candidate full name, slide a word-window of *that name's own length*
    across the query tokens and keep the best ``SequenceMatcher`` ratio; a name
    scores a hit only at/above ``cutoff``. Pinning the window to the name's word
    count means a short query token can never fuzzily match a long name, and the
    name_index already excludes single-word names — together these guarantee
    acronyms are never fuzzed into one another. Returns (records, notes); each
    record is tagged ``source="fuzzy_full_name"`` with its ``fuzzy_score``.
    """
    tokens = [t for t in _RE_WORD.findall(query.lower()) if t not in _FUZZY_STOPWORDS]
    if not tokens or not name_index:
        return [], []

    sm = difflib.SequenceMatcher()
    scored: list[tuple[float, str, tuple[str, ...]]] = []
    for norm_name, acronyms in name_index:
        n_words = norm_name.count(" ") + 1
        if n_words > len(tokens):
            continue
        sm.set_seq2(norm_name)
        best = 0.0
        for i in range(len(tokens) - n_words + 1):
            sm.set_seq1(" ".join(tokens[i:i + n_words]))
            if sm.quick_ratio() < cutoff:  # cheap upper bound — skip the real ratio
                continue
            best = max(best, sm.ratio())
        if best >= cutoff:
            scored.append((best, norm_name, acronyms))

    if not scored:
        return [], []

    scored.sort(key=lambda x: x[0], reverse=True)
    records: list[dict] = []
    notes: list[str] = []
    for score, norm_name, acronyms in scored[:max_hits]:
        for acr in acronyms:
            for rec in field_index.get(acr, []):
                records.append({**rec, "source": "fuzzy_full_name", "fuzzy_score": round(score, 3)})
        notes.append(f"Fuzzy full-name match: {norm_name!r} → {'/'.join(acronyms)} (score {score:.2f}).")
    return records, notes


def structured_lookup(
    query_or_decomposition: str | QueryDecomposition | dict,
    *,
    use_llm: bool = False,
    max_fields: int = 8,
    spec: str = DEFAULT_SPEC,
    enable_fuzzy: bool = True,
    fuzzy_cutoff: float = 0.86,
) -> StructuredLookupResult:
    """
    Return exact field/table evidence for lookup-style queries, scoped to `spec`.

    Pass either a raw query string or the output of `process_query()`. For raw
    strings, `use_llm=False` by default so this can run cheaply inside a web
    request before falling back to hybrid retrieval.
    """
    if isinstance(query_or_decomposition, str):
        decomp = process_query(query_or_decomposition, use_llm=use_llm)
        query = decomp.query
        entities = [_entity_to_dict(e) for e in decomp.entities]
    elif isinstance(query_or_decomposition, QueryDecomposition):
        query = query_or_decomposition.query
        entities = [_entity_to_dict(e) for e in query_or_decomposition.entities]
    else:
        query = str(query_or_decomposition.get("query", "")).strip()
        entities = [_entity_to_dict(e) for e in query_or_decomposition.get("entities", [])]

    if not query:
        raise ValueError("query is empty")

    field_index = load_field_index(spec)
    tables_by_figure = load_tables_by_figure(spec)

    bit_ranges = _query_bit_ranges(query)
    field_keys = _field_keys_from_entities(entities)
    figure_keys = _figures_from_entities(entities)
    notes: list[str] = []

    field_matches: list[dict] = []
    for key in field_keys:
        records = field_index.get(key, [])
        if not records:
            notes.append(f"No field_index match for {key}.")
            continue

        narrowed = [r for r in records if _field_matches_bit_ranges(r, bit_ranges)]
        if bit_ranges and not narrowed:
            notes.append(f"{key} matched, but no record overlapped requested bit/byte range.")
            continue
        field_matches.extend(narrowed or records)

    # Wide-net fallback (runs *after* the exact lookup tables above). Only when
    # no acronym resolved exactly do we fuzzy-match the query's descriptive
    # wording against field full names — acronyms always stay exact, so this can
    # never cross one acronym to another. See _fuzzy_full_name_matches.
    if enable_fuzzy and not field_matches:
        fuzzy_records, fuzzy_notes = _fuzzy_full_name_matches(
            query,
            field_index,
            _full_name_index(spec),
            cutoff=fuzzy_cutoff,
            max_hits=max_fields,
        )
        field_matches.extend(fuzzy_records)
        notes.extend(fuzzy_notes)

    # Value-keyed enumeration hits (e.g. FID 22h → "Configurable Device
    # Personality"). Prefer the deterministic pre-computed enum index; fall back
    # to a live caption-scan of the raw tables when the index isn't available so
    # nothing regresses. `enum_sources` are self-contained chunks that carry the
    # value→name answer directly, guaranteeing it surfaces even if the parent
    # figure table isn't loaded into the retriever. Prepended so the exact
    # value→name answer ranks ahead of generic field-name records.
    enum_index = load_enum_index(spec)
    enum_hits = _enum_index_hits(entities, query, enum_index)
    enum_sources: list[dict] = []
    if enum_hits:
        enum_matches = []
        for hit in enum_hits:
            for fig in (hit.get("figures") or []):
                enum_matches.append(_enum_hit_to_field(hit, fig))
            enum_sources.append(_enum_hit_to_source(hit))
    else:
        enum_matches = _enum_value_matches(entities, query, tables_by_figure)
    field_matches = _dedupe_fields(enum_matches + field_matches)[:max_fields]

    table_numbers: list[str] = []
    for field_record in field_matches:
        fig = field_record.get("parent_figure")
        if fig is not None and str(fig) not in table_numbers:
            table_numbers.append(str(fig))
    for fig in figure_keys:
        if fig not in table_numbers:
            table_numbers.append(fig)

    fields_by_figure: dict[str, list[dict]] = {}
    for field_record in field_matches:
        fig = field_record.get("parent_figure")
        if fig is not None:
            fields_by_figure.setdefault(str(fig), []).append(field_record)

    tables = [
        _table_summary(tables_by_figure[fig], fields_by_figure.get(fig, []), bit_ranges)
        for fig in table_numbers
        if fig in tables_by_figure
    ]
    sources = enum_sources + [
        _source_from_table(tables_by_figure[fig], fields_by_figure.get(fig, []), bit_ranges)
        for fig in table_numbers
        if fig in tables_by_figure
    ]

    if figure_keys and not sources and not field_matches:
        notes.append("Figure entity was extracted, but no matching table was found.")
    if not field_keys and not figure_keys and not enum_matches:
        notes.append("No field or figure entity was extracted for structured lookup.")

    found = bool(field_matches or tables or enum_sources)
    return StructuredLookupResult(
        query=query,
        found=found,
        confidence=_confidence(len(field_matches), len(tables), bit_ranges),
        entities=entities,
        fields=field_matches,
        tables=tables,
        sources=sources,
        notes=notes,
    )


def rrf_merge(
    result_lists: list[list[dict]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[dict]:
    """
    Reciprocal Rank Fusion merge of independent ranked result lists.

    For each document, sum 1/(k + rank) across every list it appears in
    (rank is 1-indexed). Documents are deduped by `id`; metadata from the
    first occurrence is preserved. Use this to fuse vector + BM25 results,
    or to combine results across decomposed sub-queries.

    Args:
        result_lists: ranked result lists shaped like search.py output
            (dicts with `id` and optional `method`).
        k: RRF constant. Cormack et al.'s classic value is 60; lower values
            sharpen the lead of top-ranked items.
        top_k: if set, return only the top_k merged results.

    Returns:
        Merged list sorted by RRF score (descending). Each result gains:
          - `rrf_score`: float
          - `method`: "rrf"
          - `contributing_methods`: source methods that surfaced this id
          - `ranks`: {method: best_rank_seen}
    """
    if not result_lists:
        return []

    scores: dict[Any, float] = {}
    representative: dict[Any, dict] = {}
    methods: dict[Any, list[str]] = {}
    ranks: dict[Any, dict[str, int]] = {}

    def _doc_key(item: dict) -> Any:
        """Stable per-document key. Falls back to a content-derived
        surrogate so id-less chunks (e.g. structured_lookup synthetic rows)
        don't all collide under a shared None key and get dropped."""
        doc_id = item.get("id") or item.get("chunk_id")
        if doc_id:
            return doc_id
        return (
            "__no_id__",
            item.get("section_id"),
            item.get("figure_number"),
            item.get("content_type"),
            (item.get("text_raw") or "")[:120],
        )

    for results in result_lists:
        seen: set = set()
        for rank, item in enumerate(results, start=1):
            doc_id = _doc_key(item)
            if doc_id in seen:
                continue
            seen.add(doc_id)

            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

            # Surface upstream gaps loudly: an item missing a method field
            # is a contract violation in search.py / structured_lookup, not
            # something to paper over with "unknown".
            method = item.get("method")
            if not method:
                method = "missing_method"
            method = str(method)

            method_list = methods.setdefault(doc_id, [])
            if method not in method_list:
                method_list.append(method)

            rank_map = ranks.setdefault(doc_id, {})
            if method not in rank_map or rank < rank_map[method]:
                rank_map[method] = rank

            if doc_id not in representative:
                representative[doc_id] = item

    merged: list[dict] = []
    for doc_id, score in scores.items():
        base = dict(representative[doc_id])
        base["rrf_score"] = score
        base["method"] = "rrf"
        base["contributing_methods"] = methods[doc_id]
        base["ranks"] = ranks[doc_id]
        merged.append(base)

    merged.sort(key=lambda r: r["rrf_score"], reverse=True)
    if top_k is not None:
        merged = merged[:top_k]
    return merged


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 structured lookup retriever.")
    sub = parser.add_subparsers(dest="mode", required=True)

    structured = sub.add_parser("structured", help="run exact field/table lookup")
    structured.add_argument("query", nargs="+")
    structured.add_argument(
        "--llm",
        action="store_true",
        help="Use query_processor's LLM classification before structured lookup.",
    )
    structured.add_argument("--max-fields", type=int, default=8)

    args = parser.parse_args(argv)
    query = " ".join(args.query)

    if args.mode == "structured":
        result = structured_lookup(query, use_llm=args.llm, max_fields=args.max_fields)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
