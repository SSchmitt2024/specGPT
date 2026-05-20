"""
Phase 2 - Step 2.3: Retriever orchestration

Houses the structured lookup path (2.3a) and the merge primitives for the
hybrid retrieval path (2.3b).

  - `structured_lookup()` — deterministic field/register/table lookup
        query -> query_processor entities -> field_index.json -> fields.json
             -> parent table from tables.json -> compact source bundle
  - `rrf_merge()` — Reciprocal Rank Fusion across vector/BM25 result lists
        (and across sub-queries when query_processor decomposes a query)

The web backend should call `structured_lookup()` first. If it returns
`found=False`, route the query to the hybrid path that runs
`search.vector_search` + `search.bm25_search` per sub-query and combines the
ranked lists via `rrf_merge()`.

CLI:
  python -m src.pipeline.retriever structured "What does bit 3 of OACS mean?"
  python -m src.pipeline.retriever structured --no-llm "What is NSSES?"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.pipeline.query_processor import Entity, QueryDecomposition, process_query
from src.pipeline.table_serializer import serialize_table


DATA_DIR = Path("data")
FIELD_INDEX_PATH = DATA_DIR / "field_index.json"
FIELDS_PATH = DATA_DIR / "fields.json"
TABLES_PATH = DATA_DIR / "tables.json"


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


@lru_cache(maxsize=1)
def load_field_index() -> dict[str, list[dict]]:
    with open(FIELD_INDEX_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_fields() -> list[dict]:
    with open(FIELDS_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_tables_by_figure() -> dict[str, dict]:
    with open(TABLES_PATH, encoding="utf-8") as f:
        tables = json.load(f)
    return {str(t.get("figure_number")): t for t in tables if t.get("figure_number") is not None}


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


def structured_lookup(
    query_or_decomposition: str | QueryDecomposition | dict,
    *,
    use_llm: bool = False,
    max_fields: int = 8,
) -> StructuredLookupResult:
    """
    Return exact field/table evidence for lookup-style queries.

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

    field_index = load_field_index()
    tables_by_figure = load_tables_by_figure()

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

    field_matches = _dedupe_fields(field_matches)[:max_fields]

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
    sources = [
        _source_from_table(tables_by_figure[fig], fields_by_figure.get(fig, []), bit_ranges)
        for fig in table_numbers
        if fig in tables_by_figure
    ]

    if figure_keys and not sources and not field_matches:
        notes.append("Figure entity was extracted, but no matching table was found.")
    if not field_keys and not figure_keys:
        notes.append("No field or figure entity was extracted for structured lookup.")

    found = bool(field_matches or tables)
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

    for results in result_lists:
        seen: set = set()
        for rank, item in enumerate(results, start=1):
            doc_id = item.get("id")
            if doc_id is None or doc_id in seen:
                continue
            seen.add(doc_id)

            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

            method = str(item.get("method") or "unknown")
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
