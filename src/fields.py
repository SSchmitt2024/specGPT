"""
NVMe field registry extractor.

Reads tables.json and extracts every named field from data-structure,
register, and command-format tables into a structured fields.json.

The NVMe spec defines fields with a consistent pattern in the Description
column of tables:

    Full Name (ABBREV): description text...

This script:
  1. Classifies each table by its header shape (register, data_structure,
     command_format, or other).
  2. For qualifying tables, regex-extracts every named field from the
     description column.
  3. Parses inline value enumerations (markdown pipe tables).
  4. Captures requirement columns (M/O/R/P) per context.
  5. Extracts cross-references to sections and figures.
  6. Writes data/fields.json (list) and data/field_index.json (abbrev->list).
"""

from __future__ import annotations

import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Constants

# Matches "Full Name (ABBREV):" at the start of a description cell.
# Captures: group(1) = full name, group(2) = abbreviation
FIELD_NAME_RE = re.compile(
    r"^(.+?)\s*\(([A-Z][A-Z0-9/:]+(?:\.[A-Z][A-Z0-9]*)*)\)\s*:"
)

# Matches "refer to section X.Y.Z" or "Refer to Figure N"
SECTION_REF_RE = re.compile(r"[Rr]efer to section\s+([\d.]+)")
FIGURE_REF_RE = re.compile(r"[Rr]efer to Figure\s+(\d+)")

# Matches a markdown pipe-table row: "| val | meaning |"
PIPE_ROW_RE = re.compile(r"^\|\s*(.+?)\s*\|\s*(.+?)\s*\|$")

# Matches separator row: "|---|---|"
PIPE_SEP_RE = re.compile(r"^\|[-\s|]+\|$")


# ---------------------------------------------------------------------------
# Table classification


def classify_table(headers: list[str], caption: str | None) -> str:
    """
    Classify a table based on its headers.

    Returns one of:
      - "register"        : Bits + Type + Reset + Description
      - "data_structure"  : Bytes + requirement columns + Description
      - "command_format"  : Bytes + Description (no requirement columns)
      - "other"           : everything else
    """
    if not headers:
        return "other"

    h_lower = [h.lower().strip() for h in headers]

    # Register tables: have "bits" and "type" and "description"
    has_bits = "bits" in h_lower
    has_description = any("description" in h or "definition" in h for h in h_lower)

    if has_bits and has_description:
        if "type" in h_lower or "reset" in h_lower:
            return "register"
        # Bits + Description only (no Type/Reset) — CDW bit-field tables
        # e.g., "Command Dword 0", completion queue entry dwords
        return "command_format"

    # Data structure tables: have "bytes" and at least one M/O/R column
    has_bytes = any("byte" in h for h in h_lower)

    if has_bytes and has_description:
        # Check for requirement columns — they contain controller type names
        # or are single-letter headers
        req_keywords = ["i/o", "admin", "disc", "controller"]
        has_req = any(
            any(kw in h for kw in req_keywords)
            for h in h_lower
            if h not in ("bytes", "description", "definition")
        )
        if has_req:
            return "data_structure"
        return "command_format"

    # Also catch "Offset" based tables (like property tables)
    has_offset = any("offset" in h for h in h_lower)
    if has_offset and has_description:
        return "command_format"

    return "other"


# ---------------------------------------------------------------------------
# Requirement column extraction


def _extract_requirements(
    row: list[str], headers: list[str], table_type: str
) -> dict[str, str]:
    """
    Extract requirement indicators (M/O/R/P) from a data_structure row.
    Maps header names to the cell values.
    """
    if table_type != "data_structure":
        return {}

    reqs = {}
    # The description is always the last column, bytes is the first.
    # Requirement columns are everything in between.
    if len(headers) < 3 or len(row) < 3:
        return {}

    for i in range(1, len(headers) - 1):
        if i >= len(row):
            break
        val = row[i].strip()
        if val and val in ("M", "O", "R", "P", "M/O", "O/R", "M/P"):
            header_name = headers[i].strip()
            reqs[header_name] = val

    return reqs


# ---------------------------------------------------------------------------
# Register-specific metadata


def _extract_register_meta(
    row: list[str], headers: list[str], table_type: str
) -> tuple[str | None, str | None]:
    """
    For register tables, extract the Type (RO/RW/etc) and Reset columns.
    Returns (type_val, reset_val).
    """
    if table_type != "register":
        return None, None

    h_lower = [h.lower().strip() for h in headers]
    type_val = None
    reset_val = None

    if "type" in h_lower:
        idx = h_lower.index("type")
        if idx < len(row):
            type_val = row[idx].strip() or None

    if "reset" in h_lower:
        idx = h_lower.index("reset")
        if idx < len(row):
            reset_val = row[idx].strip() or None

    return type_val, reset_val


# ---------------------------------------------------------------------------
# Value enumeration parsing


def _parse_inline_values(description: str) -> dict[str, str]:
    """
    Parse inline markdown pipe-table value enumerations from description text.

    Looks for patterns like:
        | Value | Definition |
        |---|---|
        | 00b | Normal operation |
        | 01b | First command |

    Returns {code: meaning} dict.
    """
    values = {}
    lines = description.split("\n")
    in_table = False
    past_header = False

    for line in lines:
        line_s = line.strip()
        if not line_s:
            if in_table and past_header:
                # Empty line after table body — table ended
                break
            continue

        if PIPE_SEP_RE.match(line_s):
            in_table = True
            past_header = True
            continue

        m = PIPE_ROW_RE.match(line_s)
        if m:
            if not in_table:
                # This is the header row
                in_table = True
                continue
            if past_header:
                key = m.group(1).strip()
                val = m.group(2).strip()
                if key and val:
                    values[key] = val

    return values


# ---------------------------------------------------------------------------
# Cross-reference extraction


def _extract_cross_refs(description: str) -> list[dict[str, str]]:
    """Extract section and figure cross-references from description text."""
    refs = []
    seen = set()

    for m in SECTION_REF_RE.finditer(description):
        sid = m.group(1).rstrip(".")
        key = ("section", sid)
        if key not in seen:
            refs.append({"type": "section", "id": sid})
            seen.add(key)

    for m in FIGURE_REF_RE.finditer(description):
        fid = m.group(1)
        key = ("figure", fid)
        if key not in seen:
            refs.append({"type": "figure", "id": fid})
            seen.add(key)

    return refs


# ---------------------------------------------------------------------------
# Description cleaning — strip the value table out of the prose


def _clean_description(description: str) -> str:
    """
    Return the prose portion of a description, stripping out inline
    markdown pipe tables.
    """
    lines = description.split("\n")
    out = []
    in_table = False

    for line in lines:
        line_s = line.strip()
        if PIPE_SEP_RE.match(line_s) or PIPE_ROW_RE.match(line_s):
            in_table = True
            continue
        if in_table and not line_s:
            in_table = False
            continue
        if not in_table:
            out.append(line)

    return "\n".join(out).strip()


# ---------------------------------------------------------------------------
# Main extraction


def extract_fields(tables: list[dict]) -> list[dict]:
    """
    Walk all tables and extract named fields.

    Returns a list of field records.
    """
    fields = []

    for table in tables:
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        caption = table.get("caption")
        figure_number = table.get("figure_number")
        printed_page = table.get("printed_page")

        table_type = classify_table(headers, caption)
        if table_type == "other":
            continue

        # Find the description column index
        desc_idx = -1
        h_lower = [h.lower().strip() for h in headers]
        for i, h in enumerate(h_lower):
            if "description" in h or "definition" in h:
                desc_idx = i
                break

        if desc_idx == -1:
            # Fallback: last column
            desc_idx = len(headers) - 1

        # Find the offset column index (first column, usually)
        offset_idx = 0

        for row in rows:
            if not row or len(row) <= desc_idx:
                continue

            desc = row[desc_idx]
            if not desc or not isinstance(desc, str):
                continue

            # Try to match the field name pattern
            m = FIELD_NAME_RE.match(desc)
            if not m:
                continue

            full_name = m.group(1).strip()
            abbreviation = m.group(2).strip()

            # Get the offset (byte or bit range)
            offset = row[offset_idx].strip() if row[offset_idx] else None

            # Determine offset type from headers
            offset_type = None
            if headers:
                first_h = headers[0].lower().strip()
                if "bit" in first_h:
                    offset_type = "bits"
                elif "byte" in first_h:
                    offset_type = "bytes"
                elif "offset" in first_h:
                    offset_type = "offset"

            # Extract requirements
            requirements = _extract_requirements(row, headers, table_type)

            # Extract register metadata
            reg_type, reg_reset = _extract_register_meta(row, headers, table_type)

            # Parse inline value enumerations
            values = _parse_inline_values(desc)

            # Extract cross-references
            cross_refs = _extract_cross_refs(desc)

            # Clean description (strip value tables)
            clean_desc = _clean_description(desc)
            # Remove the "Full Name (ABBREV): " prefix from the clean description
            prefix_end = desc.index(":") + 1
            after_prefix = desc[prefix_end:].strip()
            clean_desc = _clean_description(after_prefix)

            field_record = {
                "field_name": abbreviation,
                "full_name": full_name,
                "parent_figure": figure_number,
                "parent_caption": caption,
                "parent_type": table_type,
                "offset": offset,
                "offset_type": offset_type,
                "requirements": requirements if requirements else None,
                "register_type": reg_type,
                "register_reset": reg_reset,
                "values": values if values else None,
                "cross_refs": cross_refs if cross_refs else None,
                "description": clean_desc,
                "spec_page": printed_page,
            }

            fields.append(field_record)

    return fields


REGISTER_CAPTION_RE = re.compile(
    r"^(?:Figure\s+\d+\s*[:\-\u2013\u2014]\s*)?"
    r"Offset\s+([0-9A-Fa-f]+h)\s*[:\-\u2013\u2014\uFFFD?]\s*"
    r"([A-Z][A-Z0-9]+)\s*[\-\u2013\u2014\uFFFD?]\s*"
    r"(.+)$"
)


def synthesize_register_containers(tables: list[dict]) -> list[dict]:
    """
    Scan table captions for register containers (``Offset Xh: ABBR – Full Name``)
    and emit one pseudo-field record per unique register. Lets callers resolve
    ``idx['CAP']`` to the container itself, not just its sub-fields.
    """
    records: list[dict] = []
    seen: set[str] = set()
    for table in tables:
        cap = (table.get("caption") or "").strip()
        m = REGISTER_CAPTION_RE.match(cap)
        if not m:
            continue
        offset, abbr, full_name = m.group(1), m.group(2), m.group(3).strip()
        if abbr in seen:
            continue
        seen.add(abbr)
        records.append({
            "field_name": abbr,
            "full_name": full_name,
            "parent_figure": table.get("figure_number"),
            "parent_caption": cap,
            "parent_type": "register_container",
            "offset": offset,
            "offset_type": "offset",
            "requirements": None,
            "register_type": None,
            "register_reset": None,
            "values": None,
            "cross_refs": None,
            "description": f"{full_name} register, located at offset {offset}.",
            "spec_page": table.get("printed_page"),
        })
    return records


def build_field_index(fields: list[dict]) -> dict[str, list[dict]]:
    """
    Build a reverse index: abbreviation -> list of field records.
    Some abbreviations appear in multiple structures.
    """
    index: dict[str, list[dict]] = {}
    for f in fields:
        abbr = f["field_name"]
        if abbr not in index:
            index[abbr] = []
        index[abbr].append(f)
    return index


# ---------------------------------------------------------------------------
# CLI

if __name__ == "__main__":
    TABLES_PATH = "data/tables.json"
    FIELDS_PATH = "data/fields.json"
    INDEX_PATH = "data/field_index.json"

    with open(TABLES_PATH, "r", encoding="utf-8") as f:
        tables = json.load(f)

    print(f"loaded {len(tables)} tables")

    fields = extract_fields(tables)
    print(f"extracted {len(fields)} field definitions")

    register_containers = synthesize_register_containers(tables)
    fields.extend(register_containers)
    print(f"synthesized {len(register_containers)} register-container entries")

    # Stats
    by_type = {}
    for f in fields:
        t = f["parent_type"]
        by_type[t] = by_type.get(t, 0) + 1
    for t, c in sorted(by_type.items()):
        print(f"  {t}: {c}")

    with_values = sum(1 for f in fields if f["values"])
    with_refs = sum(1 for f in fields if f["cross_refs"])
    with_reqs = sum(1 for f in fields if f["requirements"])
    print(f"  with value enums: {with_values}")
    print(f"  with cross-refs:  {with_refs}")
    print(f"  with requirements: {with_reqs}")

    # Sample output
    if fields:
        print("\nsample fields:")
        shown = 0
        for f in fields:
            print(f"  {f['field_name']:20s} ({f['full_name'][:40]:40s}) in Figure {f['parent_figure']} [{f['parent_type']}]")
            shown += 1
            if shown >= 15:
                break

    # Build index
    index = build_field_index(fields)
    multi = sum(1 for v in index.values() if len(v) > 1)
    print(f"\nunique abbreviations: {len(index)}")
    print(f"abbreviations in multiple structures: {multi}")

    # Write outputs
    os.makedirs(os.path.dirname(FIELDS_PATH), exist_ok=True)
    with open(FIELDS_PATH, "w", encoding="utf-8") as f:
        json.dump(fields, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {FIELDS_PATH}")

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"wrote {INDEX_PATH}")
