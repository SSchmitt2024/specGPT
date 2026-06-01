"""
NVMe enumeration-table index extractor.

Reads tables.json and distills the spec's *value-keyed* enumeration tables —
Feature Identifiers (FID), Log Page Identifiers (LID), CNS Values, Command
Opcodes, and Status Codes — into a deterministic keyed lookup, enum_index.json.

Why this exists
---------------
The structured-lookup path already answers field-name questions ("what is
HMPRE?") from field_index.json. Value-keyed questions ("what feature is FID
22h?", "what log page is LID 02h?") used to rely on a *live scan* of every raw
table at query time, matched by caption regex. That works but is fragile and
reverse-only. This extractor builds a clean, pre-computed index so:

  * value -> name  ("FID 22h" -> "Configurable Device Personality") is O(1) and
    deterministic, independent of how messy the source rows are; and
  * name -> value  ("Power Management feature" -> 02h) becomes possible, so the
    canonical identifier is available up front, before query decomposition.

Everything here is hex. NVMe identifies FIDs / LIDs / opcodes / CNS / status by
hexadecimal value, so a bare "22" in a query means 0x22, never decimal 22 — the
parser treats every value token as hex.

Output (enum_index.json):

    {
      "fid": {
        "concept": "fid",
        "label": "Feature Identifier",
        "entries": [
          {"value": 34, "value_hex": "22h",
           "name": "Configurable Device Personality",
           "figures": ["198", "403"], "sections": ["5.2.26.1.24"]},
          ...
        ]
      },
      "lid": { ... },
      "cns": { ... },
      "opcode": { ... },
      "status": { ... }
    }

Run:
  python -m src.enum_tables
  python -m src.enum_tables --tables data/tables.json --out data/enum_index.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Concept definitions
#
# Each enumeration concept is identified by the caption of the table(s) that
# define it. `caption_re` selects the tables; `exclude_re` drops near-miss
# captions that contain the same words but are NOT the value table (e.g. the
# "Feature Identifiers Effects Log Page", the discovery-only "Allowed Log Page
# Identifiers", the "Connect Response ... Based on Status Code" dword table).
#
# These caption patterns intentionally mirror retriever._ENUM_CONCEPTS so the
# pre-computed index and the live-scan fallback agree on which tables count.
# ---------------------------------------------------------------------------
CONCEPTS: list[dict] = [
    {
        "concept": "fid",
        "label": "Feature Identifier",
        "caption_re": re.compile(r"feature identifiers", re.I),
        "exclude_re": re.compile(r"effects", re.I),
    },
    {
        "concept": "lid",
        "label": "Log Page Identifier",
        "caption_re": re.compile(r"log page identifiers", re.I),
        "exclude_re": re.compile(r"\ballowed\b", re.I),
    },
    {
        "concept": "cns",
        "label": "CNS Value",
        "caption_re": re.compile(r"CNS values", re.I),
        "exclude_re": None,
    },
    {
        "concept": "opcode",
        "label": "Command Opcode",
        "caption_re": re.compile(r"opcodes for", re.I),
        "exclude_re": None,
    },
    {
        "concept": "status",
        "label": "Status Code",
        "caption_re": re.compile(r"status code", re.I),
        "exclude_re": re.compile(r"dword 0|permitted to return", re.I),
    },
]

# A table cell that is a bare hex enumeration value: "22h", "0x22" (1–4 digits).
_RE_HEX_CELL = re.compile(r"^(?:0[xX][0-9A-Fa-f]{1,4}|[0-9A-Fa-f]{1,4}h)$")
# A dotted section identifier cell: "5.2.26.1.24". At least one dot so plain
# values / version-like tokens aren't mistaken for sections.
_RE_SECTION_CELL = re.compile(r"^\d+(?:\.\d+){1,6}$")
# PDF footnote artifacts that leak into name cells as a leading "<digit> ".
_RE_LEADING_FOOTNOTE = re.compile(r"^\s*\d+\s+")


def _hex_value(token: str) -> int | None:
    """Parse "22h" / "0x22" / "22" -> 34. Always hex. None for non-hex tokens."""
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


def _cell_value(cell: object) -> int | None:
    """Integer value of a cell, only if it is a bare hex token like '22h'."""
    text = str(cell).strip()
    return _hex_value(text) if _RE_HEX_CELL.match(text) else None


def _clean_name(text: str) -> str:
    """Strip leading PDF footnote markers ("4 Flush" -> "Flush") and whitespace."""
    return _RE_LEADING_FOOTNOTE.sub("", str(text).strip()).strip()


def _row_name(row: list, value_idx: int) -> str:
    """The most name-like cell of a row: most alphabetic characters, skipping
    the value cell and pure section-reference cells. A short colon-prefixed
    description ("Successful Completion: ...") is truncated at the colon so the
    name stays terse."""
    best, best_letters = "", 0
    for i, cell in enumerate(row):
        if i == value_idx:
            continue
        s = str(cell).strip()
        if _RE_SECTION_CELL.match(s):
            continue
        letters = sum(c.isalpha() for c in s)
        if letters > best_letters:
            best, best_letters = s, letters
    name = _clean_name(best)
    # Status/CNS definitions pack the term then a colon then prose; keep the term.
    if ":" in name:
        head = name.split(":", 1)[0].strip()
        if len(head) >= 2:
            name = head
    return name


def _row_sections(row: list) -> list[str]:
    """Dotted section identifiers found in a row (e.g. ['5.2.26.1.24'])."""
    out: list[str] = []
    for cell in row:
        s = str(cell).strip()
        if _RE_SECTION_CELL.match(s) and s not in out:
            out.append(s)
    return out


def _extract_concept(concept: dict, tables: list[dict]) -> dict:
    """Build one concept's index block from all tables whose caption matches."""
    cap_re = concept["caption_re"]
    exclude_re = concept.get("exclude_re")

    # (value:int, normalized-name) -> merged entry. Keying on the name too keeps
    # genuinely distinct meanings for the same value separate (common for status
    # codes across subtypes) while merging the same row seen in two figures
    # (e.g. a FID listed in both the Get Features and Set Features tables).
    merged: dict[tuple[int, str], dict] = {}

    for table in tables:
        caption = str(table.get("caption") or "")
        if not cap_re.search(caption):
            continue
        if exclude_re and exclude_re.search(caption):
            continue
        fig = str(table.get("figure_number")) if table.get("figure_number") is not None else None

        for row in (table.get("rows") or []):
            if not isinstance(row, (list, tuple)):
                continue
            # Locate the value cell (first bare-hex cell in the row).
            value_idx = next(
                (i for i, c in enumerate(row) if _cell_value(c) is not None), None
            )
            if value_idx is None:
                continue
            value = _cell_value(row[value_idx])
            if value is None:
                continue
            name = _row_name(list(row), value_idx)
            if not name:
                continue

            key = (value, name.lower())
            entry = merged.get(key)
            if entry is None:
                entry = {
                    "value": value,
                    "value_hex": f"{value:02X}h",
                    "name": name,
                    "figures": [],
                    "sections": [],
                }
                merged[key] = entry
            if fig and fig not in entry["figures"]:
                entry["figures"].append(fig)
            for sec in _row_sections(list(row)):
                if sec not in entry["sections"]:
                    entry["sections"].append(sec)

    entries = sorted(merged.values(), key=lambda e: (e["value"], e["name"].lower()))
    return {
        "concept": concept["concept"],
        "label": concept["label"],
        "entries": entries,
    }


def build_enum_index(tables: list[dict]) -> dict:
    """Build the full {concept: block} enum index from parsed tables.json."""
    index: dict[str, dict] = {}
    for concept in CONCEPTS:
        block = _extract_concept(concept, tables)
        if block["entries"]:
            index[concept["concept"]] = block
    return index


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Build the enum lookup index from tables.json.")
    parser.add_argument("--tables", type=Path, default=None,
                        help="Path to tables.json (default: $SPEC_DATA_DIR/tables.json).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (default: $SPEC_DATA_DIR/enum_index.json).")
    args = parser.parse_args(argv)

    if args.tables is not None:
        tables_path = args.tables
    else:
        from src import spec_env
        tables_path = Path(spec_env.data_path("tables.json"))
    if args.out is not None:
        out_path = args.out
    else:
        from src import spec_env
        out_path = Path(spec_env.data_path("enum_index.json"))

    if not tables_path.exists():
        print(f"ERROR: {tables_path} not found — run the Tables step first.", file=sys.stderr)
        return 1

    with open(tables_path, encoding="utf-8") as f:
        tables = json.load(f)

    index = build_enum_index(tables)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    total = sum(len(b["entries"]) for b in index.values())
    print(f"Wrote {out_path}: {total} entries across {len(index)} concepts")
    for concept, block in index.items():
        sample = ", ".join(
            f"{e['value_hex']}={e['name']}" for e in block["entries"][:3]
        )
        print(f"  {concept:8s} {len(block['entries']):4d} entries  e.g. {sample}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
