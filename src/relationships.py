"""
Phase 1.4 — Deterministic structural relationship extraction.

Reads the outputs of parser.py (data/toc.json) and tables.py (data/tables.json)
and emits a list of edges capturing:

  - figure ⊂ section        (containment)
  - figure → figure          (explicit cross-reference)
  - figure → section         (explicit cross-reference)
  - section → figure         (same, for prose-side edges once 1.3 lands)
  - section → section        (same)

Edge schema (flat for easy JSON consumption downstream):

    {
        "source": "figure:328",
        "target": "section:5.17",
        "type":   "contained_in",
        "evidence": "printed_page 321 falls inside section 5.17 (p.320–p.345)",
        "confidence": "deterministic"
    }

Cross-reference edges (type == "cross_reference") additionally carry:
    "strength": "strong" | "mention"
"strong" means the reference is gated by a verb like "see", "refer to",
"as defined in", etc. "mention" is a bare "Section X.Y" or "Figure N"
occurrence. 1.5 (LLM pass) can upgrade mentions later.

No prose input yet (1.3 still in flight). When prose arrives, call
`extract_from_prose(prose_blocks, section_lookup)` with a list of
{section_id, text} dicts and merge the result into the output.
"""

from __future__ import annotations

import bisect
import json
import re
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Regexes
#
# Cross-reference detection is intentionally permissive on matching and
# strict on *gating*. The goal is to catch every real reference in the spec
# without firing on stray words like "PCI Header section of the ...".
#
# Rules:
#   - Section refs must include at least one dot (e.g., "5.17" not "5"). This
#     loses top-level chapter refs ("Section 8") but avoids a flood of false
#     positives on generic "section" usage. 1.5 LLM pass can recover the rest.
#   - Figure refs require the literal word "Figure" followed by a number.
#     NVMe uses "Figure N" exclusively for table/figure references.

# "Section 5.17.2" / "clause 5.17.2" / "sections 5.17.2 and 5.18"
# Requires at least one dot in the number to keep precision high.
SECTION_REF_RE = re.compile(
    r"\b(?:section|clause)s?\s+(\d+(?:\.\d+)+[a-z]?)",
    re.IGNORECASE,
)

# "Figure 328" / "Figures 312 and 313"
FIGURE_REF_RE = re.compile(
    r"\bfigures?\s+(\d+)",
    re.IGNORECASE,
)

# Words preceding a reference that upgrade it from "mention" to "strong".
# We check up to ~25 chars of context before each match.
STRONG_VERBS_RE = re.compile(
    r"(?:"
    r"see|"
    r"refer(?:s|red|ring)?\s+to|"
    r"referenced\s+(?:in|by)|"
    r"as\s+(?:defined|described|specified|shown|indicated|listed)\s+in|"
    r"according\s+to|"
    r"as\s+(?:per|in)|"
    r"defined\s+in|"
    r"described\s+in|"
    r"specified\s+in|"
    r"shown\s+in|"
    r"listed\s+in"
    r")\s*$",
    re.IGNORECASE,
)

STRONG_LOOKBACK = 30  # chars of preceding context to scan for a strong verb

# Max plausible NVMe figure number. The real max in the base spec is ~820, but
# a footnote superscript sometimes gets glued onto a figure number during PDF
# text extraction (e.g. "Figure 335¹)" → "Figure 3351"), producing 4+ digit
# false positives. Any target figure > this cap is dropped.
MAX_FIGURE_NUMBER = 999

# Max plausible nesting depth for a section number. NVMe sections rarely go
# past 5 components; anything deeper is almost certainly a footnote digit
# glued onto a real section number (e.g. "5.2.13.2.19" where the trailing 9
# is a footnote). Dropped as noise.
MAX_SECTION_DEPTH = 6


# ---------------------------------------------------------------------------
# Section lookup
#
# Builds a structure that answers: "what section contains printed_page P?"
# We pick the most-recently-started section (deepest nesting) whose
# target_page <= P. Ties broken by keeping section as listed in TOC order.


def build_section_lookup(toc: list[dict]) -> dict:
    """
    Return a lookup bundle usable by `section_for_page`.

    We keep TWO parallel arrays sorted by target_page:
      - pages[]    — the start page of each entry (ascending)
      - sections[] — the corresponding section dict

    Lookup is O(log n) via bisect.

    We also include a set of all known section numbers for quick validation
    of cross-ref targets.
    """
    # TOC is already in document order. Stable sort on target_page preserves
    # the tie-break behavior we want: entries with the same target_page stay
    # in their original (depth-first) order, so the DEEPEST section at that
    # page is the one we pick as containing.
    ordered = sorted(
        enumerate(toc),
        key=lambda iv: (iv[1]["target_page"], iv[0]),
    )
    sorted_entries = [e for _, e in ordered]
    pages = [e["target_page"] for e in sorted_entries]
    section_numbers = {e["section_number"] for e in toc}

    return {
        "pages": pages,
        "sorted_entries": sorted_entries,
        "section_numbers": section_numbers,
        "toc": toc,
    }


def section_for_page(lookup: dict, page: int | None) -> dict | None:
    """
    Return the TOC entry dict for the deepest section containing `page`, or
    None if the page is before the first section or unknown.
    """
    if page is None:
        return None
    pages = lookup["pages"]
    entries = lookup["sorted_entries"]
    # Find rightmost entry with target_page <= page.
    idx = bisect.bisect_right(pages, page) - 1
    if idx < 0:
        return None
    # Walk back to find the deepest-nested section at or before `page`.
    # bisect_right on ties lands past the group, so idx is already the last
    # entry with target_page <= page — which, given our sort, is the deepest.
    return entries[idx]


# ---------------------------------------------------------------------------
# Edge construction helpers


def _mk_edge(
    source: str,
    target: str,
    edge_type: str,
    evidence: str,
    confidence: str = "deterministic",
    strength: str | None = None,
) -> dict:
    edge = {
        "source": source,
        "target": target,
        "type": edge_type,
        "evidence": evidence,
        "confidence": confidence,
    }
    if strength is not None:
        edge["strength"] = strength
    return edge


# ---------------------------------------------------------------------------
# Cross-reference extraction from free text


def _classify_strength(text: str, match_start: int) -> str:
    """
    Given a text string and the start index of a match inside it, return
    "strong" if a strong-verb construction appears in the ~30 chars of
    context immediately preceding the match, else "mention".
    """
    lo = max(0, match_start - STRONG_LOOKBACK)
    context = text[lo:match_start]
    if STRONG_VERBS_RE.search(context):
        return "strong"
    return "mention"


def _snippet(text: str, start: int, end: int, pad: int = 40) -> str:
    """Short context snippet around [start:end] for the `evidence` field."""
    lo = max(0, start - pad)
    hi = min(len(text), end + pad)
    s = text[lo:hi].replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    if lo > 0:
        s = "…" + s
    if hi < len(text):
        s = s + "…"
    return s


def extract_cross_refs_from_text(
    text: str,
    source: str,
    known_sections: set[str] | None = None,
) -> list[dict]:
    """
    Scan `text` for Section X.Y and Figure N references and emit one edge
    per reference with source=`source`. Self-references (source == target)
    are dropped.

    If `known_sections` is provided, we emit a best-guess target even for
    section numbers we don't recognize (spec may reference e.g. a sub-clause
    we didn't extract). The `evidence` field preserves the raw reference.
    """
    edges: list[dict] = []
    if not text:
        return edges

    # Section references
    for m in SECTION_REF_RE.finditer(text):
        sec_num = m.group(1).rstrip(".")
        # Drop implausibly-deep section numbers (see MAX_SECTION_DEPTH).
        if sec_num.count(".") + 1 > MAX_SECTION_DEPTH:
            continue
        target = f"section:{sec_num}"
        if target == source:
            continue
        strength = _classify_strength(text, m.start())
        edges.append(
            _mk_edge(
                source=source,
                target=target,
                edge_type="cross_reference",
                evidence=_snippet(text, m.start(), m.end()),
                strength=strength,
            )
        )

    # Figure references
    for m in FIGURE_REF_RE.finditer(text):
        fig_num_str = m.group(1)
        # Drop implausibly-large figure numbers (see MAX_FIGURE_NUMBER).
        try:
            if int(fig_num_str) > MAX_FIGURE_NUMBER:
                continue
        except ValueError:
            continue
        fig_num = fig_num_str
        target = f"figure:{fig_num}"
        if target == source:
            continue
        strength = _classify_strength(text, m.start())
        edges.append(
            _mk_edge(
                source=source,
                target=target,
                edge_type="cross_reference",
                evidence=_snippet(text, m.start(), m.end()),
                strength=strength,
            )
        )

    return edges


# ---------------------------------------------------------------------------
# Table-side extraction


# NVMe description cells almost always begin with "Field Name (ABBR):" — we
# pull that out so downstream tools can anchor edges on field abbreviations.
FIELD_HEADER_RE = re.compile(
    r"^\s*([A-Z][A-Za-z0-9 /\-]{2,80}?)\s*\(([A-Z][A-Z0-9_]{1,15})\)\s*:",
)


def _iter_row_descriptions(rows: list[list[str]]) -> Iterable[tuple[int, str]]:
    """
    Yield (row_index, description_text) for each table data row, skipping
    band headers (single-cell section dividers). The description cell is
    always the last cell of a full-width data row.
    """
    for i, r in enumerate(rows):
        if not r:
            continue
        if len(r) == 1:
            # Band header like "Controller Capabilities and Features"
            continue
        yield i, r[-1]


def extract_from_table(
    table: dict,
    section_lookup: dict,
) -> list[dict]:
    """
    Emit edges for a single captioned table:

      1. figure → section containment (based on printed_page)
      2. cross-references found in table caption, raw_text, and each row's
         description cell
    """
    edges: list[dict] = []

    fig_num = table.get("figure_number")
    if fig_num is None:
        return edges  # orphan table, no stable id

    fig_id = f"figure:{fig_num}"
    known = section_lookup["section_numbers"]

    # 1. Containment edge: figure → section
    containing = section_for_page(section_lookup, table.get("printed_page"))
    if containing is not None:
        sec_id = f"section:{containing['section_number']}"
        edges.append(
            _mk_edge(
                source=fig_id,
                target=sec_id,
                edge_type="contained_in",
                evidence=(
                    f"printed_page {table.get('printed_page')} falls inside "
                    f"section {containing['section_number']} "
                    f"({containing['title']!r})"
                ),
            )
        )

    # 2. Cross-references from caption
    caption = table.get("caption") or ""
    if caption:
        edges.extend(
            extract_cross_refs_from_text(caption, source=fig_id, known_sections=known)
        )

    # 3. Cross-references from each row's description cell.
    # We keep the source as the figure id (not a synthetic row id) because
    # downstream graph queries care about figure-level links; row-level
    # granularity is captured implicitly by the row's position in tables.json.
    for _, desc in _iter_row_descriptions(table.get("rows", [])):
        if desc:
            edges.extend(
                extract_cross_refs_from_text(
                    desc, source=fig_id, known_sections=known
                )
            )

    # 4. Fallback: run raw_text through the regex too, in case the cell
    # splitter dropped a reference mid-row (multi-line descriptions can
    # occasionally land across two cells). Dedupe happens at the end.
    raw = table.get("raw_text") or ""
    if raw:
        edges.extend(
            extract_cross_refs_from_text(raw, source=fig_id, known_sections=known)
        )

    return edges


# ---------------------------------------------------------------------------
# Prose-side extraction (stub for when 1.3 lands)


def extract_from_prose(
    prose_sections: list[dict],
    section_lookup: dict,
) -> list[dict]:
    """
    Consume the 1.3 prose.json schema. Each entry looks like:

        {
            "section_number": "5.2.13",
            "title": "Identify command",
            "paragraphs": [{"text": "...", "pdf_page": 345}, ...],
            "normative": [{"strength": "shall", "text": "..."}, ...],
            ...
        }

    Emits cross-reference edges sourced from each section's prose text.
    Paragraphs and normative statements are scanned separately so we get
    coverage even for sections whose normative list is populated but
    paragraphs are empty (common for high-level section overviews).
    """
    edges: list[dict] = []
    known = section_lookup["section_numbers"]
    for s in prose_sections:
        sec_num = s.get("section_number")
        if not sec_num:
            continue
        sec_id = f"section:{sec_num}"

        for para in s.get("paragraphs", []) or []:
            text = (para or {}).get("text") or ""
            if text:
                edges.extend(
                    extract_cross_refs_from_text(
                        text, source=sec_id, known_sections=known
                    )
                )

        for norm in s.get("normative", []) or []:
            text = (norm or {}).get("text") or ""
            if text:
                edges.extend(
                    extract_cross_refs_from_text(
                        text, source=sec_id, known_sections=known
                    )
                )

    return edges


# ---------------------------------------------------------------------------
# Dedup + summary


def dedupe(edges: list[dict]) -> list[dict]:
    """
    Collapse duplicate edges. Two edges are "the same" if they share
    (source, target, type). Keeps the strongest strength and the first
    evidence string seen.
    """
    order = {"strong": 2, "mention": 1, None: 0}
    by_key: dict[tuple, dict] = {}
    for e in edges:
        k = (e["source"], e["target"], e["type"])
        if k not in by_key:
            by_key[k] = e
            continue
        existing = by_key[k]
        # Upgrade strength if this one is stronger.
        if order.get(e.get("strength")) > order.get(existing.get("strength")):
            existing["strength"] = e["strength"]
            existing["evidence"] = e["evidence"]
    return list(by_key.values())


def summarize(edges: list[dict]) -> dict:
    """Small stats bundle for CLI output / sanity checks."""
    by_type: dict[str, int] = {}
    by_strength: dict[str, int] = {}
    unknown_section_targets = 0
    unknown_figure_targets = 0
    for e in edges:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
        s = e.get("strength")
        if s:
            by_strength[s] = by_strength.get(s, 0) + 1
    return {
        "total": len(edges),
        "by_type": by_type,
        "by_strength": by_strength,
    }


# ---------------------------------------------------------------------------
# Public entry point


def extract_hierarchy(toc: list[dict]) -> list[dict]:
    """
    Emit parent_of / child_of edges for every section that has a parent
    in the TOC. E.g., section 3.1.3.3 -> child_of -> section 3.1.3.

    This makes the section tree fully navigable: from any section you can
    walk up to the parent or down to children.
    """
    known = {e["section_number"] for e in toc}
    edges: list[dict] = []
    for entry in toc:
        sec = entry["section_number"]
        parts = sec.split(".")
        if len(parts) < 2:
            continue
        parent = ".".join(parts[:-1])
        if parent not in known:
            continue
        edges.append(
            _mk_edge(
                source=f"section:{sec}",
                target=f"section:{parent}",
                edge_type="child_of",
                evidence=f"{sec} is a sub-section of {parent}",
            )
        )
    return edges


def build_relationships(
    toc: list[dict],
    tables: list[dict],
    prose_blocks: list[dict] | None = None,
) -> list[dict]:
    lookup = build_section_lookup(toc)
    edges: list[dict] = []

    for t in tables:
        edges.extend(extract_from_table(t, lookup))

    if prose_blocks:
        edges.extend(extract_from_prose(prose_blocks, lookup))

    edges.extend(extract_hierarchy(toc))

    return dedupe(edges)


# ---------------------------------------------------------------------------
# CLI

if __name__ == "__main__":
    from src import spec_env
    TOC_PATH       = spec_env.data_path("toc.json")
    TABLES_PATH    = spec_env.data_path("tables.json")
    PROSE_PATH     = spec_env.data_path("prose.json")          # optional, produced by 1.3
    OUTPUT_PATH    = spec_env.data_path("relationships.json")

    with open(TOC_PATH, encoding="utf-8") as f:
        toc = json.load(f)
    with open(TABLES_PATH, encoding="utf-8") as f:
        tables = json.load(f)

    prose = None
    if Path(PROSE_PATH).exists():
        with open(PROSE_PATH, encoding="utf-8") as f:
            prose = json.load(f)
        print(f"[info] including {len(prose)} prose blocks from {PROSE_PATH}")

    edges = build_relationships(toc, tables, prose_blocks=prose)

    stats = summarize(edges)
    print(f"extracted {stats['total']} relationships")
    print(f"  by type:     {stats['by_type']}")
    print(f"  by strength: {stats['by_strength']}")

    # Print a handful of examples of each type for eyeballing.
    seen_types: set[str] = set()
    print()
    print("sample edges:")
    for e in edges:
        t = e["type"]
        if t in seen_types:
            continue
        seen_types.add(t)
        print(f"  [{t}] {e['source']} -> {e['target']}")
        print(f"      evidence: {e['evidence'][:100]}")
        if len(seen_types) >= 5:
            break

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(edges, f, indent=2, ensure_ascii=False)
    print()
    print(f"wrote {OUTPUT_PATH}")
