"""
Refresh structural fields on cards.json after a TOC rebuild — without re-running
the LLM summary step.

Preserves: summary, keywords (LLM-derived).
Refreshes: section_id, title, parent_section, child_sections, tables,
           prose_blocks, relationships, normative_count, level.

Cards whose section_id no longer appears in the new TOC are dropped.
New TOC sections are added as stubs (empty summary/keywords) so a later
LLM run can fill them.

Usage:
    python scripts/helpers/refresh_cards_structural.py
"""

from __future__ import annotations

import json
import os
import sys

# Re-use helpers from the canonical generator so logic stays in one place.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm.generate_cards import (
    TOC_PATH, PROSE_PATH, TABLES_PATH, RELATIONSHIPS_PATH,
    OUTPUT_PATH, STATE_PATH,
    DEFAULT_SPEC_DOCUMENT, DEFAULT_SPEC_VERSION,
    _load_json, _save_json,
    _flatten_prose, _parent_section,
    _build_child_map, _tables_by_section, _relationships_by_section,
)


def main() -> None:
    toc = _load_json(TOC_PATH, [])
    prose = _load_json(PROSE_PATH, [])
    tables = _load_json(TABLES_PATH, [])
    relationships = _load_json(RELATIONSHIPS_PATH, [])
    existing = _load_json(OUTPUT_PATH, [])

    if not toc or not prose:
        print("ERROR: toc.json or prose.json missing.", file=sys.stderr)
        sys.exit(1)

    prose_by_section = {p["section_number"]: p for p in prose}
    children_map = _build_child_map(toc)
    tables_map = _tables_by_section(tables, prose_by_section, relationships)
    rels_map = _relationships_by_section(relationships)

    existing_by_id = {c.get("section_id"): c for c in existing if c.get("section_id")}

    new_cards: list[dict] = []
    preserved = 0
    new_sections = 0

    for entry in toc:
        sec_num = entry["section_number"]
        title = entry["title"]
        section_prose = prose_by_section.get(sec_num, {})
        prose_text, prose_idxs = _flatten_prose(section_prose)
        table_figs = sorted(tables_map.get(sec_num, []))
        section_rels = rels_map.get(sec_num, [])
        normative = section_prose.get("normative") or []

        prior = existing_by_id.get(sec_num)
        if prior:
            summary = prior.get("summary", "")
            keywords = prior.get("keywords", [])
            preserved += 1
        else:
            summary = ""
            keywords = []
            new_sections += 1

        new_cards.append({
            "section_id": sec_num,
            "title": title,
            "spec_document": prior.get("spec_document", DEFAULT_SPEC_DOCUMENT) if prior else DEFAULT_SPEC_DOCUMENT,
            "spec_version": prior.get("spec_version", DEFAULT_SPEC_VERSION) if prior else DEFAULT_SPEC_VERSION,
            "summary": summary,
            "keywords": keywords,
            "parent_section": _parent_section(sec_num),
            "child_sections": children_map.get(sec_num, []),
            "tables": table_figs,
            "prose_blocks": prose_idxs,
            "relationships": section_rels,
            "normative_count": len(normative),
            "level": entry.get("level"),
        })

    dropped = len(existing_by_id) - preserved

    # Rewrite state file to match the new card population — sections with
    # non-empty summaries are "processed", everything else needs LLM.
    processed = sorted(c["section_id"] for c in new_cards if c.get("summary"))
    state = {"processed": processed}

    _save_json(OUTPUT_PATH, new_cards)
    _save_json(STATE_PATH, state)

    # Reporting
    empty_summary = sum(1 for c in new_cards if not c["summary"])
    fig_links = sum(len(c["tables"]) for c in new_cards)
    duplicates = len(new_cards) - len({c["section_id"] for c in new_cards})

    print(f"cards written:        {len(new_cards)}")
    print(f"  duplicate ids:      {duplicates}")
    print(f"  preserved summary:  {preserved}")
    print(f"  new (stub) cards:   {new_sections}")
    print(f"  dropped cards:      {dropped}  (section_ids no longer in toc)")
    print(f"  empty summary:      {empty_summary}")
    print(f"  total figure links: {fig_links}  (target: 717)")
    print(f"\nwrote {OUTPUT_PATH}")
    print(f"wrote {STATE_PATH}  (processed={len(processed)})")


if __name__ == "__main__":
    main()
