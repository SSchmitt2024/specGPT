"""
Attach `parent_section` to every entry in `tables.json` by joining on
`printed_page` against the rebuilt `toc.json`.

Closes the §1.4 containment gap from `00_data_effectiveness_report.md` —
plan required every table to carry its parent section ID, but the original
parser emitted 0/717.

Logic source: re-uses `build_section_lookup` + `section_for_page` from
`src/relationships.py` so the page-range join exactly matches the
`contained_in` edges in `relationships.json` (single source of truth).

Usage:
    python scripts/scrip_helpers/add_parent_section_to_tables.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.relationships import build_section_lookup, section_for_page

TOC_PATH = "data/toc.json"
TABLES_PATH = "data/tables.json"
BACKUP_PATH = "data/tables_pre_parent_section_backup.json"


def main() -> None:
    with open(TOC_PATH, encoding="utf-8") as f:
        toc = json.load(f)
    with open(TABLES_PATH, encoding="utf-8") as f:
        tables = json.load(f)

    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(TABLES_PATH, BACKUP_PATH)
        print(f"backed up {TABLES_PATH} -> {BACKUP_PATH}")

    lookup = build_section_lookup(toc)

    attached = 0
    missed = 0
    for t in tables:
        page = t.get("printed_page")
        sec = section_for_page(lookup, page)
        if sec is None:
            t["parent_section"] = None
            missed += 1
        else:
            t["parent_section"] = sec["section_number"]
            attached += 1

    with open(TABLES_PATH, "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)

    print(f"tables total:       {len(tables)}")
    print(f"  parent attached:  {attached}")
    print(f"  parent missing:   {missed}")
    print(f"wrote {TABLES_PATH}")


if __name__ == "__main__":
    main()
