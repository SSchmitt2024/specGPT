"""
Rebuild data/definitions.json from the existing data/prose.json without
re-scanning the PDF.

Why: the original extractor only walked §1.5.x. The spec actually has three
definition sections — §1.5, §1.6 (I/O Command Set specific), and §1.7
(NVM Command Set specific) — so terms like `LBA` (§1.7.2) and `User Data
Format` (§1.6.5) were silently missing. The fix lives in
``src/prose._build_definitions`` (broadened prefix check). This script just
re-applies that builder against the prose JSON we already have on disk.

Usage:
    python scripts/scrip_helpers/rebuild_definitions.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prose import _build_definitions

PROSE_PATH = "data/prose.json"
DEFS_PATH = "data/definitions.json"
BACKUP_PATH = "data/definitions_pre_1_6_1_7_backup.json"


def main() -> None:
    with open(PROSE_PATH, encoding="utf-8") as f:
        sections = json.load(f)

    if os.path.exists(DEFS_PATH) and not os.path.exists(BACKUP_PATH):
        shutil.copy2(DEFS_PATH, BACKUP_PATH)
        print(f"backed up {DEFS_PATH} -> {BACKUP_PATH}")

    before = {}
    if os.path.exists(DEFS_PATH):
        with open(DEFS_PATH, encoding="utf-8") as f:
            before = json.load(f)

    defs = _build_definitions(sections)

    with open(DEFS_PATH, "w", encoding="utf-8") as f:
        json.dump(defs, f, indent=2, ensure_ascii=False)

    added = sorted(set(defs) - set(before))
    removed = sorted(set(before) - set(defs))

    print(f"definitions before: {len(before)}")
    print(f"definitions after:  {len(defs)}")
    print(f"  added:   {len(added)}")
    print(f"  removed: {len(removed)}")
    if added:
        print("\nfirst 25 added terms:")
        for t in added[:25]:
            print(f"  + {t}")
    if removed:
        print("\nfirst 10 removed terms:")
        for t in removed[:10]:
            print(f"  - {t}")

    print("\nspot check:")
    for k in (
        "logical block address (LBA)",
        "logical block",
        "User Data Format",
        "Endurance Group Host Read Command",
    ):
        v = defs.get(k)
        status = "OK" if v else "MISS"
        snippet = (v or "")[:80].replace("\n", " ")
        print(f"  [{status}] {k!r}: {snippet}")

    print(f"\nwrote {DEFS_PATH}")


if __name__ == "__main__":
    main()
