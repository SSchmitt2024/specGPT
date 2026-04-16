"""
Rebuild toc.json from the PDF bookmark tree (doc.get_toc()).

The printed TOC only covers depth 1-3, and the previous parser.py + deep_sections.py
pipeline had systematic bugs: hierarchy drift in §3, duplicate 5.3.1, phantom annex
children, and 102 false-positive body headings. The PDF bookmarks are the authoritative
source — 959 entries, L1–L7, correct titles and hierarchy.

Usage:
    python src/toc_rebuild.py

Outputs:
    data/toc.json           — rebuilt TOC (overwrites)
    data/toc_old_backup.json — backup of the previous toc.json
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pymupdf


# ---------------------------------------------------------------------------- #
# Constants                                                                    #
# ---------------------------------------------------------------------------- #

PDF_PATH = "nvme_spec/NVMe_spec_full.pdf"
OUTPUT_PATH = "data/toc.json"
BACKUP_PATH = "data/toc_old_backup.json"

# PDF page index → printed page number.  Printed page 1 = PDF page 25 (0-indexed 24).
PAGE_OFFSET = 24

# Extract section number from bookmark title: "3.2.1.1 Namespace Overview" → ("3.2.1.1", "Namespace Overview")
# Also handles appendix sub-sections: "A.1 Overview", "B.5.1. Shadow Doorbell..."
# Pattern: optional letter, then digit(s), then optional .digit groups, optional trailing dot.
# Examples: "1", "3.2.1.1", "A.1", "B.5.1.", "C"
SECTION_NUM_RE = re.compile(r"^([A-Z](?:\.\d+)+|\d+(?:\.\d+)*)\.?\s+(.+)$")

# Annex line: "Annex A. Sanitize Operation Considerations (Informative)"
ANNEX_RE = re.compile(r"^Annex\s+([A-Z])\.?\s*(.*)$", re.IGNORECASE)


# ---------------------------------------------------------------------------- #
# Core                                                                         #
# ---------------------------------------------------------------------------- #

def rebuild_toc(pdf_path: str = PDF_PATH) -> list[dict]:
    """
    Read the PDF bookmark outline and convert to our toc.json schema.

    Each entry:
        section_number: str   — "3.2.1.1", "A.4", etc.
        title:          str   — clean title without the section number prefix
        level:          int   — 1-indexed depth from the bookmark tree
        target_page:    int   — printed page number (PDF page - PAGE_OFFSET)
    """
    doc = pymupdf.open(pdf_path)
    raw_toc = doc.get_toc()  # list of [level, title, page_number]
    doc.close()

    entries: list[dict] = []
    for level, raw_title, pdf_page in raw_toc:
        # Clean garbled characters from PDF extraction
        title = _fix_garbled(raw_title.strip())
        printed_page = pdf_page - PAGE_OFFSET

        # Parse section number from title
        annex_m = ANNEX_RE.match(title)
        sec_m = SECTION_NUM_RE.match(title)

        if annex_m:
            # "Annex A. Sanitize Operation Considerations (Informative)"
            section_number = annex_m.group(1)
            clean_title = annex_m.group(2).strip() or f"Annex {section_number}"
        elif sec_m:
            # "3.2.1.1 Namespace Overview"
            section_number = sec_m.group(1)
            clean_title = sec_m.group(2).strip()
        else:
            # Bare title with no section number — skip or flag
            print(f"[warn] no section number in bookmark: L{level} p.{pdf_page} '{title[:60]}'")
            continue

        entries.append({
            "section_number": section_number,
            "title": clean_title,
            "level": level,
            "target_page": printed_page,
        })

    return entries


def _fix_garbled(text: str) -> str:
    """Normalize common PDF extraction artifacts."""
    text = text.replace("\ufffd", "")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


# ---------------------------------------------------------------------------- #
# Validation                                                                   #
# ---------------------------------------------------------------------------- #

def validate(entries: list[dict]) -> dict:
    """Quick sanity checks on the rebuilt TOC."""
    from collections import Counter

    stats: dict = {
        "total_entries": len(entries),
    }

    # Level distribution
    levels = Counter(e["level"] for e in entries)
    stats["level_distribution"] = dict(sorted(levels.items()))

    # Duplicate section numbers
    seen: dict[str, int] = {}
    dups = []
    for e in entries:
        sn = e["section_number"]
        seen[sn] = seen.get(sn, 0) + 1
    for sn, count in seen.items():
        if count > 1:
            dups.append(f"{sn} (x{count})")
    stats["duplicate_section_numbers"] = dups or "none"

    # Check section_number depth matches level
    mismatches = []
    for e in entries:
        expected_depth = e["section_number"].count(".") + 1
        # Annex letters count as depth 1
        if e["section_number"][0].isalpha() and not e["section_number"][0].isdigit():
            expected_depth = e["section_number"].count(".") + 1
        if expected_depth != e["level"]:
            mismatches.append(f"{e['section_number']} (expected L{expected_depth}, got L{e['level']})")
    stats["level_mismatches"] = mismatches[:10] if mismatches else "none"
    if len(mismatches) > 10:
        stats["level_mismatches_total"] = len(mismatches)

    # Negative pages
    neg = [e for e in entries if e["target_page"] < 0]
    stats["negative_pages"] = len(neg)

    return stats


# ---------------------------------------------------------------------------- #
# CLI                                                                          #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    print(f"reading bookmarks from {PDF_PATH}...")
    entries = rebuild_toc(PDF_PATH)
    print(f"  extracted {len(entries)} entries")

    # Validate
    print("\nvalidation:")
    stats = validate(entries)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Backup old toc.json if it exists
    old_path = Path(OUTPUT_PATH)
    if old_path.exists():
        shutil.copy2(old_path, BACKUP_PATH)
        print(f"\nbacked up old toc.json to {BACKUP_PATH}")

    # Write
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"wrote {OUTPUT_PATH}")

    # Show samples
    print("\nfirst 15 entries:")
    for e in entries[:15]:
        print(f"  {e['section_number']:<15} L{e['level']}  {e['title'][:55]:<55}  p.{e['target_page']}")

    print("\nsample deep entries (L4+):")
    shown = 0
    for e in entries:
        if e["level"] >= 4:
            print(f"  {e['section_number']:<20} L{e['level']}  {e['title'][:50]:<50}  p.{e['target_page']}")
            shown += 1
            if shown >= 15:
                break

    print("\nsection 5.3 area:")
    for e in entries:
        if e["section_number"].startswith("5.3"):
            print(f"  {e['section_number']:<15} L{e['level']}  {e['title'][:55]:<55}  p.{e['target_page']}")
