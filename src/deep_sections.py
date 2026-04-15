"""
Deep section discovery for the NVMe spec (Phase 1.1b).

The printed TOC only contains depth 1-3 entries. The spec body contains
deeper sub-sections using two heading formats:

  Depth 4: Bold title only, NO section number in the text.
           Example: "Discovery Controller" on its own bold line = section 3.1.3.3

  Depth 5+: Bold section number on one line, bold title on the next line.
            Example: "3.1.3.3.1" (bold) then "Discovery Controller Async Event Config" (bold)

This script:
  1. Scans the full PDF body for bold heading-like lines.
  2. Extracts depth-5+ sections from explicit section-number-only lines.
  3. Detects depth-4 sections as bold title-only headings not in the current TOC.
  4. Assigns sequential section numbers to depth-4 headings within their parent.
  5. Outputs enriched data/toc.json with all section levels merged in.

After running this, re-run prose.py and relationships.py to propagate the
enriched TOC through the pipeline.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pymupdf


# ---------------------------------------------------------------------------
# Constants

BOLD_FLAG = 16
MIN_HEADING_SIZE = 9.5

# Running header: "NVM Express ... Revision 2.3"
RUNNING_HEADER_RE = re.compile(r"NVM Express.*?Revision\s+[\d.]+", re.IGNORECASE)

# Figure/Table captions
CAPTION_RE = re.compile(r"^(Figure|Table)\s+\d+\s*:", re.IGNORECASE)

# Matches a standalone section number (depth 4+): "3.1.3.3", "5.2.12.1.14", "A.1.2.3"
# Must have at least 3 dots (depth >= 4).
DEEP_SECTION_NUM_RE = re.compile(r"^([A-Z]?\d+(?:\.\d+){3,})\.?$")

# Matches ANY section number (depth 2+) at line start for depth-4 title-combined lines
# (rare but possible)
ANY_SECTION_NUM_RE = re.compile(r"^([A-Z]?\d+(?:\.\d+){2,})\.?\s+(.+)")

# Page offset: pdf_page_idx - printed_page_number
PAGE_OFFSET = 23


# ---------------------------------------------------------------------------
# Text normalization (mirrors prose.py)

def _fix_garbled(text: str) -> str:
    text = text.replace("\ufffd", "")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text


def _norm_title(s: str) -> str:
    s = _fix_garbled(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# PDF scanning

def _collect_bold_headings(page: pymupdf.Page) -> list[dict]:
    """
    Extract all bold heading-like lines from a page.

    Returns list of dicts with: text, bbox, size, bold, pdf_page.
    Filters out running headers, captions, pure page numbers, and
    non-bold/small lines.
    """
    d = page.get_text("dict")
    out = []
    for block in d.get("blocks", []):
        if "lines" not in block:
            continue
        for line in block["lines"]:
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            bold = all(bool(s.get("flags", 0) & BOLD_FLAG) for s in spans)
            if not bold:
                continue
            size = max(s.get("size", 0) for s in spans)
            if size < MIN_HEADING_SIZE:
                continue
            text = "".join(s["text"] for s in spans)
            text = _fix_garbled(text)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            bbox = line.get("bbox") or (
                min(s["bbox"][0] for s in spans),
                min(s["bbox"][1] for s in spans),
                max(s["bbox"][2] for s in spans),
                max(s["bbox"][3] for s in spans),
            )
            # Filter noise
            if RUNNING_HEADER_RE.search(text):
                continue
            if CAPTION_RE.match(text):
                continue
            if re.fullmatch(r"\d{1,4}", text):
                continue
            out.append({
                "text": text,
                "bbox": tuple(bbox),
                "size": size,
            })
    return out


def scan_body_headings(
    pdf_path: str,
    first_page: int = 24,
    last_page: int | None = None,
) -> list[dict]:
    """
    Scan all body pages and return a flat list of bold heading lines
    in document order, each tagged with pdf_page.
    """
    doc = pymupdf.open(pdf_path)
    if last_page is None:
        last_page = doc.page_count - 1

    all_headings = []
    for pi in range(first_page, last_page + 1):
        page = doc[pi]
        headings = _collect_bold_headings(page)
        for h in headings:
            h["pdf_page"] = pi
            h["printed_page"] = pi - PAGE_OFFSET
        all_headings.extend(headings)

    doc.close()
    return all_headings


# ---------------------------------------------------------------------------
# Phase 1: Extract depth-5+ sections (explicit section numbers)

def extract_numbered_sections(headings: list[dict]) -> list[dict]:
    """
    Find bold lines that are standalone section numbers (depth 4+).
    Pair each with the next bold line on the same page as the title.

    Returns list of {section_number, title, level, pdf_page, printed_page}.
    """
    sections = []
    i = 0
    while i < len(headings):
        h = headings[i]
        m = DEEP_SECTION_NUM_RE.match(h["text"])
        if m:
            sec_num = m.group(1)
            level = sec_num.count(".") + 1
            # Next bold line should be the title
            title = ""
            if i + 1 < len(headings):
                nxt = headings[i + 1]
                # Title should be on same page or very close
                if nxt["pdf_page"] == h["pdf_page"]:
                    # Make sure it's not ANOTHER section number
                    if not DEEP_SECTION_NUM_RE.match(nxt["text"]):
                        title = nxt["text"]
                        i += 1  # skip the title line

            sections.append({
                "section_number": sec_num,
                "title": title or f"(untitled sub-section {sec_num})",
                "level": level,
                "target_page": h["printed_page"],
                "pdf_page": h["pdf_page"],
                "source": "body_numbered",
            })
        else:
            # Check for combined format: "8.1.18.6.1 Establishing or Reducing..."
            m2 = ANY_SECTION_NUM_RE.match(h["text"])
            if m2:
                sec_num = m2.group(1)
                level = sec_num.count(".") + 1
                if level >= 4:
                    title = m2.group(2).strip()
                    sections.append({
                        "section_number": sec_num,
                        "title": title,
                        "level": level,
                        "target_page": h["printed_page"],
                        "pdf_page": h["pdf_page"],
                        "source": "body_combined",
                    })
        i += 1

    return sections


# ---------------------------------------------------------------------------
# Phase 2: Detect depth-4 title-only headings

def detect_depth4_sections(
    headings: list[dict],
    existing_toc: list[dict],
    numbered_sections: list[dict],
) -> list[dict]:
    """
    Find bold title-only headings that are NOT in the current TOC and NOT
    section-number lines. These are depth-4 sub-section headings.

    Groups them by parent section and assigns sequential numbers.
    """
    # Build lookup of known titles WITH their page locations.
    # A bold title in the body only matches a TOC entry if it's within
    # a few pages of that entry's target_page. This prevents false
    # positives from the definitions section (1.5.x) whose titles
    # happen to match real sub-section headings elsewhere in the spec.
    known_title_pages: list[tuple[str, int]] = []
    for entry in existing_toc:
        known_title_pages.append((_norm_title(entry["title"]), entry["target_page"]))

    # Also exclude titles from numbered sections we already found
    for s in numbered_sections:
        known_title_pages.append((_norm_title(s["title"]), s["target_page"]))

    def _is_known_title(text: str, printed_page: int, tolerance: int = 3) -> bool:
        """Check if a title matches a known TOC entry near the same page."""
        norm = _norm_title(text)
        for known_norm, known_page in known_title_pages:
            if norm == known_norm and abs(printed_page - known_page) <= tolerance:
                return True
        return False

    # Build parent section lookup: for each depth-3 section, what's its page range?
    depth3_sections = []
    for i, entry in enumerate(existing_toc):
        if entry["level"] == 3:
            # Find end page: next section's target_page, or end of doc
            end_page = 9999
            for j in range(i + 1, len(existing_toc)):
                if existing_toc[j]["level"] <= 3:
                    end_page = existing_toc[j]["target_page"]
                    break
            depth3_sections.append({
                "section_number": entry["section_number"],
                "title": entry["title"],
                "start_page": entry["target_page"],
                "end_page": end_page,
                "start_pdf": entry["target_page"] + PAGE_OFFSET,
                "end_pdf": end_page + PAGE_OFFSET,
            })

    # Also need depth-2 sections that might have depth-4 children
    # (some depth-3 sections in the TOC are actually depth-2 with depth-4 children)

    # Collect candidate headings: bold titles not matching known entries
    # and not matching section number patterns
    candidates = []
    for h in headings:
        text = h["text"]
        # Skip section-number lines
        if DEEP_SECTION_NUM_RE.match(text):
            continue
        if ANY_SECTION_NUM_RE.match(text):
            sec_m = ANY_SECTION_NUM_RE.match(text)
            if sec_m and sec_m.group(1).count(".") >= 2:
                continue

        # Skip if matches a known TOC title near the same page
        if _is_known_title(text, h["printed_page"]):
            continue

        # Skip very short "titles" (likely bold emphasis in prose)
        if len(text) < 5:
            continue

        # Skip lines that look like table content (single words like "Value:")
        if text.endswith(":") and " " not in text.strip(":"):
            continue

        # Skip lines that start with a section number followed by title text —
        # these are depth 1-3 headings already captured in the TOC.
        # Pattern: "1 INTRODUCTION", "3.2 NVM Subsystem Entities", "B.5.1 Shadow"
        if re.match(r"^[A-Z]?\d+(?:\.\d+)*\.?\s+[A-Z]", text):
            continue

        # Skip "Annex X. Title" lines (depth-1 headings)
        if re.match(r"^Annex\s+[A-Z]", text, re.IGNORECASE):
            continue

        # Skip lines that are just "Case N:" or similar prose emphasis
        if re.match(r"^Case\s+\d+", text):
            continue

        candidates.append(h)

    # Group candidates by parent section
    # For each candidate, find the depth-3 (or depth-2) section it falls within
    parent_children: dict[str, list[dict]] = {}

    for cand in candidates:
        pp = cand["printed_page"]
        parent = None
        # Find the deepest section whose range contains this page
        for d3 in depth3_sections:
            if d3["start_page"] <= pp < d3["end_page"]:
                parent = d3
        if parent is None:
            # Try depth-2 sections
            for entry in existing_toc:
                if entry["level"] == 2 and entry["target_page"] <= pp:
                    parent = {
                        "section_number": entry["section_number"],
                        "start_page": entry["target_page"],
                    }
            if parent is None:
                continue

        parent_num = parent["section_number"]
        if parent_num not in parent_children:
            parent_children[parent_num] = []
        parent_children[parent_num].append(cand)

    # For parents that already have depth-5+ children, we know their depth-4
    # numbering. Use depth-5 children to validate depth-4 ordering.
    numbered_by_parent: dict[str, list[str]] = {}
    for s in numbered_sections:
        parts = s["section_number"].split(".")
        if len(parts) >= 5:
            d4_parent = ".".join(parts[:4])
            d3_parent = ".".join(parts[:3])
            if d3_parent not in numbered_by_parent:
                numbered_by_parent[d3_parent] = set()
            numbered_by_parent[d3_parent].add(d4_parent)

    # Assign sequential numbers
    depth4_sections = []
    for parent_num, children in parent_children.items():
        # Sort by document position
        children.sort(key=lambda c: (c["pdf_page"], c["bbox"][1]))

        for idx, child in enumerate(children, start=1):
            sec_num = f"{parent_num}.{idx}"
            depth4_sections.append({
                "section_number": sec_num,
                "title": child["text"],
                "level": parent_num.count(".") + 2,
                "target_page": child["printed_page"],
                "pdf_page": child["pdf_page"],
                "source": "body_title_only",
            })

    return depth4_sections


# ---------------------------------------------------------------------------
# Phase 3: Infer missing parents from children

def infer_missing_parents(
    numbered_sections: list[dict],
    depth4_sections: list[dict],
    existing_toc: list[dict],
) -> list[dict]:
    """
    For depth-5+ sections whose depth-4 parent wasn't detected, create
    a parent entry. Similarly for depth-7 sections missing depth-6 parents.
    """
    all_known = set()
    for e in existing_toc:
        all_known.add(e["section_number"])
    for s in numbered_sections:
        all_known.add(s["section_number"])
    for s in depth4_sections:
        all_known.add(s["section_number"])

    inferred = []
    seen = set()

    for s in numbered_sections:
        parts = s["section_number"].split(".")
        # Check all ancestor levels above depth-3
        for depth in range(4, len(parts)):
            ancestor = ".".join(parts[:depth])
            if ancestor not in all_known and ancestor not in seen:
                seen.add(ancestor)
                inferred.append({
                    "section_number": ancestor,
                    "title": f"(sub-section {ancestor})",
                    "level": depth,
                    "target_page": s["target_page"],
                    "pdf_page": s["pdf_page"],
                    "source": "inferred_parent",
                })
                all_known.add(ancestor)

    return inferred


# ---------------------------------------------------------------------------
# Merge and output

def merge_toc(
    existing_toc: list[dict],
    new_sections: list[dict],
) -> list[dict]:
    """
    Merge new sections into the existing TOC, maintaining document order.
    New sections are interleaved based on target_page and level.
    """
    # Clean new sections to match TOC schema
    clean_new = []
    for s in new_sections:
        clean_new.append({
            "section_number": s["section_number"],
            "title": s["title"],
            "level": s["level"],
            "target_page": s["target_page"],
        })

    # Combine
    combined = list(existing_toc) + clean_new

    # Sort by target_page, then by section_number for stable ordering
    def sort_key(entry):
        # Parse section number into numeric tuple for proper ordering
        parts = []
        for p in entry["section_number"].split("."):
            try:
                parts.append((0, int(p)))
            except ValueError:
                parts.append((1, ord(p[0]) if p else 0))
        return (entry["target_page"], parts)

    combined.sort(key=sort_key)

    return combined


# ---------------------------------------------------------------------------
# Validation

def validate(
    enriched_toc: list[dict],
    relationships_path: str | None = None,
    fields_path: str | None = None,
) -> dict:
    """
    Check how many previously-broken references are now resolved.
    """
    known = {e["section_number"] for e in enriched_toc}

    stats = {
        "total_sections": len(enriched_toc),
        "new_sections": len([e for e in enriched_toc if e.get("level", 0) >= 4]),
    }

    if relationships_path and os.path.exists(relationships_path):
        with open(relationships_path, encoding="utf-8") as f:
            rels = json.load(f)
        missing_before = set()
        for e in rels:
            if e["target"].startswith("section:"):
                sec = e["target"].split(":", 1)[1]
                if sec not in known:
                    missing_before.add(sec)
        stats["rels_still_missing"] = len(missing_before)
        stats["rels_still_missing_examples"] = sorted(missing_before)[:10]

    if fields_path and os.path.exists(fields_path):
        with open(fields_path, encoding="utf-8") as f:
            fields = json.load(f)
        missing_field_refs = set()
        for field in fields:
            if field.get("cross_refs"):
                for cr in field["cross_refs"]:
                    if cr["type"] == "section" and cr["id"] not in known:
                        missing_field_refs.add(cr["id"])
        stats["field_refs_still_missing"] = len(missing_field_refs)
        stats["field_refs_still_missing_examples"] = sorted(missing_field_refs)[:10]

    return stats


# ---------------------------------------------------------------------------
# CLI

if __name__ == "__main__":
    PDF_PATH = "nvme_spec/NVMe_spec_full.pdf"
    TOC_PATH = "data/toc.json"
    OUTPUT_PATH = "data/toc.json"  # overwrite in place
    BACKUP_PATH = "data/toc_depth3_backup.json"

    with open(TOC_PATH, encoding="utf-8") as f:
        existing_toc = json.load(f)

    print(f"existing TOC: {len(existing_toc)} entries (max depth 3)")

    # --- Phase 1: Scan body ---
    print("\nscanning PDF body for bold headings...")
    headings = scan_body_headings(PDF_PATH)
    print(f"  found {len(headings)} bold heading-like lines")

    # --- Phase 2: Extract depth-5+ ---
    print("\nextracting depth-5+ sections (explicit numbers)...")
    numbered = extract_numbered_sections(headings)
    print(f"  found {len(numbered)} numbered sections")
    by_level = {}
    for s in numbered:
        by_level.setdefault(s["level"], 0)
        by_level[s["level"]] += 1
    for lvl in sorted(by_level):
        print(f"    depth {lvl}: {by_level[lvl]}")

    # --- Phase 3: Detect depth-4 ---
    print("\ndetecting depth-4 title-only headings...")
    depth4 = detect_depth4_sections(headings, existing_toc, numbered)
    print(f"  found {len(depth4)} depth-4 sections")

    # --- Phase 4: Infer missing parents ---
    print("\ninferring missing parent sections...")
    inferred = infer_missing_parents(numbered, depth4, existing_toc)
    print(f"  inferred {len(inferred)} parent sections")

    # --- Merge ---
    all_new = numbered + depth4 + inferred
    print(f"\ntotal new sections: {len(all_new)}")

    # Backup original TOC
    os.makedirs(os.path.dirname(BACKUP_PATH), exist_ok=True)
    with open(BACKUP_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_toc, f, indent=2, ensure_ascii=False)
    print(f"backed up original TOC to {BACKUP_PATH}")

    enriched = merge_toc(existing_toc, all_new)
    print(f"enriched TOC: {len(enriched)} entries")

    # --- Validate ---
    print("\nvalidation:")
    stats = validate(
        enriched,
        relationships_path="data/relationships.json",
        fields_path="data/fields.json",
    )
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # --- Write ---
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {OUTPUT_PATH}")

    # Show some examples
    print("\nsample new entries:")
    shown = 0
    for e in enriched:
        if e.get("level", 0) >= 4:
            print(f"  {e['section_number']:<20} L{e['level']}  {e['title'][:55]:<55}  p.{e['target_page']}")
            shown += 1
            if shown >= 20:
                break
