"""
NVMe spec prose extractor (Phase 1.3).

Reads the NVMe base spec PDF and emits per-section prose text:

    {
        "section_number": "1.5.4",
        "title": "Admin Queue",
        "level": 3,
        "start_pdf_page": 28,
        "end_pdf_page": 29,
        "paragraphs": [
            {"text": "An Admin Queue ...", "pdf_page": 28},
            ...
        ],
        "normative": [
            {"strength": "shall", "text": "An Admin Queue shall ...", "pdf_page": 28},
            ...
        ]
    }

Also emits `definitions.json` — a `{term: definition}` lookup built from
section 1.5 (the spec's Definitions section, where each subsection is one
defined term).

Strategy:
  1. Load toc.json (from parser.py).
  2. Walk every PDF content page, collect:
       - all visual text lines with their bbox, font size, bold flag
       - all table bboxes (from pymupdf find_tables)
  3. Find each TOC entry's heading position in the body by matching the
     title against bold lines in order. This gives us [start, end) page/y
     boundaries per section.
  4. Extract prose for each section: keep only lines that fall within
     [start, end), are NOT inside a table bbox, are NOT the heading line
     itself, and are NOT running headers.
  5. Group lines into paragraphs using the block index from get_text("dict")
     and y-gap heuristics.
  6. Tag shall/should/may sentences as normative requirements.
  7. Parse section 1.5.x as the definitions table.

This file does NOT depend on tables.py — we re-run find_tables() locally
to get bbox exclusions. That way 1.3 can be developed in parallel with 1.2.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pymupdf


# ---------------------------------------------------------------------------
# Constants

# Running header on every spec page: "NVM Express ... Revision 2.3"
RUNNING_HEADER_RE = re.compile(
    r"NVM Express.*?Revision\s+[\d.]+",
    re.IGNORECASE,
)

# Figure/Table captions are bold and look like headings — filter them out.
CAPTION_RE = re.compile(r"^(Figure|Table)\s+\d+\s*:", re.IGNORECASE)

# ANNEX / Annex prefix on body headings: "Annex B. Host Considerations"
ANNEX_BODY_RE = re.compile(r"^Annex\s+([A-Z])\.\s*(.+)$", re.IGNORECASE)

# Section-number prefix on a body heading line: "1.5", "B.5.1", "8.2.3.1"
SECTION_NUM_RE = re.compile(r"^([A-Z]?\d+(?:\.\d+)*)\.?$")

# Normative keywords. Tag whole sentences that contain one of these.
NORMATIVE_WORDS = ("shall", "should", "may")

# Bold flag bit in PyMuPDF span flags.
BOLD_FLAG = 16

# Body text font size (pt). Headings are usually >= 10 and bold; body is 9-10.
# Used only as a sanity floor — real heading classification is by bold flag +
# TOC title match.
MIN_HEADING_SIZE = 9.5

# y-tolerance for merging spans that share a visual line.
LINE_Y_TOL = 2.5

# When matching a TOC entry to a body heading, we search forward from the
# current walk position up to this many pages ahead. The TOC `target_page`
# is usually exact, but we've seen ±1 page drift in some spec renderings.
HEADING_SEARCH_WINDOW_PAGES = 4

# Page offset: PDF page index - printed page number. printed p.1 is pdf idx 24.
PAGE_OFFSET = 23

# Inclusive bounds on the PDF page index where real content lives.
# The full spec is 784 pages; content starts at pdf idx 24 (printed p.1).
# We'll auto-detect the end in __main__ but a sane floor is here.
DEFAULT_FIRST_CONTENT_PAGE = 24


# ---------------------------------------------------------------------------
# Text normalization


def _fix_garbled(text: str) -> str:
    """
    Replace the black-diamond replacement char used by pymupdf for chars it
    can't map (typically ® and similar symbols in this PDF).

    We also normalize common unicode artifacts so downstream search works.
    """
    text = text.replace("\ufffd", "")
    # Non-breaking space -> regular space
    text = text.replace("\u00a0", " ")
    # Curly quotes -> straight, for easier matching (but keep em-dash as is)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text


def _norm_title(s: str) -> str:
    """
    Normalize a title for fuzzy comparison between TOC and body heading.

    Lowercase, strip punctuation except letters/digits/space, squeeze
    whitespace. This is tolerant enough for the small variations we see
    (e.g., smart quotes vs apostrophes, trailing whitespace, trademark sign).
    """
    s = _fix_garbled(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Per-page extraction


def _collect_page_lines(
    page: pymupdf.Page,
) -> list[dict]:
    """
    Walk the dict structure of a page and return a list of "visual lines".

    Each line dict has:
        bbox:  (x0, y0, x1, y1)
        text:  concatenated span text, whitespace-tidied
        bold:  True if ALL spans are bold
        size:  max span size
        block_no: source block index (for paragraph grouping)

    Spans that share a bbox y-center are merged into one visual line. This
    lets us treat "1.5" and "Definitions" on the same y as one heading line.
    """
    d = page.get_text("dict")
    out: list[dict] = []
    for bi, block in enumerate(d.get("blocks", [])):
        if "lines" not in block:
            continue
        for line in block["lines"]:
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans)
            text = _fix_garbled(text)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            bold = all(bool(s.get("flags", 0) & BOLD_FLAG) for s in spans)
            size = max(s.get("size", 0) for s in spans)
            bbox = line.get("bbox") or (
                min(s["bbox"][0] for s in spans),
                min(s["bbox"][1] for s in spans),
                max(s["bbox"][2] for s in spans),
                max(s["bbox"][3] for s in spans),
            )
            out.append(
                {
                    "bbox": tuple(bbox),
                    "text": text,
                    "bold": bold,
                    "size": size,
                    "block_no": bi,
                }
            )
    return out


def _collect_table_bboxes(page: pymupdf.Page) -> list[tuple]:
    """
    Return bboxes of every table find_tables() detects on the page. Used to
    exclude lines that live inside a table region from the prose output.

    We don't care about nesting here — any line inside any table bbox is
    excluded. The top-level tables.py pipeline owns structured table output.
    """
    try:
        tabs = page.find_tables().tables
    except Exception:
        return []
    return [tuple(t.bbox) for t in tabs]


def _in_any_bbox(line_bbox, bboxes: list[tuple], slack: float = 1.0) -> bool:
    lx0, ly0, lx1, ly1 = line_bbox
    lcy = (ly0 + ly1) / 2
    lcx = (lx0 + lx1) / 2
    for bx0, by0, bx1, by1 in bboxes:
        if (
            bx0 - slack <= lcx <= bx1 + slack
            and by0 - slack <= lcy <= by1 + slack
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Document-level scan


def scan_document(
    pdf_path: str,
    first_page_idx: int,
    last_page_idx: int,
) -> list[dict]:
    """
    Walk every content page and return a flat list of page records:

        {
            "pdf_page": pdf_idx,
            "printed_page": pdf_idx - PAGE_OFFSET,
            "lines": [ line dict, ... ],
            "table_bboxes": [ bbox, ... ],
        }

    We read the whole doc once up front so the section-splitting pass has
    O(pages) access without re-parsing each PDF page.
    """
    doc = pymupdf.open(pdf_path)
    pages: list[dict] = []
    for pi in range(first_page_idx, last_page_idx + 1):
        page = doc[pi]
        lines = _collect_page_lines(page)
        tbboxes = _collect_table_bboxes(page)
        pages.append(
            {
                "pdf_page": pi,
                "printed_page": pi - PAGE_OFFSET,
                "lines": lines,
                "table_bboxes": tbboxes,
            }
        )
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# TOC -> body heading matching


def _heading_candidates(page_record: dict) -> list[dict]:
    """
    Return lines on this page that look like section headings.

    A heading is bold, size >= MIN_HEADING_SIZE, not a running header, not
    a figure/table caption, not a pure page number.
    """
    out = []
    for ln in page_record["lines"]:
        if not ln["bold"]:
            continue
        if ln["size"] < MIN_HEADING_SIZE:
            continue
        t = ln["text"]
        if RUNNING_HEADER_RE.search(t):
            continue
        if CAPTION_RE.match(t):
            continue
        if re.fullmatch(r"\d{1,4}", t):
            continue
        out.append(ln)
    return out


def _strip_leading_number(text_norm: str) -> str:
    """
    Strip a normalized leading section-number token from a body heading.

    Handles the forms we see in the spec:
      "1 5 definitions"              -> "definitions"
      "b 5 1 shadow doorbell ..."    -> "shadow doorbell ..."
      "annex a sanitize ..."         -> "sanitize ..."   (annex prefix)

    Note normalization has already dropped punctuation, so "B.5.1." became
    "b 5 1" and "Annex A." became "annex a".
    """
    # Annex prefix first (longest first)
    s = re.sub(r"^annex\s+[a-z]\s+", "", text_norm)
    # Optional letter then run of digit groups
    s = re.sub(r"^([a-z]\s+)?\d+(?:\s+\d+)*\s+", "", s)
    return s


def _match_heading(
    toc_entry: dict,
    page_records: list[dict],
    page_idx_to_offset: dict[int, int],
    start_from: tuple[int, float],
) -> tuple[int, float] | None:
    """
    Find the (pdf_page, y) position of `toc_entry`'s body heading.

    Starts scanning from `start_from` (so headings are matched strictly in
    document order) and looks up to HEADING_SEARCH_WINDOW_PAGES pages past
    the TOC target_page.

    Matching logic:
      - Build a normalized heading text, strip any leading section-number
        or annex prefix
      - If the normalized heading line alone doesn't match the TOC title,
        try concatenating with the next 1-2 bold lines on the same page
        (handles headings that wrap across visual lines, e.g. Figure 143's
        "NVM Subsystem Sanitize Operation and Format NVM Admin Command
        Processing Restrictions" split across 2 lines)
      - Accept exact match, or TOC title is a prefix of the body heading
        (handles "(Informative)" suffixes on annex headings)
    """
    want_title = _norm_title(toc_entry["title"])
    target_printed = toc_entry["target_page"]
    want_pdf_page = target_printed + PAGE_OFFSET

    start_pi, start_y = start_from
    lo = max(start_pi, want_pdf_page - 1)
    hi = want_pdf_page + HEADING_SEARCH_WINDOW_PAGES

    best: tuple[int, float] | None = None
    for pi in range(lo, hi + 1):
        off = page_idx_to_offset.get(pi)
        if off is None:
            continue
        page_rec = page_records[off]
        candidates = _heading_candidates(page_rec)
        for i, ln in enumerate(candidates):
            ly = ln["bbox"][1]
            if pi == start_pi and ly <= start_y:
                continue
            if pi < start_pi:
                continue
            # Try 1, 2, 3 line concatenations starting at this position.
            # The returned y is the TOP of the first heading line so it
            # also serves as the END boundary of the previous section (the
            # previous section should not swallow this heading).
            for span_len in (1, 2, 3):
                if i + span_len > len(candidates):
                    break
                group = candidates[i : i + span_len]
                combined = " ".join(g["text"] for g in group)
                text_norm = _norm_title(combined)
                stripped = _strip_leading_number(text_norm)

                if text_norm == want_title or stripped == want_title:
                    return (pi, ly)
                # TOC title is prefix of body heading (annex "(Informative)")
                if stripped.startswith(want_title + " ") or text_norm.startswith(
                    want_title + " "
                ):
                    if best is None:
                        best = (pi, ly)
                # Body heading is prefix of TOC title (wrapped heading)
                if (
                    want_title.startswith(stripped + " ") and len(stripped) > 0
                ):
                    if best is None:
                        best = (pi, ly)
    return best


def resolve_section_bounds(
    toc: list[dict],
    page_records: list[dict],
) -> list[dict]:
    """
    For each TOC entry, find its body heading position and attach:
        _start: (pdf_page, y) start-of-section (just after the heading line)
        _end:   (pdf_page, y) end-of-section (= next section's start, or EOD)
        _heading_line: the heading line dict itself (so we can exclude it)

    Entries whose heading we couldn't find are kept but flagged with
    `_missing: True` — the validation pass reports these.
    """
    # Fast lookup: pdf_page_idx -> index into page_records
    pi2off = {pr["pdf_page"]: i for i, pr in enumerate(page_records)}

    resolved: list[dict] = []
    cursor = (page_records[0]["pdf_page"], -1.0)  # before any line

    for entry in toc:
        hit = _match_heading(entry, page_records, pi2off, cursor)
        rec = dict(entry)
        if hit is None:
            rec["_missing"] = True
            rec["_start"] = None
        else:
            rec["_start"] = hit
            cursor = hit
        resolved.append(rec)

    # Fill _end as next resolved _start, or end of doc for the last one.
    eod = (
        page_records[-1]["pdf_page"],
        float("inf"),
    )
    for i, rec in enumerate(resolved):
        if rec.get("_missing"):
            rec["_end"] = None
            continue
        nxt = None
        for j in range(i + 1, len(resolved)):
            if not resolved[j].get("_missing"):
                nxt = resolved[j]["_start"]
                break
        rec["_end"] = nxt if nxt is not None else eod

    return resolved


# ---------------------------------------------------------------------------
# Prose assembly


def _iter_lines_in_range(
    start: tuple[int, float],
    end: tuple[int, float],
    page_records: list[dict],
    pi2off: dict[int, int],
):
    """
    Yield (page_record, line) for every line whose visual position is in
    [start, end). Crossing page boundaries is handled naturally.
    """
    spi, sy = start
    epi, ey = end
    for pi in range(spi, epi + 1):
        off = pi2off.get(pi)
        if off is None:
            continue
        pr = page_records[off]
        for ln in pr["lines"]:
            ly = ln["bbox"][1]
            if pi == spi and ly <= sy:
                continue
            if pi == epi and ly >= ey:
                continue
            yield pr, ln


def _line_is_noise(line: dict) -> bool:
    """
    Return True if this visual line should be dropped entirely from prose.

    We filter:
      - Running page headers (exact top-of-page "NVM Express ... Revision 2.3")
      - Standalone printed page numbers
      - Figure/Table captions — these are tagged by 1.2 table extractor
        and would duplicate content if kept in prose
      - Bold 10pt+ heading lines — these are section or sub-section headings
        (numbered or not) and the heading-matching pass already tracks them.
        Leaving them out prevents heading continuation lines from leaking
        in as the first "paragraph" of a section (see section 5.1.1 where
        the title wraps across two lines).
      - Pure whitespace artifacts
    """
    t = line["text"]
    bbox = line["bbox"]
    if bbox[1] < 90 and RUNNING_HEADER_RE.search(t):
        return True
    if re.fullmatch(r"\d{1,4}", t):
        return True
    if CAPTION_RE.match(t):
        return True
    # Bold heading-style lines. The heading-match pass uses only exact
    # matches against the TOC; continuation wrap lines don't match but
    # are still bold, so this blanket filter drops them cleanly.
    if line["bold"] and line["size"] >= MIN_HEADING_SIZE:
        return True
    if not t.strip():
        return True
    return False


def _group_paragraphs(lines_with_page: list[tuple[dict, dict]]) -> list[dict]:
    """
    Group a sequence of (page_record, line) tuples into paragraphs.

    A paragraph break is introduced when:
      - we cross a page boundary, OR
      - the block_no changes, OR
      - the vertical gap between two consecutive lines on the same page
        is larger than ~1.8x the line height (a blank-line equivalent)

    Each paragraph keeps the pdf_page of its first line for citation.
    """
    paragraphs: list[dict] = []
    cur_lines: list[str] = []
    cur_pdf_page: int | None = None
    prev_line: dict | None = None
    prev_page_idx: int | None = None
    prev_block: int | None = None

    def flush():
        nonlocal cur_lines, cur_pdf_page
        if cur_lines:
            text = " ".join(cur_lines).strip()
            text = re.sub(r"\s+", " ", text)
            if text:
                paragraphs.append({"text": text, "pdf_page": cur_pdf_page})
        cur_lines = []
        cur_pdf_page = None

    for pr, ln in lines_with_page:
        pi = pr["pdf_page"]
        bbox = ln["bbox"]
        line_height = max(1.0, bbox[3] - bbox[1])
        start_new = False
        if prev_line is None:
            start_new = True
        else:
            if pi != prev_page_idx:
                start_new = True
            elif ln["block_no"] != prev_block:
                start_new = True
            else:
                gap = bbox[1] - prev_line["bbox"][3]
                if gap > line_height * 0.9:
                    start_new = True
        if start_new:
            flush()
            cur_pdf_page = pi
        cur_lines.append(ln["text"])
        prev_line = ln
        prev_page_idx = pi
        prev_block = ln["block_no"]

    flush()
    return paragraphs


# ---------------------------------------------------------------------------
# Normative tagging


# Split paragraph into sentences. Keep it simple — the spec uses normal
# English punctuation with lots of parentheticals and "i.e.,"/"e.g.," so a
# naive split is fine. We split on ". " / "? " / "! " followed by a capital
# letter or the end of string.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _extract_normative(
    paragraphs: list[dict], section_number: str
) -> list[dict]:
    """
    Scan paragraphs for sentences that contain a normative keyword
    (shall / should / may). Returns a list of:

        { "strength": "shall", "text": "...", "pdf_page": N }

    A sentence can have multiple keywords; we tag with the strongest found
    (shall > should > may) to match NVMe conformance semantics.

    Sections 1.4.1 (Keywords) and 1.4.2 (Numerical Descriptions) are skipped:
    they DEFINE the normative keywords rather than make normative claims,
    so every "shall/should/may" mention there is a false positive.

    Sentences shorter than 4 words are dropped — these are usually a bare
    keyword definition header ("may", "shall") picked up as its own "paragraph".
    """
    # Skip keyword-definition sections entirely
    if section_number in ("1.4.1", "1.4.2"):
        return []

    out: list[dict] = []
    for para in paragraphs:
        for sent in SENTENCE_SPLIT_RE.split(para["text"]):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent.split()) < 4:
                continue
            strength = None
            for word in NORMATIVE_WORDS:
                if re.search(rf"\b{word}\b", sent, re.IGNORECASE):
                    strength = word
                    break  # NORMATIVE_WORDS ordered strongest-first
            if strength:
                out.append(
                    {
                        "strength": strength,
                        "text": sent,
                        "pdf_page": para["pdf_page"],
                    }
                )
    return out


# ---------------------------------------------------------------------------
# Public API


def extract_prose(
    pdf_path: str,
    toc_path: str,
    first_page_idx: int = DEFAULT_FIRST_CONTENT_PAGE,
    last_page_idx: int | None = None,
) -> tuple[list[dict], dict[str, str]]:
    """
    Main entry point. Returns (sections, definitions).

    - `sections` is a list of per-TOC-entry dicts with prose + normative tags.
    - `definitions` is a {term: definition} lookup built from section 1.5.x.
    """
    with open(toc_path, encoding="utf-8") as f:
        toc = json.load(f)

    # Open briefly just to discover page count if needed.
    doc = pymupdf.open(pdf_path)
    if last_page_idx is None:
        last_page_idx = doc.page_count - 1
    doc.close()

    page_records = scan_document(pdf_path, first_page_idx, last_page_idx)
    resolved = resolve_section_bounds(toc, page_records)
    pi2off = {pr["pdf_page"]: i for i, pr in enumerate(page_records)}

    sections: list[dict] = []
    for rec in resolved:
        base = {
            "section_number": rec["section_number"],
            "title": rec["title"],
            "level": rec["level"],
            "target_page": rec["target_page"],
        }
        if rec.get("_missing"):
            sections.append(
                {
                    **base,
                    "missing_heading": True,
                    "paragraphs": [],
                    "normative": [],
                    "start_pdf_page": None,
                    "end_pdf_page": None,
                }
            )
            continue

        start = rec["_start"]
        end = rec["_end"]

        lines_in_range: list[tuple[dict, dict]] = []
        for pr, ln in _iter_lines_in_range(start, end, page_records, pi2off):
            if _line_is_noise(ln):
                continue
            if _in_any_bbox(ln["bbox"], pr["table_bboxes"]):
                continue
            lines_in_range.append((pr, ln))

        paragraphs = _group_paragraphs(lines_in_range)
        normative = _extract_normative(paragraphs, rec["section_number"])

        sections.append(
            {
                **base,
                "start_pdf_page": start[0],
                "end_pdf_page": end[0] if end[0] != float("inf") else None,
                "paragraphs": paragraphs,
                "normative": normative,
            }
        )

    definitions = _build_definitions(sections)
    return sections, definitions


def _build_definitions(sections: list[dict]) -> dict[str, str]:
    """
    Walk the 1.5.x subsections (each one is a single defined term) and
    flatten them into a {term: definition} lookup. The term is the section
    title; the definition is all paragraphs joined.
    """
    out: dict[str, str] = {}
    for s in sections:
        sn = s["section_number"]
        if not sn.startswith("1.5."):
            continue
        # Skip 1.5 parent itself; only leaf subsections
        if sn.count(".") < 1:
            continue
        term = s["title"].strip()
        text = " ".join(p["text"] for p in s["paragraphs"]).strip()
        if term and text:
            out[term] = text
    return out


# ---------------------------------------------------------------------------
# Validation helpers


def summarize(sections: list[dict]) -> None:
    """
    Print a terse validation summary to stdout: heading-match rate, prose
    coverage, normative tag counts, and a few spot-check lines.
    """
    total = len(sections)
    missing = sum(1 for s in sections if s.get("missing_heading"))
    resolved = total - missing
    empty_prose = sum(
        1
        for s in sections
        if not s.get("missing_heading") and not s["paragraphs"]
    )
    total_paragraphs = sum(len(s["paragraphs"]) for s in sections)
    total_normative = sum(len(s["normative"]) for s in sections)
    strengths: dict[str, int] = {"shall": 0, "should": 0, "may": 0}
    for s in sections:
        for n in s["normative"]:
            strengths[n["strength"]] = strengths.get(n["strength"], 0) + 1

    print(f"sections:        {total}")
    print(f"  heading matched: {resolved}")
    print(f"  heading missing: {missing}")
    print(f"  empty prose:     {empty_prose}")
    print(f"paragraphs:      {total_paragraphs}")
    print(f"normative tags:  {total_normative}")
    for k in ("shall", "should", "may"):
        print(f"  {k}: {strengths.get(k, 0)}")

    if missing:
        print()
        print("first 10 missing:")
        shown = 0
        for s in sections:
            if s.get("missing_heading"):
                print(
                    f"  {s['section_number']:<10} p.{s['target_page']:<4} {s['title'][:60]}"
                )
                shown += 1
                if shown >= 10:
                    break


# ---------------------------------------------------------------------------
# CLI

if __name__ == "__main__":
    PDF_PATH = "nvme_spec/NVMe_spec_full.pdf"
    TOC_PATH = "data/toc.json"
    SECTIONS_OUT = "data/prose.json"
    DEFS_OUT = "data/definitions.json"

    first_arg = sys.argv[1] if len(sys.argv) > 1 else None
    last_arg = sys.argv[2] if len(sys.argv) > 2 else None
    first_pi = int(first_arg) if first_arg else DEFAULT_FIRST_CONTENT_PAGE
    last_pi = int(last_arg) if last_arg else None

    sections, definitions = extract_prose(
        PDF_PATH, TOC_PATH, first_page_idx=first_pi, last_page_idx=last_pi
    )

    summarize(sections)

    os.makedirs("data", exist_ok=True)
    with open(SECTIONS_OUT, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {SECTIONS_OUT}")

    with open(DEFS_OUT, "w", encoding="utf-8") as f:
        json.dump(definitions, f, indent=2, ensure_ascii=False)
    print(f"wrote {DEFS_OUT} ({len(definitions)} terms)")
