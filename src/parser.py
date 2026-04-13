##==================================================================================##
##                                                                                  ##
## this file parses and stores the spec into an indexed and machine readable format ##
##                                                                                  ##
##==================================================================================##

import json
import re
from pathlib import Path

import pymupdf


# ---------------------------------------------------------------------------- #
# Tunable constants                                                            #
# ---------------------------------------------------------------------------- #

# Regex for a TOC entry: title text, 3+ dot leader chars, trailing int.
# Example match: "idempotent command ........................... 10"
#   group(1) = "idempotent command"
#   group(2) = "10"
TOC_LINE_RE = re.compile(r"^(.*?)\s*\.{3,}\s*(\d+)\s*$")

# Strip a leading section number from a title, e.g. "1 INTRODUCTION" -> "INTRODUCTION",
# "B.5.1. Shadow Doorbell" -> "Shadow Doorbell". Matches optional leading letter
# (for appendices A/B/C/...), optional dot, digits with dotted sub-parts, optional
# trailing dot, then whitespace, then the real title.
LEADING_SECTION_NUM_RE = re.compile(r"^[A-Z]?\.?\d+(?:\.\d+)*\.?\s+(.+)$")

# Reject lines from the "List of Figures" / "List of Tables" sections of the TOC.
# These entries all start with "Figure N:" or "Table N:" and are NOT real sections.
# Filtering them here keeps them out of the main TOC entirely.
FIGURE_TABLE_RE = re.compile(r"^(Figure|Table)\s+\d+", re.IGNORECASE)

# Detect an "ANNEX X." prefix on an L1 chapter title so we can number annexes
# as A/B/C... instead of continuing the integer count (10, 11, 12). We also
# strip the prefix from the title so the title reads cleanly.
ANNEX_RE = re.compile(r"^ANNEX\s+([A-Z])\.\s*(.+)$", re.IGNORECASE)

# Y-coordinate tolerance (PDF points) when bucketing words into visual lines.
# Superscripts sit on a slightly different baseline than surrounding text, so
# we allow some slack. 3pt works for body text around 9-10pt.
Y_TOLERANCE = 3.0

# Explicit x0 thresholds for hierarchy level assignment in the NVMe spec TOC.
# Derived empirically from the observed x0 distribution:
#   ~72pt  -> level 1 (top-level chapters: "1 INTRODUCTION")
#   ~82pt  -> level 2 (sections: "1.1 Overview")
#   >=87pt -> level 3 (subsections, definitions, appendix; multiple x0s at
#            92, 132, 142 all collapse here because the NVMe TOC has at most
#            3 logical levels and these are layout variants)
# If you ever run this on a different spec version, sanity-check the x0
# distribution with the [debug] prints and re-tune these thresholds.
LEVEL_X0_THRESHOLDS = [
    (77.0, 1),   # x0 <  77  -> level 1
    (87.0, 2),   # x0 <  87  -> level 2
    (float("inf"), 3),  # everything else -> level 3
]

# Max hierarchy depth we'll emit. The NVMe spec TOC is capped at 3 levels,
# but we keep this as a constant so validation can catch any bugs that would
# produce deeper section numbers like "1.1.1.1".
MAX_LEVEL = 3


# ---------------------------------------------------------------------------- #
# Public entry point                                                           #
# ---------------------------------------------------------------------------- #

def parse_toc(
    pdf_path: str,
    first_toc_page: int,
    last_toc_page: int,
) -> list[dict]:
    """
    Parse the table of contents from the given PDF.

    Args:
        pdf_path:       path to the PDF file
        first_toc_page: 0-indexed page number where the TOC starts
        last_toc_page:  0-indexed page number where the TOC ends (inclusive)

    Returns:
        A list of dicts, one per TOC entry, in document order. See the module
        docstring for the schema of each dict.
    """
    # Pass 1: walk pages, reconstruct lines, parse each one into a raw entry.
    # We don't know hierarchy levels yet -- we stash the raw x0 and fill level
    # + section_number in pass 2 once we can cluster all x0s across all pages.
    raw_entries: list[dict] = []

    with pymupdf.open(pdf_path) as doc:
        for page_idx in range(first_toc_page, last_toc_page + 1):
            page = doc[page_idx]
            for line_text, leftmost_x0 in _reconstruct_lines(page):
                parsed = _parse_toc_line(line_text)
                if parsed is None:
                    continue  # skip headers, footers, noise
                title, target_page = parsed
                raw_entries.append({
                    "title":       title,
                    "x0":          leftmost_x0,
                    "target_page": target_page,
                })

    if not raw_entries:
        return []

    # Pass 2: discover hierarchy levels by clustering all observed x0 values.
    level_of_x0 = _compute_level_thresholds([e["x0"] for e in raw_entries])

    # Debug visibility: print how many hierarchy levels we discovered and the
    # x0 range of each one, plus the entry count per level and sample titles.
    discovered_levels = sorted(set(level_of_x0.values()))
    print(f"[debug] discovered {len(discovered_levels)} hierarchy level(s):")
    for lvl in discovered_levels:
        xs = [x for x, l in level_of_x0.items() if l == lvl]
        entries_at_level = [e for e in raw_entries if level_of_x0[e["x0"]] == lvl]
        print(f"[debug]   L{lvl}: x0 in [{min(xs):.1f}, {max(xs):.1f}]  "
              f"({len(entries_at_level)} entries)")
        # show first 5 and a few from the middle/end so we catch any noise
        to_show = list(entries_at_level[:5])
        if len(entries_at_level) > 10:
            mid = len(entries_at_level) // 2
            to_show += entries_at_level[mid:mid+3]
            to_show += entries_at_level[-3:]
        for e in to_show:
            print(f"[debug]     example: {e['title'][:70]}  (x0={e['x0']:.1f}, p.{e['target_page']})")

    # Pass 3: walk entries in order, running a counter stack to compute
    # section numbers. Emit the final list.
    #
    # Annex handling: when an L1 entry matches "ANNEX X. ...", we switch the
    # L1 counter from an int to the letter X, and strip "ANNEX X." from the
    # title. Sublevels still use ints so we get A.1, A.1.1, etc.
    counters: list = []
    entries: list[dict] = []
    for raw in raw_entries:
        level = level_of_x0[raw["x0"]]
        title = raw["title"]

        annex_m = ANNEX_RE.match(title) if level == 1 else None
        if annex_m:
            letter = annex_m.group(1).upper()
            title = annex_m.group(2).strip()
            counters = [letter]
        else:
            _bump_counters(counters, level)

        section_number = ".".join(str(c) for c in counters)
        entries.append({
            "section_number": section_number,
            "title":          title,
            "level":          level,
            "target_page":    raw["target_page"],
        })

    return entries


# ---------------------------------------------------------------------------- #
# Helpers                                                                      #
# ---------------------------------------------------------------------------- #

def _reconstruct_lines(page) -> list[tuple[str, float]]:
    """
    Pull all words from a page and reconstruct them into visual lines by
    bucketing on y-coordinate. Returns (line_text, leftmost_x0) tuples in
    reading order (top-to-bottom).

    More robust than page.get_text() family methods because it ignores
    PyMuPDF's block/line grouping, which is unreliable on this PDF.
    """
    words = page.get_text("words")
    if not words:
        return []

    # Word tuple format: (x0, y0, x1, y1, "word_text", block_no, line_no, word_no)
    # We ignore block_no/line_no and bucket by the y-center of each word.

    def y_center(w) -> float:
        return (w[1] + w[3]) / 2

    # Sort by y-center primarily so we can do a single linear bucketing pass.
    words_sorted = sorted(words, key=lambda w: (y_center(w), w[0]))

    lines: list[list[tuple]] = []
    for w in words_sorted:
        if lines and abs(y_center(lines[-1][0]) - y_center(w)) < Y_TOLERANCE:
            lines[-1].append(w)
        else:
            lines.append([w])

    result: list[tuple[str, float]] = []
    for line in lines:
        # Re-sort within each line strictly by x0 to guarantee left-to-right.
        line.sort(key=lambda w: w[0])
        line_text = " ".join(w[4] for w in line)
        leftmost_x0 = line[0][0]
        result.append((line_text, leftmost_x0))

    return result


def _parse_toc_line(line: str) -> tuple[str, int] | None:
    """
    Try to parse a line as a TOC entry. Returns (title, target_page) on
    success, None if the line doesn't match the TOC format (header, footer,
    noise).
    """
    stripped = line.strip()
    if len(stripped) < 3:
        return None

    m = TOC_LINE_RE.match(stripped)
    if not m:
        return None

    title = m.group(1).strip()
    if not title:
        return None

    # Drop "List of Figures" / "List of Tables" entries — not real sections.
    if FIGURE_TABLE_RE.match(title):
        return None

    # Strip leading section number like "1 INTRODUCTION" -> "INTRODUCTION".
    sm = LEADING_SECTION_NUM_RE.match(title)
    if sm:
        title = sm.group(1).strip()

    if not title:
        return None

    try:
        target_page = int(m.group(2))
    except ValueError:
        return None

    return title, target_page


def _compute_level_thresholds(x0_values: list[float]) -> dict[float, int]:
    """
    Map each observed x0 value to a hierarchy level using the explicit
    LEVEL_X0_THRESHOLDS table. Returns { x0 -> level }, 1-indexed.

    We don't auto-cluster because the NVMe TOC has noisy layout variants
    (extra x0s at 92, 132, 142 that are all logically level 3). Explicit
    thresholds are more predictable and easier to debug.
    """
    level_of_x0: dict[float, int] = {}
    for x in set(x0_values):
        for upper, lvl in LEVEL_X0_THRESHOLDS:
            if x < upper:
                level_of_x0[x] = min(lvl, MAX_LEVEL)
                break
    return level_of_x0


def _bump_counters(counters: list[int], level: int) -> None:
    """
    Update the hierarchy counter stack in-place for a new entry at `level`.

    Three cases:
      1. level > len(counters)  -> descended deeper: push a new 1
                                   (pad with 1s if the TOC skipped levels).
      2. level == len(counters) -> same level: increment the top counter.
      3. level <  len(counters) -> popped back up: truncate to `level`
                                   entries, then increment the top counter.
    """
    if level > len(counters):
        while len(counters) < level - 1:
            counters.append(1)
        counters.append(1)
    elif level == len(counters):
        counters[-1] += 1
    else:
        del counters[level:]
        counters[-1] += 1


# ---------------------------------------------------------------------------- #
# Script entry point                                                           #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    # ----- tune these for your PDF -----
    PDF_PATH       = "nvme_spec/NVMe_spec_TOC.pdf"
    FIRST_TOC_PAGE = 2    # 0-indexed; pages before this are skipped
    LAST_TOC_PAGE  = 22   # 0-indexed inclusive; tune until all TOC is captured
    OUTPUT_PATH    = "data/toc.json"
    # ------------------------------------

    entries = parse_toc(PDF_PATH, FIRST_TOC_PAGE, LAST_TOC_PAGE)

    print(f"parsed {len(entries)} TOC entries")
    print()
    print("first 20 entries:")
    for e in entries[:20]:
        sec   = e["section_number"]
        title = e["title"]
        if len(title) > 55:
            title = title[:52] + "..."
        print(f"  {sec:<10} L{e['level']}  {title:<55} -> p.{e['target_page']}")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print()
    print(f"wrote {OUTPUT_PATH}")
