"""
NVMe spec table extractor.

Pulls "Figure N:" tables out of the NVMe base spec PDF into a list of
structured JSON objects. NVMe calls *all* tables "Figure N:" even though
they're tabular — that's what we key off.

Output shape (one entry per logical table, multi-page tables merged):

    {
        "figure_number": 328,
        "caption": "Identify – Identify Controller Data Structure, I/O Command Set Independent",
        "printed_page": 322,
        "pdf_page": 345,
        "headers": ["Bytes", "I/O", "Admin", "Disc", "Description"],
        "rows": [
            ["76", "O", "O", "R", "Controller Multi-Path I/O ..."],
            ["77", "M", "M", "M", "Maximum Data Transfer Size ..."],
            ...
        ],
        "raw_text": "<original page text slice for this table>",
    }

Uses PyMuPDF's page.find_tables(), then:
  1. Filters out nested tables (bbox contained inside another table's bbox —
     these are sub-tables inside description cells that are already present
     verbatim in the parent cell's text).
  2. Collapses "snow" columns: find_tables() splits each logical column into
     2–3 physical columns because of drawn cell borders. We drop None/empty
     cells in each row, preserving order, which leaves one cell per logical
     column.
  3. Attaches the "Figure N:" caption from the text immediately above the
     table bbox.
  4. Merges continuations: tables on page N+1 with no caption above them are
     treated as continuations of the last captioned table.

This is intentionally tolerant — if a table is weird, we still emit it with
whatever shape we got, and the raw_text field lets downstream callers fall
back to plain text search.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pymupdf


# ---------------------------------------------------------------------------
# Constants

# NVMe uses "Figure N: caption" for ALL tables. This captures the whole line
# up to the newline so we can record the caption text.
FIGURE_CAPTION_RE = re.compile(r"Figure\s+(\d+):\s*(.+)")

# Spec pages carry a running footer like "NVM Express® Base Specification,
# Revision 2.3" and a page number. Strip these from raw_text so they don't
# pollute the raw_text field when we need it for search.
RUNNING_HEADER_RE = re.compile(
    r"NVM Express.*?Revision [\d.]+\s*",
    re.IGNORECASE,
)

# How far above the table bbox to look for its "Figure N:" caption.
# 80pt is generous — captions typically sit within 20pt of the first row.
CAPTION_LOOKUP_DY = 80.0


# ---------------------------------------------------------------------------
# Row collapsing


def _clean_header_cell(cell: str) -> str:
    """
    Normalize a header cell: collapse internal newlines to spaces, strip
    stray footnote markers, squeeze whitespace.

    Footnote markers look like ` 1 ` between a column word and a group word
    (e.g., "Administrative 1 Controller"). We remove them anywhere in the
    string when they appear adjacent to a capitalized word on the right.
    Leading single-digit markers ("1 Controller") are also stripped.
    """
    s = cell.replace("\n", " ").strip()
    # Leading footnote digit: "1 Controller" -> "Controller"
    s = re.sub(r"^\d\s+(?=[A-Z])", "", s)
    # Interior footnote digit: "Size 1 Controller" -> "Size Controller"
    s = re.sub(r"\s\d\s+(?=[A-Z])", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _clean_data_cell(cell: str) -> str:
    """
    Normalize a data cell: strip a leading footnote marker that got glued to
    an M/O/R/P requirement letter (e.g., "3\nM" -> "M"). Leaves regular
    prose cells alone except for whitespace tidying.
    """
    s = cell.replace("\n", " ").strip()
    # "3 M", "5 O" at start when cell is very short (just footnote + letter)
    m = re.match(r"^\d+[,\d]*\s+([MORP](?:/[MORP])*)$", s)
    if m:
        return m.group(1)
    s = re.sub(r"\s+", " ", s)
    return s


def _cell_is_empty(cell) -> bool:
    if cell is None:
        return True
    if isinstance(cell, str) and not cell.strip():
        return True
    return False


def _merge_raw_header_rows(row_a: list, row_b: list) -> list:
    """
    Merge two raw (un-collapsed) find_tables rows by position. Used to fold
    a sub-header row (e.g., ['', '', '', 'I/O', '', '', 'Admin', ...]) into
    the main header row, preserving original column positions.

    At each position:
      - if row_b has content, use row_b (optionally prefixed by row_a content
        when both are present), else use row_a
      - None/empty propagates as None

    This keeps row width constant at find_tables' detected grid width, which
    is important: any downstream `_collapse_row` pass sees the same grid for
    header and data rows, so they line up.
    """
    n = max(len(row_a), len(row_b))
    out = []
    for i in range(n):
        a = row_a[i] if i < len(row_a) else None
        b = row_b[i] if i < len(row_b) else None
        if _cell_is_empty(b):
            out.append(a)
        elif _cell_is_empty(a):
            out.append(b)
        else:
            # Both present: b is the sub-header, a is the group label.
            out.append(f"{a} {b}")
    return out


# A "primary key" cell is the first column of a data row — in NVMe tables this
# is a byte/bit offset ("385", "391:390", "31:16") or a hex value ("0h", "Ch",
# "01b"). When we see a short row starting with one of these, we know it's a
# NEW data row whose middle columns happened to come back empty (not a sub-row
# to fold into the previous row's description cell).
PRIMARY_KEY_RE = re.compile(
    r"^(?:\d+(?::\d+)?|[0-9A-Fa-f]+h|[01]+b)$"
)


def _looks_like_primary_key(cell: str) -> bool:
    return bool(PRIMARY_KEY_RE.match(cell.strip()))


def _looks_like_band_header(row: list[str]) -> bool:
    """
    A band header is a single-cell row inside a table that acts as a section
    divider for the rows below it (e.g., Figure 328 has rows like "Controller
    Capabilities and Features" spanning all columns).

    Exactly one cell, >= 10 chars, multiple words, mostly letters/spaces —
    this distinguishes from header fragments like "bytes)".
    """
    if len(row) != 1:
        return False
    s = row[0]
    if len(s) < 10 or len(s.split()) < 2:
        return False
    letter_frac = sum(c.isalpha() or c.isspace() for c in s) / max(1, len(s))
    return letter_frac > 0.85


def _build_header_and_rows(
    raw_rows: list[list],
) -> tuple[list[str], list[list[str]]]:
    """
    Take find_tables' raw row grid and return (headers, rows) in collapsed
    logical-column form.

    Strategy:
      1. Collapse data rows to determine target column count (the mode of
         data row widths beyond row 0).
      2. Start with raw row 0 as the header.
      3. If collapsed(header) has fewer cells than target, position-merge
         raw row 1 into the header. Re-check. Try up to 3 header rows total
         (main + subgroup + sub-subgroup is the worst case in NVMe).
      4. Data rows = everything after the header block, each collapsed.

    This handles the common NVMe pattern where the table has a group header
    like "Controller Support Requirements" over 3 sub-columns (I/O / Admin /
    Disc), which find_tables exposes as two separate raw rows.
    """
    if not raw_rows:
        return [], []

    # Step 1: estimate the true column count by looking past the header block.
    candidate_widths = [
        len(_collapse_row(r)) for r in raw_rows[1:]
    ]
    # Drop pure single-cell rows (band headers) when computing the mode —
    # they shouldn't bias the count downward.
    non_band = [w for w in candidate_widths if w > 1]
    if non_band:
        target_cols = max(set(non_band), key=non_band.count)
    else:
        target_cols = len(_collapse_row(raw_rows[0]))

    # Step 2: build the header row, folding in sub-header rows as needed.
    header_raw = raw_rows[0]
    header_rows_consumed = 1
    for extra in raw_rows[1:4]:  # at most 3 additional rows
        if len(_collapse_row(header_raw)) >= target_cols:
            break
        # Only fold if `extra` looks header-ish: more than 1 cell and no cell
        # looks like real data (avoid eating "00h", numeric ids, etc.)
        extra_collapsed = _collapse_row(extra)
        if len(extra_collapsed) == 0:
            header_rows_consumed += 1
            continue
        # Data row sentinel: contains a hex byte offset, a length number,
        # or looks like a full data row (count matches target).
        if len(extra_collapsed) == target_cols:
            break
        header_raw = _merge_raw_header_rows(header_raw, extra)
        header_rows_consumed += 1

    headers = [_clean_header_cell(h) for h in _collapse_row(header_raw)]
    rows = [
        [_clean_data_cell(c) for c in _collapse_row(r)]
        for r in raw_rows[header_rows_consumed:]
    ]
    rows = [r for r in rows if r]

    # Drop leftover header-fragment rows that appear before the first full
    # data row. These are rows like ['(in'] or ['bytes)'] that belong to a
    # multi-line header we didn't fully absorb above. We only peel off rows
    # at the very start, and we preserve band headers (single prose cell).
    while rows:
        first = rows[0]
        if len(first) == target_cols or _looks_like_band_header(first):
            break
        # Append fragment content into the nearest-matching header cell so
        # no information is dropped entirely.
        for i, frag in enumerate(first):
            if i < len(headers):
                headers[i] = _clean_header_cell(headers[i] + " " + frag)
        rows = rows[1:]

    # Walk rows and decide: is a short row (a) a new primary data row whose
    # middle columns just happened to be empty, or (b) a nested sub-row that
    # should be folded into the previous row's description cell?
    #
    # Heuristic: if the first cell looks like a primary key (byte/bit offset,
    # hex value), it's a new row — PAD with empty strings so column alignment
    # is preserved. Otherwise, fold into the previous parent row's last cell.
    #
    # This is what lets Figure 328 byte 391:390 stay as its own row even when
    # its I/O / Admin / Disc cells come back as empty strings instead of None.
    merged_rows: list[list[str]] = []
    for r in rows:
        if len(r) == target_cols or _looks_like_band_header(r):
            merged_rows.append(r)
            continue
        if r and _looks_like_primary_key(r[0]) and len(r) < target_cols:
            # New primary row: pad the middle with empty strings, keeping the
            # last cell as the description. If only 1 cell, leave description
            # empty.
            if len(r) == 1:
                padded = [r[0]] + [""] * (target_cols - 1)
            else:
                padded = [r[0]] + [""] * (target_cols - len(r)) + list(r[1:])
            merged_rows.append(padded)
            continue
        if merged_rows and len(merged_rows[-1]) == target_cols:
            sub_text = " ".join(r)
            parent = merged_rows[-1]
            parent[-1] = (parent[-1] + "\n" + sub_text).strip()
        else:
            # No full-width parent to fold into — keep as-is so nothing is lost.
            merged_rows.append(r)

    return headers, merged_rows


def _collapse_row(row: list) -> list[str]:
    """
    Collapse a find_tables() row that has been exploded across phantom columns
    into one cell per logical column.

    find_tables() often yields rows like:
        ['76', None, None, 'O', None, None, 'O', None, None, 'R', None, None, 'desc', None, None]

    The pattern is: one non-None value followed by a run of Nones (the
    merged-cell continuation markers). We keep the non-None values in order.

    We also treat empty strings '' as None, because find_tables() uses '' for
    empty header cells whereas it uses None for merged-cell continuations.
    Both mean "no data here".
    """
    out: list[str] = []
    for cell in row:
        if cell is None:
            continue
        if isinstance(cell, str):
            s = cell.strip()
            if not s:
                continue
            out.append(s)
        else:
            out.append(str(cell))
    return out


# ---------------------------------------------------------------------------
# Caption lookup


def _find_caption_above(page: pymupdf.Page, table_bbox) -> tuple[int, str] | None:
    """
    Look for a 'Figure N: caption' line whose y position sits just above the
    given table bbox. Returns (figure_number, caption_text) or None.

    We scan text blocks on the page, find any that match FIGURE_CAPTION_RE,
    and pick the one closest to (and above) the table's top edge. Captions
    are usually right on top of the table, but sometimes a few lines of prose
    sit between them — we allow up to CAPTION_LOOKUP_DY points of slack.
    """
    top = table_bbox[1]
    best: tuple[float, int, str] | None = None  # (distance, fig_num, caption)

    for block in page.get_text("blocks"):
        if len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[:5]
        if not isinstance(text, str):
            continue
        # Caption must be ABOVE the table
        if y1 > top + 2.0:
            continue
        distance = top - y1
        if distance > CAPTION_LOOKUP_DY:
            continue
        m = FIGURE_CAPTION_RE.search(text)
        if not m:
            continue
        fig_num = int(m.group(1))
        caption = m.group(2).strip()
        # Collapse internal whitespace
        caption = re.sub(r"\s+", " ", caption)
        if best is None or distance < best[0]:
            best = (distance, fig_num, caption)

    if best is None:
        return None
    return best[1], best[2]


# ---------------------------------------------------------------------------
# Nested table filtering


def _bbox_contains(outer, inner, slack: float = 2.0) -> bool:
    """Return True if `inner` bbox sits fully inside `outer` bbox (with slack)."""
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    return (
        ix0 >= ox0 - slack
        and iy0 >= oy0 - slack
        and ix1 <= ox1 + slack
        and iy1 <= oy1 + slack
    )


def _y_in_range(bbox, y_lo: float, y_hi: float, slack: float = 2.0) -> bool:
    """True if the vertical center of `bbox` falls within [y_lo, y_hi]."""
    cy = (bbox[1] + bbox[3]) / 2.0
    return y_lo - slack <= cy <= y_hi + slack


def _build_nesting_tree(tables: list) -> tuple[list, dict]:
    """
    Given all tables on a page, return:
      - top_level: list of tables that aren't contained in any other table
      - children: dict mapping id(table) -> list of its *direct* child tables

    A table is a direct child of another if it's contained in that table's
    bbox AND there is no third table between them in the containment chain.
    This builds a proper nesting tree (byte row -> bit-field table -> enum
    table) rather than a flat "contained in" relationship.
    """
    # Map each table to its immediate parent (or None if top-level).
    parent: dict[int, object] = {}
    for t in tables:
        # Find smallest enclosing table (= direct parent).
        best = None
        best_area = float("inf")
        for other in tables:
            if other is t:
                continue
            if not _bbox_contains(other.bbox, t.bbox):
                continue
            if other.bbox == t.bbox:
                continue
            area = (other.bbox[2] - other.bbox[0]) * (other.bbox[3] - other.bbox[1])
            if area < best_area:
                best = other
                best_area = area
        parent[id(t)] = best

    top_level = [t for t in tables if parent[id(t)] is None]
    children: dict[int, list] = {id(t): [] for t in tables}
    for t in tables:
        p = parent[id(t)]
        if p is not None:
            children[id(p)].append(t)
    return top_level, children


# ---------------------------------------------------------------------------
# Per-page extraction


def _format_nested_as_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """
    Render a nested sub-table as a compact markdown pipe table. Used at the
    first level of nesting (where the outer table is emitted as structured
    JSON rows/headers, not as markdown — so pipe escaping only has to happen
    once).

    At deeper nesting levels, use `_format_nested_as_inline` instead: markdown
    pipe tables inside markdown pipe table cells would require escaping that
    makes the output hard to read.
    """
    if not headers and not rows:
        return ""
    col_count = max(len(headers), max((len(r) for r in rows), default=0))
    if col_count == 0:
        return ""

    def pad(row: list[str]) -> list[str]:
        return [c.replace("|", "\\|").replace("\n", " ").strip() for c in row] + [
            ""
        ] * (col_count - len(row))

    lines = []
    if headers:
        lines.append("| " + " | ".join(pad(headers)) + " |")
        lines.append("|" + "---|" * col_count)
    for r in rows:
        lines.append("| " + " | ".join(pad(r)) + " |")
    return "\n".join(lines)


def _format_nested_as_inline(headers: list[str], rows: list[list[str]]) -> str:
    """
    Render a nested sub-table as inline text that avoids pipe characters, so
    it can be embedded inside a markdown pipe-table cell at a deeper nesting
    level without escape soup.

    Format:
        [Header1 / Header2: row1_col1 → row1_col2; row2_col1 → row2_col2; ...]

    Keeps the semantic structure legible for LLM consumers without colliding
    with surrounding markdown table syntax.
    """
    if not headers and not rows:
        return ""
    col_count = max(len(headers), max((len(r) for r in rows), default=0))
    if col_count == 0:
        return ""

    def clean(s: str) -> str:
        return s.replace("|", "/").replace("\n", " ").strip()

    header_label = " / ".join(clean(h) for h in headers) if headers else ""
    row_strs = []
    for r in rows:
        padded = list(r) + [""] * (col_count - len(r))
        row_strs.append(" → ".join(clean(c) for c in padded))
    body = "; ".join(row_strs)
    if header_label:
        return f"[{header_label}: {body}]"
    return f"[{body}]"


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.split("\n"))


def _text_in_rect(page: pymupdf.Page, x0: float, y0: float, x1: float, y1: float) -> str:
    """Extract plain text from a rectangular region of the page."""
    if y1 <= y0 or x1 <= x0:
        return ""
    clip = pymupdf.Rect(x0, y0, x1, y1)
    txt = page.get_text("text", clip=clip)
    txt = RUNNING_HEADER_RE.sub("", txt)
    return txt.strip()


def _extract_prose_around_children(
    page: pymupdf.Page,
    desc_cell_bbox: tuple[float, float, float, float],
    child_bboxes: list[tuple[float, float, float, float]],
) -> str:
    """
    Extract only the prose portion of a description cell that is NOT covered
    by any nested child table. Returns the prose text pieces joined with a
    placeholder marker <NESTED> where children should be spliced back in.

    Rationale: find_tables gives us the description cell's full text verbatim,
    including the text that lives inside nested sub-tables. If we keep that
    AND also append rendered children, we duplicate content. Instead, we
    walk y-bands of the cell and pull text only from the regions NOT covered
    by children.
    """
    x0, y0, x1, y1 = desc_cell_bbox
    if not child_bboxes:
        return _text_in_rect(page, x0, y0, x1, y1)

    # Sort children by y-top. Walk y-bands between children.
    kids = sorted(child_bboxes, key=lambda b: b[1])
    pieces: list[str] = []
    cursor_y = y0
    for cb in kids:
        c_y0, c_y1 = cb[1], cb[3]
        if c_y0 > cursor_y:
            txt = _text_in_rect(page, x0, cursor_y, x1, c_y0)
            if txt:
                pieces.append(txt)
        pieces.append("<NESTED>")
        cursor_y = max(cursor_y, c_y1)
    if cursor_y < y1:
        tail = _text_in_rect(page, x0, cursor_y, x1, y1)
        if tail:
            pieces.append(tail)

    return "\n".join(pieces)


def _render_table_recursive(
    table,
    children_map: dict,
    page: pymupdf.Page,
    depth: int = 0,
) -> tuple[list[str], list[list[str]]]:
    """
    Render `table` into (headers, rows), inlining any child tables into the
    appropriate parent-row description cell as formatted markdown. Recurses
    depth-first so deeply nested tables get assembled bottom-up.

    The parent row's description cell is rebuilt by extracting only the
    prose regions (y-bands not covered by any child) from the original page,
    then splicing each rendered child back in at its y position as markdown.
    This preserves structure without duplicating nested content.
    """
    raw_rows = table.extract()
    if not raw_rows:
        return [], []

    headers, rows = _build_header_and_rows(raw_rows)
    if not rows:
        return headers, rows

    my_children = children_map.get(id(table), [])
    if not my_children:
        return headers, rows

    # Render each child recursively first (bottom-up assembly).
    # Depth 0 = top-level table (returned as headers/rows).
    # Depth 1 = first-level nested, rendered as a markdown pipe table inside
    #           a top-level description cell — safe, only one level of pipes.
    # Depth >= 2 = deeper nest, rendered as inline bracket format to avoid
    #              pipe collisions with the outer markdown table that will
    #              wrap it.
    rendered: list[tuple[float, str, tuple]] = []  # (cy, md, bbox)
    for child in my_children:
        c_headers, c_rows = _render_table_recursive(
            child, children_map, page, depth + 1
        )
        if depth == 0:
            md = _format_nested_as_markdown(c_headers, c_rows)
        else:
            md = _format_nested_as_inline(c_headers, c_rows)
        if md:
            cy = (child.bbox[1] + child.bbox[3]) / 2.0
            rendered.append((cy, md, tuple(child.bbox)))

    if not rendered:
        return headers, rows

    # Figure out parent row y-ranges.
    visual_rows = list(table.rows)
    header_visual = max(0, len(visual_rows) - len(rows))
    data_visual = visual_rows[header_visual:]

    # Parent table's description column x-range: use the rightmost *non-None*
    # cell of the first data visual row. find_tables explodes each logical
    # column into 2-3 cells with trailing Nones for merged-cell continuation,
    # so cells[-1] is typically None.
    desc_x0: float | None = None
    desc_x1: float | None = None
    if data_visual:
        first_row = data_visual[0]
        if hasattr(first_row, "cells") and first_row.cells:
            for cell in reversed(first_row.cells):
                if cell is not None:
                    desc_x0 = cell[0]
                    desc_x1 = cell[2]
                    break
    if desc_x0 is None:
        desc_x0 = table.bbox[0]
        desc_x1 = table.bbox[2]

    # Bucket rendered children by parent data-row index.
    per_row: dict[int, list[tuple[float, str, tuple]]] = {}
    for cy, md, cb in rendered:
        for vi, vrow in enumerate(data_visual):
            y_lo = vrow.bbox[1]
            y_hi = vrow.bbox[3]
            if y_lo - 2.0 <= cy <= y_hi + 2.0:
                per_row.setdefault(vi, []).append((cy, md, cb))
                break

    # For each row with children, rebuild the description cell from prose
    # regions around each child bbox, then splice the rendered markdown in.
    for vi, kids in per_row.items():
        if vi >= len(rows):
            continue
        parent_row = rows[vi]
        if not parent_row:
            continue
        visual = data_visual[vi]
        row_y0 = visual.bbox[1]
        row_y1 = visual.bbox[3]
        desc_bbox = (desc_x0, row_y0, desc_x1, row_y1)
        kid_bboxes = [cb for _, _, cb in kids]
        prose = _extract_prose_around_children(page, desc_bbox, kid_bboxes)

        # Replace <NESTED> markers with rendered markdown (in y order).
        kids_sorted = sorted(kids, key=lambda k: k[0])
        result_parts: list[str] = []
        for part in prose.split("\n"):
            if part.strip() == "<NESTED>":
                if kids_sorted:
                    _, md, _ = kids_sorted.pop(0)
                    result_parts.append(md)
            else:
                if part:
                    result_parts.append(part)
        # Any remaining kids (shouldn't happen, but be safe) go at the end.
        for _, md, _ in kids_sorted:
            result_parts.append(md)

        parent_row[-1] = "\n".join(result_parts).strip()

    return headers, rows


def _extract_page_tables(
    page: pymupdf.Page, pdf_page_idx: int, page_offset: int
) -> list[dict]:
    """
    Extract all top-level tables on a single page. Nested tables are folded
    into their parent's Description cell as formatted markdown, preserving
    the structural hierarchy of bit-field and enum breakdowns.
    """
    tabs = page.find_tables()
    if not tabs.tables:
        return []

    top_level, children_map = _build_nesting_tree(tabs.tables)
    printed_page = pdf_page_idx - page_offset

    results: list[dict] = []
    for t in top_level:
        headers, rows = _render_table_recursive(t, children_map, page)
        if not headers and not rows:
            continue

        cap = _find_caption_above(page, t.bbox)
        raw_text = _extract_raw_text_in_bbox(page, t.bbox)

        results.append(
            {
                "figure_number": cap[0] if cap else None,
                "caption": cap[1] if cap else None,
                "printed_page": printed_page,
                "pdf_page": pdf_page_idx,
                "headers": headers,
                "rows": rows,
                "raw_text": raw_text,
                "_bbox": tuple(t.bbox),  # kept for continuation merging; removed before emit
            }
        )

    # Sort by y position top-to-bottom so "continuation" logic works across pages.
    results.sort(key=lambda r: r["_bbox"][1])
    return results


def _extract_raw_text_in_bbox(page: pymupdf.Page, bbox) -> str:
    """
    Return the page's text that falls inside the given bbox, with running
    headers stripped. Used as the raw_text field on each table entry.
    """
    clip = pymupdf.Rect(bbox)
    text = page.get_text("text", clip=clip)
    text = RUNNING_HEADER_RE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Continuation merging


def _merge_continuations(all_tables: list[dict]) -> list[dict]:
    """
    Merge all fragments that share a figure number into a single logical table.

    This handles both cases:
      (a) Multi-page figures where the caption repeats on each page
          (e.g., Figure 328 Identify Controller spanning ~10 pages).
      (b) Single figures split into multiple find_tables() detections on the
          same page due to visual breaks in the rendered table.

    Orphan tables (no figure_number) are attached to the most recent captioned
    figure seen in document order.

    Within each merged figure: if a fragment's first row equals the first
    fragment's header row, we drop it as a repeated header. Same for the
    fragment's "header" row itself — find_tables() re-detects the header on
    continuation pages, which would otherwise inject a duplicate.
    """
    by_figure: dict[int, dict] = {}
    order: list[int] = []
    last_fig: int | None = None
    orphans_before_first: list[dict] = []

    for t in all_tables:
        fig = t["figure_number"]

        if fig is None:
            # Orphan — attach to the most recently seen figure.
            if last_fig is None:
                orphans_before_first.append(t)
                continue
            fig = last_fig
            t = {**t, "figure_number": fig}

        if fig not in by_figure:
            by_figure[fig] = {
                "figure_number": fig,
                "caption": t["caption"],
                "printed_page": t["printed_page"],
                "pdf_page": t["pdf_page"],
                "headers": t["headers"],
                "rows": list(t["rows"]),
                "raw_text": t["raw_text"],
            }
            order.append(fig)
        else:
            parent = by_figure[fig]
            # Drop repeated header row on continuation fragments.
            cont_rows = t["rows"]
            if t["headers"] == parent["headers"]:
                pass  # headers row is already excluded from `rows`
            else:
                # Fragment's "header" row is actually data — keep it.
                cont_rows = [t["headers"]] + cont_rows
            # Also drop a repeated header if it reappears in row 0.
            if cont_rows and cont_rows[0] == parent["headers"]:
                cont_rows = cont_rows[1:]
            parent["rows"].extend(cont_rows)
            parent["raw_text"] += "\n" + t["raw_text"]

        last_fig = fig

    merged = [by_figure[f] for f in order]
    # Any orphans before the first captioned figure — emit as standalone so
    # we can audit them later. This should basically never happen.
    for o in orphans_before_first:
        merged.insert(0, {k: v for k, v in o.items() if k != "_bbox"})
    return merged


# ---------------------------------------------------------------------------
# Public API


def extract_tables(
    pdf_path: str,
    page_offset: int,
    first_content_page_idx: int = 0,
    last_content_page_idx: int | None = None,
) -> list[dict]:
    """
    Run the full table-extraction pass over a spec PDF.

    - `pdf_path`: path to the full NVMe spec PDF
    - `page_offset`: pdf_page_idx - printed_page_number (constant for a spec)
    - `first_content_page_idx`: skip front matter (cover, TOC)
    - `last_content_page_idx`: optional end index (inclusive); default = last page
    """
    doc = pymupdf.open(pdf_path)
    if last_content_page_idx is None:
        last_content_page_idx = doc.page_count - 1

    all_tables: list[dict] = []
    for i in range(first_content_page_idx, last_content_page_idx + 1):
        page = doc[i]
        page_tables = _extract_page_tables(page, i, page_offset)
        all_tables.extend(page_tables)

    merged = _merge_continuations(all_tables)

    # Strip internal-only fields before returning
    for t in merged:
        t.pop("_bbox", None)

    return merged


# ---------------------------------------------------------------------------
# CLI

if __name__ == "__main__":
    PDF_PATH        = "nvme_spec/NVMe_spec_full.pdf"
    PAGE_OFFSET     = 23  # PDF idx 24 is printed p.1
    FIRST_CONTENT   = 24  # skip cover + TOC
    OUTPUT_PATH     = "data/tables.json"

    tables = extract_tables(
        PDF_PATH,
        page_offset=PAGE_OFFSET,
        first_content_page_idx=FIRST_CONTENT,
    )

    print(f"extracted {len(tables)} tables")
    captioned = sum(1 for t in tables if t["figure_number"] is not None)
    print(f"  with caption: {captioned}")
    print(f"  orphan:       {len(tables) - captioned}")

    if tables:
        print()
        print("sample captions:")
        shown = 0
        for t in tables:
            if t["figure_number"] is None:
                continue
            print(f"  Figure {t['figure_number']:>4}: {t['caption'][:70]}  (p.{t['printed_page']}, {len(t['rows'])} rows)")
            shown += 1
            if shown >= 10:
                break

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {OUTPUT_PATH}")
