"""Parse the UNH-IOL NVM Command Set Conformance test plan PDF into
per-case rows and (optionally) upsert them into the Supabase `test_plans`
table.

One row per selectable unit: a Case (level-3 TOC entry) or a Sub-Case
(level-4 TOC entry). Test-level metadata (purpose, setup, discussion...)
is denormalized onto every row of that test so injection needs a single
row fetch.

Reference-style cases ("Follow steps as described in test case 1.26.3",
"follow the test procedure of test case 3 except for the modified steps
below") are materialized at ingest: base steps are copied and patched.
`raw_text` always keeps the verbatim PDF segment as a fallback.

Usage:
    python scripts/ingest_iol_testplans.py            # parse -> JSON artifact
    python scripts/ingest_iol_testplans.py --load     # ... and upsert to Supabase
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
PDF_DEFAULT = ROOT / "nvme_spec" / "UNH-IOL_NVM_Command_Set_Conformance_v25.0_2026.02.03.pdf"
OUT_DEFAULT = ROOT / "data" / "iol_testplans" / "testplans.json"

# Header/footer lines repeated on every page.
_BOILERPLATE = (
    re.compile(r"University of New Hampshire InterOperability Laboratory"),
    re.compile(r"UNH.?IOL NVMe Testing Service"),
    re.compile(r"^NVM Command Set Conformance Test Suite\s*$"),
    re.compile(r"©\s*20\d\d\s*UNH.?IOL"),
    re.compile(r"^\d{1,3}\s*$"),  # bare page number
)

_TEST_FIELDS = [
    "Purpose", "References", "Resource Requirements",
    "Last Modification", "Discussion", "Test Setup",
    # Some tests (e.g. 1.8) define one procedure at test level, with the
    # level-3 entries being one-line variants that inherit it.
    "Test Procedure", "Observable Results", "Observable Result",
    "Possible Problems", "Possible Problem",
]
_CASE_FIELDS = ["Test Procedure", "Observable Results", "Observable Result",
                "Possible Problems", "Possible Problem", "Sub-Cases", "Notes"]


def _clean_page(page: fitz.Page) -> str:
    lines = []
    for line in page.get_text().splitlines():
        if any(p.search(line.strip()) for p in _BOILERPLATE):
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _title_regex(title: str) -> str:
    """Whitespace-tolerant regex for a TOC title as it appears in body text."""
    tokens = [re.escape(t) for t in _norm(title).split(" ") if t]
    return r"\s+".join(tokens)


def _split_fields(segment: str, labels: list[str]) -> dict[str, str]:
    """Split a text segment on `Label:` markers. Returns {label: body}."""
    # Label followed by a colon, or a bare label alone on its line
    # (one case, 13.1/13, omits the colon after "Test Procedure").
    # `[ \t.]*` start: one label appears as ".Observable Results:" (PDF artifact).
    pat = re.compile(
        r"^[ \t.]*(" + "|".join(re.escape(l) for l in labels) + r")\s*(?::\s*|[ \t]*$)",
        re.MULTILINE,
    )
    out: dict[str, str] = {}
    matches = list(pat.finditer(segment))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(segment)
        label = m.group(1)
        # Normalize singular/plural label variants
        if label.startswith("Observable"):
            label = "Observable Results"
        elif label.startswith("Possible"):
            label = "Possible Problems"
        body = segment[m.end():end].strip()
        if label in out:  # rare duplicated label; keep both
            out[label] += "\n" + body
        else:
            out[label] = body
    if matches:
        out["_preamble"] = segment[: matches[0].start()].strip()
    else:
        out["_preamble"] = segment.strip()
    return out


def _parse_numbered(body: str) -> list[dict]:
    """Parse '1. text' numbered items (multiline continuation)."""
    items: list[dict] = []
    cur_n: int | None = None
    cur: list[str] = []
    for line in body.splitlines():
        # Number may share the line with text or sit alone ("1. \nConfigure...")
        m = re.match(r"^\s*(\d{1,2})\.(?:\s+(\S.*))?\s*$", line)
        # First item may start above 1: the PDF numbers steps continuously
        # across some cases (e.g. case 7 starting at step 32).
        ok = m and (cur_n is None or int(m.group(1)) == cur_n + 1)
        if ok:
            if cur_n is not None:
                items.append({"n": cur_n, "text": _norm(" ".join(cur))})
            cur_n, cur = int(m.group(1)), [m.group(2) or ""]
        elif cur_n is not None:
            cur.append(line.strip())
        # text before the first number is ignored here (kept in raw_text)
    if cur_n is not None:
        items.append({"n": cur_n, "text": _norm(" ".join(cur))})
    items = [it for it in items if it["text"]]
    if not items and body.strip():
        # unnumbered prose body (e.g. a single Verify sentence)
        items = [{"n": 1, "text": _norm(body)}]
    return items


def parse_pdf(pdf_path: Path) -> list[dict]:
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()

    # Restrict to Group 1..13 (skip front matter and appendices).
    entries = []
    in_groups = False
    for lvl, title, page in toc:
        t = _norm(title)
        if lvl == 1:
            in_groups = bool(re.match(r"Group \d+:", t))
        if in_groups and lvl >= 1:
            entries.append({"level": lvl, "title": t, "page": page})
    last_page = entries[-1]["page"] + 8  # slack past the final entry

    # Cleaned full text + page start offsets (pages are 1-indexed in TOC).
    page_off: dict[int, int] = {}
    buf: list[str] = []
    pos = 0
    first_page = entries[0]["page"]
    for p in range(first_page - 1, min(last_page, len(doc))):
        page_off[p + 1] = pos
        t = _clean_page(doc[p])
        buf.append(t)
        pos += len(t)
    text = "".join(buf)

    # Locate each heading with a forward-moving cursor anchored to its page.
    cursor = 0
    for e in entries:
        anchor = page_off.get(e["page"], cursor)
        start = max(cursor, anchor - 200)
        if e["level"] == 1:
            pat = re.compile(_title_regex(e["title"]))
        elif e["level"] == 2:
            m_id = re.match(r"Test (\d+\.\d+)", e["title"])
            e["test_id"] = m_id.group(1) if m_id else None
            pat = re.compile(r"Test\s+" + re.escape(e["test_id"] or "") + r"\s*[–-]")
        elif e["level"] == 3:
            m_id = re.match(r"Case (\d+):", e["title"])
            e["case_num"] = m_id.group(1) if m_id else None
            rest = _norm(e["title"].split(":", 1)[1])[:40] if ":" in e["title"] else ""
            pat = re.compile(
                r"Case\s+" + re.escape(e["case_num"] or "") + r"\s*:\s*" +
                _title_regex(rest)[: 200]
            ) if e["case_num"] else re.compile(_title_regex(e["title"][:50]))
        else:  # level 4 sub-case: "N. Title"
            m_id = re.match(r"(\d+)\.\s*(.*)", e["title"])
            e["sub_num"] = m_id.group(1) if m_id else None
            rest = _norm(m_id.group(2))[:40] if m_id else e["title"][:40]
            pat = re.compile(
                (re.escape(e["sub_num"]) + r"\.\s+" if e.get("sub_num") else "") +
                _title_regex(rest)
            )
        m = pat.search(text, start)
        if not m and start > anchor - 200:
            m = pat.search(text, max(0, anchor - 200))  # tolerate TOC page drift
        e["start"] = m.start() if m else None
        if m:
            cursor = m.end()

    located = [e for e in entries if e["start"] is not None]
    missed = [e for e in entries if e["start"] is None]

    # Segment and build rows.
    rows: list[dict] = []
    all_tests: list[dict] = []
    group_name = ""
    test_meta: dict = {}
    test_id = ""
    parent_case: dict | None = None
    for i, e in enumerate(located):
        end = located[i + 1]["start"] if i + 1 < len(located) else len(text)
        seg = text[e["start"]:end]
        if e["level"] == 1:
            group_name = e["title"]
            continue
        if e["level"] == 2:
            test_id = e.get("test_id") or e["title"]
            fields = _split_fields(seg, _TEST_FIELDS)
            test_meta = {
                "test_id": test_id,
                "test_title": _norm(e["title"]),
                "group_name": group_name,
                "purpose": fields.get("Purpose"),
                "references_text": fields.get("References"),
                "requirements": fields.get("Resource Requirements"),
                "last_modification": fields.get("Last Modification"),
                "discussion": fields.get("Discussion"),
                "setup": fields.get("Test Setup"),
            }
            # Test-level procedure, inherited by one-line variant cases.
            test_meta["_test_steps"] = _parse_numbered(fields.get("Test Procedure", ""))
            test_meta["_test_observables"] = _parse_numbered(fields.get("Observable Results", ""))
            test_meta["_pp"] = fields.get("Possible Problems")
            test_meta["_raw"] = seg.strip()[:20000]
            test_meta["_page"] = e["page"]
            all_tests.append(test_meta)
            parent_case = None
            continue

        fields = _split_fields(seg, _CASE_FIELDS)
        row = dict(test_meta)
        row.update({
            "title": _norm(e["title"]),
            "pdf_page": e["page"],
            "steps": _parse_numbered(fields.get("Test Procedure", "")),
            "observables": _parse_numbered(fields.get("Observable Results", "")),
            "possible_problems": fields.get("Possible Problems"),
            "raw_text": seg.strip()[:20000],
            "materialized_from": None,
        })
        if e["level"] == 3:
            # "Case N: Title" or bare "N. Title" variant entries (e.g. Test 1.8)
            mnum = re.match(r"(\d+)\.\s", e["title"])
            row["case_num"] = e.get("case_num") or (
                mnum.group(1) if mnum else _norm(e["title"])[:20])
            row["subcase_num"] = None
            row["id"] = f"{test_id}/{row['case_num']}"
            # Preamble of a Sub-Cases block applies to level-4 children.
            row["_subcases_preamble"] = fields.get("Sub-Cases", "")
            parent_case = row
        else:
            base = parent_case or {}
            row["case_num"] = base.get("case_num")
            row["subcase_num"] = e.get("sub_num")
            row["id"] = f"{test_id}/{row['case_num']}/{row['subcase_num']}"
            row["_parent_id"] = base.get("id")
        rows.append(row)

    # 26 tests (Group 4/5/6 register tests mostly) have no cases at all:
    # the test itself is the selectable unit.
    tests_with_rows = {r["test_id"] for r in rows}
    for tm in all_tests:
        if tm["test_id"] in tests_with_rows:
            continue
        row = dict(tm)
        row.update({
            "id": tm["test_id"], "case_num": None, "subcase_num": None,
            "title": tm["test_title"], "pdf_page": tm["_page"],
            "steps": tm["_test_steps"], "observables": tm["_test_observables"],
            "possible_problems": tm["_pp"], "raw_text": tm["_raw"],
            "materialized_from": None,
        })
        rows.append(row)

    # The PDF has a few duplicate numbers (two "Case 1" in Test 1.10, two
    # "20." items in 1.3/1): suffix later duplicates so ids stay unique.
    seen: dict[str, int] = {}
    for r in rows:
        n = seen.get(r["id"], 0)
        seen[r["id"]] = n + 1
        if n:
            r["id"] = f"{r['id']}{chr(ord('a') + n)}"

    _materialize(rows)
    for r in rows:
        r.pop("_subcases_preamble", None)
        r.pop("_parent_id", None)
        for k in ("_test_steps", "_test_observables", "_pp", "_raw", "_page"):
            r.pop(k, None)
    if missed:
        print(f"WARNING: {len(missed)} TOC headings not located:", file=sys.stderr)
        for e in missed[:20]:
            print(f"  L{e['level']} p{e['page']} {e['title'][:80]}", file=sys.stderr)
    return rows


_REF_CASE = re.compile(
    r"(?:follow (?:the )?(?:steps|test procedure)(?: as described in| of)?|"
    r"as described in|same test procedure for) (?:test )?case (\d+(?:\.\d+){0,2})",
    re.IGNORECASE)
_REF_PROC_OF = re.compile(
    r"follow the test procedure of test case (\d+)", re.IGNORECASE)
_REPLACE_STEP = re.compile(r"Replace steps? (\d+)(?:\s*(?:-|through|and)\s*(\d+))?"
                           r" (?:from|of) the test case", re.IGNORECASE)


def _resolve_ref(ref: str, test_id: str, by_id: dict) -> dict | None:
    """'1.26.3' -> case 3 of test 1.26; bare '3' -> case 3 of current test."""
    parts = ref.split(".")
    if len(parts) >= 3:
        cand = f"{parts[0]}.{parts[1]}/{parts[2]}"
    else:
        cand = f"{test_id}/{ref}"
    return by_id.get(cand)


def _materialize(rows: list[dict]) -> None:
    by_id = {r["id"]: r for r in rows}
    for r in rows:
        if r["steps"]:
            # Case body may still start with "Follow steps as described in
            # test case X" as its single step: expand that too.
            if len(r["steps"]) <= 2:
                m = _REF_CASE.search(r["steps"][0]["text"])
                if m:
                    base = _resolve_ref(m.group(1), r["test_id"], by_id)
                    if base and base["steps"]:
                        extra = r["steps"][1:]
                        r["steps"] = [dict(s) for s in base["steps"]] + extra
                        r["materialized_from"] = base["id"]
                        if not r["observables"]:
                            r["observables"] = [dict(o) for o in base["observables"]]
            continue
        # No steps of its own: try parent Sub-Cases preamble, own body,
        # then fall back to inheriting the parent case / test-level procedure.
        base = None
        parent = by_id.get(r.get("_parent_id") or "")
        preamble = ((parent or {}).get("_subcases_preamble") or "") + " " + \
            ((parent or {}).get("raw_text") or "")[:800]
        m = _REF_CASE.search(r["raw_text"][:600]) or _REF_PROC_OF.search(preamble) \
            or _REF_CASE.search(preamble)
        if m:
            base = _resolve_ref(m.group(1), r["test_id"], by_id)
        if not (base and base["steps"]) and parent and parent["steps"]:
            base = parent  # variant sub-case: same procedure as its case
        if not (base and base["steps"]) and r.get("_test_steps"):
            # one-line variant case inheriting the test-level procedure
            r["steps"] = [dict(s) for s in r["_test_steps"]]
            r["observables"] = r["observables"] or [dict(o) for o in r["_test_observables"]]
            r["materialized_from"] = r["test_id"]
            continue
        if not (base and base["steps"]):
            continue
        steps = [dict(s) for s in base["steps"]]
        # Apply "Replace step N ..." modifications from this sub-case's body.
        # Drop the first line (the sub-case's own heading, e.g. "2. Sanitize
        # Operation") or it would be parsed as a numbered item and swallow
        # the replacement steps that follow it.
        body = r["raw_text"].split("\n", 1)[1] if "\n" in r["raw_text"] else ""
        body_items = _parse_numbered(body)
        rm = _REPLACE_STEP.search(r["raw_text"])
        if rm:
            lo = int(rm.group(1))
            hi = int(rm.group(2) or rm.group(1))
            repl = [it for it in body_items
                    if not _REPLACE_STEP.search(it["text"])
                    and not _REF_CASE.search(it["text"])]
            steps = ([s for s in steps if s["n"] < lo]
                     + repl
                     + [s for s in steps if s["n"] > hi])
            for k, s in enumerate(steps):
                s["n"] = k + 1
        r["steps"] = steps
        r["observables"] = r["observables"] or [dict(o) for o in base["observables"]]
        r["materialized_from"] = base["id"]


def load_supabase(rows: list[dict]) -> None:
    sys.path.insert(0, str(ROOT))
    from src.pipeline.search import supabase_client
    sb = supabase_client()
    cols = ["id", "test_id", "case_num", "subcase_num", "title", "test_title",
            "group_name", "purpose", "references_text", "requirements",
            "last_modification", "discussion", "setup", "steps", "observables",
            "possible_problems", "raw_text", "materialized_from", "pdf_page"]
    payload = [{c: r.get(c) for c in cols} for r in rows]
    for i in range(0, len(payload), 200):
        sb.table("test_plans").upsert(payload[i:i + 200]).execute()
        print(f"upserted {min(i + 200, len(payload))}/{len(payload)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=PDF_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--load", action="store_true", help="upsert to Supabase")
    args = ap.parse_args()

    rows = parse_pdf(args.pdf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))
    n_mat = sum(1 for r in rows if r["materialized_from"])
    n_steps = sum(1 for r in rows if r["steps"])
    print(f"{len(rows)} rows ({n_steps} with steps, {n_mat} materialized) -> {args.out}")
    if args.load:
        load_supabase(rows)


if __name__ == "__main__":
    main()
