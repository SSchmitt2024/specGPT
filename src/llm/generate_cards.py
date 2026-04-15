"""
Phase 1.6 — Metadata card generation.

One card per section, following the BUILD_PLAN schema:

  {
    "section_id": "8.2.3",
    "title": "Eye Opening Measurement Log Page",
    "spec_document": "NVM Express Base Specification",
    "spec_version": "<version>",
    "summary": "<LLM-generated 2-4 sentence summary>",
    "keywords": ["..."],
    "relationships": [...],      // filtered from relationships.json
    "parent_section": "8.2",
    "child_sections": ["8.2.3.1", ...],
    "tables": [<figure_number>, ...],
    "prose_blocks": [<paragraph indices>],
    "normative_count": <int>
  }

Most fields are assembled DETERMINISTICALLY from existing data files.
Only `summary` and `keywords` are LLM-generated — that's the expensive step
and it's why this script has retry/resume/rate-limit plumbing.

Sections with no prose get a stub card (structural placeholder) with
summary="" and no LLM call — saves budget.

Run:
  python -m src.llm.generate_cards             # all sections
  python -m src.llm.generate_cards --limit 10  # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

from .client import generate_json

# ---------------------------------------------------------------------------
# Paths

TOC_PATH = "data/toc.json"
PROSE_PATH = "data/prose.json"
TABLES_PATH = "data/tables.json"
RELATIONSHIPS_PATH = "data/relationships.json"
OUTPUT_PATH = "data/cards.json"
STATE_PATH = "data/cards_state.json"

# Spec identity (override via CLI if multiple specs get indexed).
DEFAULT_SPEC_DOCUMENT = "NVM Express Base Specification"
DEFAULT_SPEC_VERSION = "2.1"

# LLM-summary gating
MIN_PROSE_CHARS = 200
MAX_PROSE_CHARS = 6000


# ---------------------------------------------------------------------------
# Prompt

SYSTEM_PROMPT = """You summarize sections of the NVMe specification for engineers.

Your output is JSON with two fields:
  - "summary": 2-4 sentences. What this section defines, who uses it, and the \
key mechanism or value. Concrete and specific — name commands, structures, \
fields, or registers where relevant. No marketing language, no filler.
  - "keywords": 4-10 short technical terms that would help a search engine \
surface this section. Prefer NVMe domain terms (command names, acronyms, \
register names). No generic words like "the", "controller" alone, or \
"specification".

Return JSON only. Example:
{"summary": "Defines the Set Features command (FID 0Dh) used to configure the Host Memory Buffer (HMB). Hosts supply a descriptor list in CDW12-CDW15 before setting the enable bit in CDW11. The controller uses HMB allocations to cache metadata between commands.", "keywords": ["Set Features", "Host Memory Buffer", "HMB", "FID 0Dh", "CDW11", "descriptor list"]}
"""

USER_TEMPLATE = """Section {section_number} — {title}

PROSE:
{prose}

{tables_block}

Summarize for engineers. Return JSON.
"""


# ---------------------------------------------------------------------------
# Helpers

def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _flatten_prose(section: dict) -> tuple[str, list[int]]:
    parts: list[str] = []
    idxs: list[int] = []
    for i, pg in enumerate(section.get("paragraphs", [])):
        if isinstance(pg, dict):
            t = pg.get("text", "")
        else:
            t = str(pg)
        if t:
            parts.append(t)
            idxs.append(i)
    return "\n\n".join(parts), idxs


def _parent_section(section_number: str) -> str | None:
    parts = section_number.split(".")
    if len(parts) < 2:
        return None
    return ".".join(parts[:-1])


def _build_child_map(toc: list[dict]) -> dict[str, list[str]]:
    """Map each section -> list of direct child section numbers."""
    known = {e["section_number"] for e in toc}
    children: dict[str, list[str]] = defaultdict(list)
    for entry in toc:
        parent = _parent_section(entry["section_number"])
        if parent and parent in known:
            children[parent].append(entry["section_number"])
    return children


def _tables_by_section(tables: list[dict], prose_by_section: dict) -> dict[str, list[int]]:
    """
    Assign each table to the section whose prose page range contains it.
    Falls back to printed_page matching when possible.
    """
    # Build (start_pdf_page, end_pdf_page, section_number) triples.
    ranges = []
    for sec_num, sec in prose_by_section.items():
        s = sec.get("start_pdf_page")
        e = sec.get("end_pdf_page")
        if s is not None and e is not None:
            ranges.append((s, e, sec_num))

    by_section: dict[str, list[int]] = defaultdict(list)
    for t in tables:
        fig = t.get("figure_number")
        pdf_page = t.get("pdf_page")
        if fig is None or pdf_page is None:
            continue
        # Find the deepest (narrowest) range that contains pdf_page.
        best = None
        best_span = None
        for s, e, sec_num in ranges:
            if s <= pdf_page <= e:
                span = e - s
                if best is None or span < best_span:
                    best = sec_num
                    best_span = span
        if best:
            by_section[best].append(fig)
    return by_section


def _relationships_by_section(rels: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rels:
        src = r.get("source", "")
        tgt = r.get("target", "")
        # Include the edge under any section that appears as source or target.
        for endpoint in (src, tgt):
            if endpoint.startswith("section:"):
                sec_num = endpoint.split(":", 1)[1]
                out[sec_num].append(r)
    return out


def _summarize(section_number: str, title: str, prose: str, table_figs: list[int]) -> dict:
    """LLM call — returns {'summary': str, 'keywords': [str,...]}."""
    if len(prose) > MAX_PROSE_CHARS:
        prose = prose[:MAX_PROSE_CHARS] + " […truncated]"

    tables_block = ""
    if table_figs:
        tables_block = f"TABLES IN THIS SECTION: Figure {', Figure '.join(str(f) for f in table_figs[:10])}"

    user = USER_TEMPLATE.format(
        section_number=section_number,
        title=title,
        prose=prose,
        tables_block=tables_block,
    )

    parsed, _ = generate_json(user, system=SYSTEM_PROMPT, max_output_tokens=512)
    if not isinstance(parsed, dict):
        return {"summary": "", "keywords": []}
    summary = parsed.get("summary", "") or ""
    keywords = parsed.get("keywords", []) or []
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k).strip() for k in keywords if str(k).strip()]
    return {"summary": summary.strip(), "keywords": keywords}


# ---------------------------------------------------------------------------
# Main

def run(
    limit: int | None = None,
    resume: bool = True,
    spec_document: str = DEFAULT_SPEC_DOCUMENT,
    spec_version: str = DEFAULT_SPEC_VERSION,
) -> None:
    toc = _load_json(TOC_PATH, [])
    prose = _load_json(PROSE_PATH, [])
    tables = _load_json(TABLES_PATH, [])
    relationships = _load_json(RELATIONSHIPS_PATH, [])

    if not toc or not prose:
        print("ERROR: toc.json or prose.json missing.", file=sys.stderr)
        sys.exit(1)

    prose_by_section = {p["section_number"]: p for p in prose}
    children_map = _build_child_map(toc)
    tables_map = _tables_by_section(tables, prose_by_section)
    rels_map = _relationships_by_section(relationships)

    cards: list[dict] = _load_json(OUTPUT_PATH, []) if resume else []
    card_by_id = {c["section_id"]: c for c in cards}
    state: dict = _load_json(STATE_PATH, {"processed": []}) if resume else {"processed": []}
    processed = set(state["processed"])

    # Process every TOC entry (keeps ordering stable).
    targets = toc[:]
    if limit:
        # Prefer unprocessed entries first when limiting.
        unprocessed = [e for e in targets if e["section_number"] not in processed]
        targets = unprocessed[:limit]

    print(f"cards already built: {len(processed)}")
    print(f"to process this run: {len(targets)}")

    for i, entry in enumerate(targets, start=1):
        sec_num = entry["section_number"]
        if sec_num in processed and limit is None:
            continue

        title = entry["title"]
        section_prose = prose_by_section.get(sec_num, {})
        prose_text, prose_idxs = _flatten_prose(section_prose)

        table_figs = sorted(tables_map.get(sec_num, []))
        section_rels = rels_map.get(sec_num, [])
        normative = section_prose.get("normative") or []

        base_card = {
            "section_id": sec_num,
            "title": title,
            "spec_document": spec_document,
            "spec_version": spec_version,
            "summary": "",
            "keywords": [],
            "parent_section": _parent_section(sec_num),
            "child_sections": children_map.get(sec_num, []),
            "tables": table_figs,
            "prose_blocks": prose_idxs,
            "relationships": section_rels,
            "normative_count": len(normative),
            "level": entry.get("level"),
        }

        if len(prose_text) >= MIN_PROSE_CHARS:
            try:
                print(f"[{i}/{len(targets)}] {sec_num} {title[:60]}")
                llm_out = _summarize(sec_num, title, prose_text, table_figs)
                base_card["summary"] = llm_out["summary"]
                base_card["keywords"] = llm_out["keywords"]
            except Exception as e:  # noqa: BLE001
                print(f"  ! summary failed: {e}")
        else:
            # Structural stub — no LLM call.
            pass

        # Upsert
        if sec_num in card_by_id:
            # Merge in place
            card_by_id[sec_num].update(base_card)
        else:
            cards.append(base_card)
            card_by_id[sec_num] = base_card

        processed.add(sec_num)
        state["processed"] = sorted(processed)

        _save_json(OUTPUT_PATH, cards)
        _save_json(STATE_PATH, state)

    stub_count = sum(1 for c in cards if not c["summary"])
    print(f"\ntotal cards: {len(cards)}  (stubs without summary: {stub_count})")
    print(f"wrote {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# CLI

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Process at most N sections.")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--spec-document", default=DEFAULT_SPEC_DOCUMENT)
    ap.add_argument("--spec-version", default=DEFAULT_SPEC_VERSION)
    args = ap.parse_args()

    run(
        limit=args.limit,
        resume=not args.no_resume,
        spec_document=args.spec_document,
        spec_version=args.spec_version,
    )
