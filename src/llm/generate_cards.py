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

# LLM-summary gating.
#
# Originally MIN_PROSE_CHARS was 200 — anything shorter got a structural stub
# with no LLM call. That left 290/1036 cards empty (definition-style sections
# under §1.5/§1.6/§1.7 are typically a single short sentence) and a rerun
# couldn't recover them because cards_state.json marked them "processed" the
# first time around.
#
# New behaviour: any section with at least MIN_PROSE_CHARS of body text is
# summarized directly. Sections with shorter or zero prose are summarized from
# a synthetic "skeleton" built from the title, child-section titles, and the
# table captions that live under this section. That keeps every section
# represented in the vector index without re-scraping the PDF.
MIN_PROSE_CHARS = 50
MAX_PROSE_CHARS = 6000
MIN_SKELETON_SIGNAL = 1  # at least 1 child or 1 table to bother prompting


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


SKELETON_TEMPLATE = """Section {section_number} — {title}

This section has minimal or no body prose in the spec. Use the structural \
context below to write a short, accurate summary of what this section \
*defines or organizes* — do NOT invent technical detail. Anchor on the \
section title and the names of its children/tables.

{parent_block}
CHILD SUBSECTIONS:
{child_block}

{tables_block}

Return JSON. Keep the summary to 1-2 sentences max.
"""


def _build_skeleton_prompt(
    section_number: str,
    title: str,
    parent_title: str | None,
    child_pairs: list[tuple[str, str]],
    table_captions: list[str],
) -> str:
    parent_block = (
        f"PARENT SECTION: {parent_title}\n\n" if parent_title else ""
    )
    if child_pairs:
        child_block = "\n".join(
            f"  - {sn} {ct}" for sn, ct in child_pairs[:25]
        )
    else:
        child_block = "  (none)"
    if table_captions:
        tables_block = "TABLES IN THIS SECTION:\n" + "\n".join(
            f"  - {c}" for c in table_captions[:10]
        )
    else:
        tables_block = ""
    return SKELETON_TEMPLATE.format(
        section_number=section_number,
        title=title,
        parent_block=parent_block,
        child_block=child_block,
        tables_block=tables_block,
    )


# ---------------------------------------------------------------------------
# Helpers

def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


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


def _tables_by_section(
    tables: list[dict],
    prose_by_section: dict,
    relationships: list[dict],
) -> dict[str, list[int]]:
    """
    Assign each table to its parent section.

    Primary source: ``contained_in`` edges from relationships.json — these
    carry the structurally-correct figure→section mapping produced by the
    relationship extractor.

    Fallback (for any figure not covered by a ``contained_in`` edge): the
    old page-range heuristic that finds the deepest prose section whose
    pdf_page range contains the table's page.
    """
    by_section: dict[str, list[int]] = defaultdict(list)
    covered_figs: set[int] = set()

    # --- primary: relationships.json contained_in edges ---
    for r in relationships:
        if r.get("type") != "contained_in":
            continue
        src = r.get("source", "")
        tgt = r.get("target", "")
        if src.startswith("figure:") and tgt.startswith("section:"):
            try:
                fig = int(src.split(":", 1)[1])
            except (ValueError, IndexError):
                continue
            sec = tgt.split(":", 1)[1]
            by_section[sec].append(fig)
            covered_figs.add(fig)

    # --- fallback: page-range heuristic for uncovered figures ---
    ranges = []
    for sec_num, sec in prose_by_section.items():
        s = sec.get("start_pdf_page")
        e = sec.get("end_pdf_page")
        if s is not None and e is not None:
            ranges.append((s, e, sec_num))

    for t in tables:
        fig = t.get("figure_number")
        pdf_page = t.get("pdf_page")
        if fig is None or pdf_page is None or fig in covered_figs:
            continue
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


def _parse_llm_card(parsed) -> dict:
    if not isinstance(parsed, dict):
        return {"summary": "", "keywords": []}
    summary = parsed.get("summary", "") or ""
    keywords = parsed.get("keywords", []) or []
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k).strip() for k in keywords if str(k).strip()]
    return {"summary": summary.strip(), "keywords": keywords}


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
    return _parse_llm_card(parsed)


def _summarize_from_skeleton(
    section_number: str,
    title: str,
    parent_title: str | None,
    child_pairs: list[tuple[str, str]],
    table_captions: list[str],
) -> dict:
    """LLM call for sections with no usable prose. Uses title + children +
    table captions instead. Returns the same shape as ``_summarize``."""
    user = _build_skeleton_prompt(
        section_number, title, parent_title, child_pairs, table_captions
    )
    parsed, _ = generate_json(user, system=SYSTEM_PROMPT, max_output_tokens=384)
    return _parse_llm_card(parsed)


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
    tables_map = _tables_by_section(tables, prose_by_section, relationships)
    rels_map = _relationships_by_section(relationships)

    cards: list[dict] = _load_json(OUTPUT_PATH, []) if resume else []
    card_by_id = {c["section_id"]: c for c in cards}
    state: dict = _load_json(STATE_PATH, {"processed": []}) if resume else {"processed": []}
    processed = set(state["processed"])

    # Resume semantics: a section counts as "processed" only if its existing
    # card actually has a summary. Sections marked processed in a prior run
    # but with empty summary (the old behaviour: structural stub gated by
    # MIN_PROSE_CHARS) get re-attempted automatically.
    processed = {
        sn for sn in processed
        if card_by_id.get(sn, {}).get("summary")
    }

    # Lookups for skeleton-fallback context.
    title_by_id = {e["section_number"]: e["title"] for e in toc}
    table_caption_by_fig: dict[int, str] = {}
    for t in tables:
        fn = t.get("figure_number")
        cap = (t.get("caption") or t.get("title") or "").strip()
        if fn is not None and cap:
            table_caption_by_fig[fn] = cap

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

        # Preserve any prior LLM output we already have on this card so we
        # never blank it out by accident on a re-run.
        prior = card_by_id.get(sec_num)
        if prior and prior.get("summary"):
            base_card["summary"] = prior["summary"]
            base_card["keywords"] = prior.get("keywords", [])

        # Pick a summarization strategy.
        attempted_llm = False
        try:
            if len(prose_text) >= MIN_PROSE_CHARS:
                attempted_llm = True
                print(f"[{i}/{len(targets)}] {sec_num} (prose {len(prose_text)}c) {title[:55]}")
                llm_out = _summarize(sec_num, title, prose_text, table_figs)
                if llm_out["summary"]:
                    base_card["summary"] = llm_out["summary"]
                    base_card["keywords"] = llm_out["keywords"]
            else:
                child_ids = children_map.get(sec_num, [])
                child_pairs = [
                    (cid, title_by_id.get(cid, "")) for cid in child_ids
                ]
                table_caps = [
                    f"Figure {fn}: {table_caption_by_fig.get(fn, '')}".rstrip(": ")
                    for fn in table_figs
                ]
                signal = len(child_pairs) + len(table_caps)
                if signal >= MIN_SKELETON_SIGNAL or prose_text:
                    attempted_llm = True
                    parent_title = title_by_id.get(_parent_section(sec_num) or "")
                    print(f"[{i}/{len(targets)}] {sec_num} (skeleton "
                          f"children={len(child_pairs)} tables={len(table_caps)} "
                          f"prose={len(prose_text)}c) {title[:45]}")
                    llm_out = _summarize_from_skeleton(
                        sec_num, title, parent_title, child_pairs, table_caps,
                    )
                    if llm_out["summary"]:
                        base_card["summary"] = llm_out["summary"]
                        base_card["keywords"] = llm_out["keywords"]
        except (ImportError, ModuleNotFoundError):
            raise
        except Exception as e:  # noqa: BLE001
            print(f"  ! summary failed: {e}")

        # Upsert
        if sec_num in card_by_id:
            card_by_id[sec_num].update(base_card)
        else:
            cards.append(base_card)
            card_by_id[sec_num] = base_card

        # Only mark "processed" if we actually have a summary on the card.
        # That way a future re-run with a better prompt or higher signal
        # automatically retries any section we couldn't summarize.
        if base_card["summary"]:
            processed.add(sec_num)
        elif not attempted_llm:
            # Pure structural stub with nothing to attempt — don't mark
            # processed either (keeps the door open for later enrichment).
            pass
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
