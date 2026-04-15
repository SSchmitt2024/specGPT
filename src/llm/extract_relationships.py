"""
Phase 1.5 — LLM-assisted relationship extraction.

Sends each section's prose + a curated entity list to Gemini and asks it
to return implicit relationships the regex pass missed (e.g. "the Set
Features command uses the Host Memory Buffer feature", "log page X is
returned by the Get Log Page admin command").

Outputs:
  data/relationships_llm.json  — new edges tagged confidence="llm"

Merged with the deterministic edges downstream by the graph build step.
Resumable: on restart, already-processed sections are skipped.

Run:
  python -m src.llm.extract_relationships            # full spec
  python -m src.llm.extract_relationships --limit 5  # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .client import generate_json

# ---------------------------------------------------------------------------
# Paths

PROSE_PATH = "data/prose.json"
TOC_PATH = "data/toc.json"
FIELDS_PATH = "data/fields.json"
OUTPUT_PATH = "data/relationships_llm.json"
STATE_PATH = "data/relationships_llm_state.json"

# Only call the LLM for sections with at least this many prose characters.
# Below this, there's nothing for the LLM to extract relationships from.
MIN_PROSE_CHARS = 400

# Cap prose size per call to keep input tokens bounded.
MAX_PROSE_CHARS = 6000


# ---------------------------------------------------------------------------
# Prompt

SYSTEM_PROMPT = """You are an NVMe specification analyst. Your job is to read a section of the \
NVMe spec and extract RELATIONSHIPS between named entities that a regex-based \
cross-reference pass would miss.

Entity types you care about:
  - command     (e.g. "Set Features", "Get Log Page", "Identify")
  - feature     (e.g. "Host Memory Buffer", "Arbitration")
  - log_page    (e.g. "SMART / Health Information", "Eye Opening Measurement")
  - structure   (e.g. "Submission Queue Entry", "Identify Controller data structure")
  - field       (e.g. "CDW10", "FID", "NSID")
  - section     (e.g. "5.1.2")
  - figure      (e.g. "Figure 312")

Relationship types you emit:
  - uses                (command -> feature / structure)
  - returned_by         (log_page / structure -> command)
  - posts_to            (command -> queue type)
  - requires            (command / feature -> command / feature)
  - defined_in          (entity -> section)
  - configured_by       (feature -> command)
  - superseded_by       (entity -> entity)
  - related_to          (fallback when a weaker association is implied)

RULES:
  1. Only extract relationships that the prose IMPLIES or STATES.
   Do not invent facts. If unsure, omit.
  2. Skip relationships that are ALREADY explicit cross-references
   (e.g. "see Section X.Y") — those are captured elsewhere.
  3. Every relationship must cite evidence: a short verbatim quote from
   the prose (≤ 25 words) proving the link.
  4. Entity names must match the prose spelling.
  5. Return JSON ONLY, no preamble.

Output schema:
{
  "relationships": [
    {
      "source": "<entity name>",
      "source_type": "<entity type>",
      "target": "<entity name>",
      "target_type": "<entity type>",
      "relation": "<one of the relationship types>",
      "evidence": "<short verbatim quote>"
    }
  ]
}

If nothing new is implied, return {"relationships": []}.
"""


USER_TEMPLATE = """Section: {section_number} — {title}

PROSE:
{prose}

Return implicit relationships as JSON.
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


def _flatten_prose(section: dict) -> str:
    parts: list[str] = []
    for pg in section.get("paragraphs", []):
        if isinstance(pg, dict):
            t = pg.get("text", "")
        else:
            t = str(pg)
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def _normalize_edge(edge: dict, section_number: str) -> dict | None:
    """Validate + normalize an LLM edge to our internal edge schema."""
    required = ("source", "target", "relation", "evidence")
    if not all(k in edge and edge[k] for k in required):
        return None

    src_type = edge.get("source_type", "entity")
    tgt_type = edge.get("target_type", "entity")

    return {
        "source": f"{src_type}:{edge['source']}",
        "target": f"{tgt_type}:{edge['target']}",
        "type": edge["relation"],
        "evidence": f"[{section_number}] {edge['evidence']}",
        "confidence": "llm",
        "origin_section": section_number,
    }


# ---------------------------------------------------------------------------
# Main

def run(limit: int | None = None, resume: bool = True) -> None:
    prose_data = _load_json(PROSE_PATH, [])
    if not prose_data:
        print(f"ERROR: {PROSE_PATH} is empty or missing.", file=sys.stderr)
        sys.exit(1)

    all_edges: list[dict] = _load_json(OUTPUT_PATH, []) if resume else []
    state: dict = _load_json(STATE_PATH, {"processed": []}) if resume else {"processed": []}
    processed = set(state["processed"])

    # Pick candidates: sections with enough prose to be worth the call.
    candidates = []
    for sec in prose_data:
        if sec["section_number"] in processed:
            continue
        prose = _flatten_prose(sec)
        if len(prose) < MIN_PROSE_CHARS:
            continue
        candidates.append((sec, prose))

    if limit:
        candidates = candidates[:limit]

    print(f"sections to process: {len(candidates)}")
    print(f"already done: {len(processed)}")

    for i, (sec, prose) in enumerate(candidates, start=1):
        section_number = sec["section_number"]
        title = sec["title"]
        if len(prose) > MAX_PROSE_CHARS:
            prose = prose[:MAX_PROSE_CHARS] + " […truncated]"

        user = USER_TEMPLATE.format(
            section_number=section_number,
            title=title,
            prose=prose,
        )

        print(f"[{i}/{len(candidates)}] {section_number} {title[:60]}")
        try:
            parsed, result = generate_json(user, system=SYSTEM_PROMPT, max_output_tokens=4096)
        except Exception as e:  # noqa: BLE001
            print(f"  ! failed: {e}")
            continue

        rels = parsed.get("relationships", []) if isinstance(parsed, dict) else []
        new_edges = []
        for raw in rels:
            if not isinstance(raw, dict):
                continue
            edge = _normalize_edge(raw, section_number)
            if edge:
                new_edges.append(edge)

        all_edges.extend(new_edges)
        processed.add(section_number)
        state["processed"] = sorted(processed)

        # Persist after every call so a crash doesn't lose progress.
        _save_json(OUTPUT_PATH, all_edges)
        _save_json(STATE_PATH, state)

        tokens_in = result.prompt_tokens or 0
        tokens_out = result.output_tokens or 0
        print(f"  -> {len(new_edges)} new edges  (in={tokens_in} out={tokens_out})")

    print(f"\ntotal LLM edges in {OUTPUT_PATH}: {len(all_edges)}")


# ---------------------------------------------------------------------------
# CLI

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Process at most N sections.")
    ap.add_argument("--no-resume", action="store_true", help="Start from scratch (ignore state).")
    args = ap.parse_args()

    run(limit=args.limit, resume=not args.no_resume)
