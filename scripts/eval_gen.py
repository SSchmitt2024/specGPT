"""
Phase 2 — Step 2.2: Eval Set Generator

Generates a curated QA eval set from Phase 1 data. Each item has a query,
expected question type, expected source sections, and a reference answer.

Strategy by type:
  lookup      — constructed from fields.json, no LLM (deterministic)
  structural  — LLM-generated from cards with structural summaries
  relational  — LLM-generated from cards referencing multiple concepts
  procedural  — LLM-generated from cards describing sequences/procedures

LLM pacing: Gemini free tier is ~9 RPM. 35 LLM items ≈ 4 minutes.

Output: data/eval_set.json

Run:
  python -m src.pipeline.eval_gen                   # ~60 items
  python -m src.pipeline.eval_gen --count 20
  python -m src.pipeline.eval_gen --types lookup structural
  python -m src.pipeline.eval_gen --dry-run         # plan only, no LLM calls
  python -m src.pipeline.eval_gen --output path/to/out.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.client import generate_json


DATA_DIR = Path("data")
CARDS_PATH = DATA_DIR / "cards.json"
FIELDS_PATH = DATA_DIR / "fields.json"
FIELD_INDEX_PATH = DATA_DIR / "field_index.json"
DEFAULT_OUTPUT = DATA_DIR / "eval_set.json"

# Fraction of total items per type (must sum to 1.0)
TYPE_FRACTIONS = {
    "lookup":     0.40,
    "structural": 0.25,
    "relational": 0.20,
    "procedural": 0.15,
}

_LLM_SYSTEM = (
    "You are generating NVMe specification Q&A eval pairs for testing a RAG retrieval system. "
    "Given a spec section summary, produce one question and a concise reference answer. "
    "Respond with valid JSON only."
)

_LLM_PROMPT = """\
Section: {section_id} — {title}
Summary: {summary}

Generate one {qtype} question that an NVMe implementer would ask about this section,
plus a 2-4 sentence reference answer derived from the summary above.

Question style guide:
  structural  → ask about the organization, fields, or layout of a structure/command
  relational  → ask how this section's feature interacts with or depends on another feature
  procedural  → ask how to implement, sequence, or perform something described here

Return JSON exactly:
{{"question": "...", "answer": "...", "key_terms": ["...", ...]}}"""


# ---------------------------------------------------------------------------
# Data loading

def _load(path: Path) -> list | dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Lookup items — deterministic, no LLM

def _build_lookup_items(fields: list[dict], field_index: dict, n: int) -> list[dict]:
    good = [
        f for f in fields
        if f.get("name")
        and len(f.get("description", "")) > 25
        and f.get("offset")
    ]

    # One item per unique field name
    seen: set = set()
    unique: list[dict] = []
    for f in good:
        key = f["name"].upper()
        if key not in seen:
            seen.add(key)
            unique.append(f)

    random.seed(42)
    sample = random.sample(unique, min(n, len(unique)))

    items: list[dict] = []
    for i, f in enumerate(sample):
        name = f["name"]
        description = f["description"].strip()
        offset = f.get("offset", "")
        figure = str(f.get("figure_number", ""))

        # Expected sections from field_index (best-effort)
        index_entries = field_index.get(name.upper(), [])
        expected_sections = list({e["section_id"] for e in index_entries if e.get("section_id")})

        gold = (
            f"{name} (bits {offset}, Figure {figure}): {description}"
            if offset and figure
            else f"{name}: {description}"
        )

        items.append({
            "id": f"lookup_{i+1:03d}",
            "query": f"What does {name} indicate?",
            "type": "lookup",
            "expected_sections": expected_sections,
            "expected_fields": [name],
            "expected_figure": figure,
            "gold_answer": gold,
            "source": f"fields.json:{name}",
            "tags": ["bit-field"],
        })

    return items


# ---------------------------------------------------------------------------
# LLM-generated items (structural / relational / procedural)

_TYPE_KEYWORDS: dict[str, list[str]] = {
    "structural": [
        "data structure", "format", "field", "command", "register",
        "consists", "contains", "layout", "identifies", "specifies",
    ],
    "relational": [
        "interact", "relationship", "depend", "associate", "link",
        "refer", "related", "map", "required", "enable",
    ],
    "procedural": [
        "procedure", "sequence", "step", "shall", "submit", "process",
        "initialize", "configure", "perform", "issue", "complete",
    ],
}


def _card_score(card: dict, qtype: str) -> int:
    text = ((card.get("summary") or "") + " " + (card.get("title") or "")).lower()
    return sum(1 for kw in _TYPE_KEYWORDS.get(qtype, []) if kw in text)


def _gen_llm_items(cards: list[dict], qtype: str, n: int, dry_run: bool) -> list[dict]:
    eligible = [c for c in cards if len(c.get("summary", "")) > 60]
    pool = sorted(eligible, key=lambda c: _card_score(c, qtype), reverse=True)[:max(n * 4, 40)]

    random.seed(43 + abs(hash(qtype)) % 97)
    sample = random.sample(pool, min(n, len(pool)))

    if dry_run:
        return [
            {
                "id": f"{qtype}_{i+1:03d}",
                "query": f"[DRY RUN — {qtype} from {c.get('section_id')}]",
                "type": qtype,
                "expected_sections": [c.get("section_id", "")],
                "gold_answer": "[DRY RUN]",
                "source": f"cards.json:{c.get('section_id')}",
                "tags": [qtype],
            }
            for i, c in enumerate(sample)
        ]

    items: list[dict] = []
    for i, card in enumerate(sample):
        section_id = card.get("section_id", "?")
        title = card.get("title", "")
        summary = card.get("summary", "")

        prompt = _LLM_PROMPT.format(
            section_id=section_id,
            title=title,
            summary=summary[:800],
            qtype=qtype,
        )

        try:
            parsed, _ = generate_json(prompt, system=_LLM_SYSTEM, temperature=0.3, max_output_tokens=512)
        except Exception as e:
            print(f"  [LLM error on {section_id}: {e}]", file=sys.stderr)
            continue

        question = (parsed.get("question") or "").strip()
        answer = (parsed.get("answer") or "").strip()
        key_terms = parsed.get("key_terms") or []

        if not question or not answer:
            continue

        items.append({
            "id": f"{qtype}_{i+1:03d}",
            "query": question,
            "type": qtype,
            "expected_sections": [section_id],
            "gold_answer": answer,
            "key_terms": key_terms,
            "source": f"cards.json:{section_id}",
            "tags": [qtype],
        })
        print(f"  [{qtype}] {section_id}: {question[:70]}...")

    return items


# ---------------------------------------------------------------------------
# Main

def generate_eval_set(
    count: int = 60,
    types: list[str] | None = None,
    dry_run: bool = False,
    output: Path = DEFAULT_OUTPUT,
    seed: int = 42,
) -> list[dict]:
    active_types = types or list(TYPE_FRACTIONS)

    # Validate data files exist
    for path in (CARDS_PATH, FIELDS_PATH, FIELD_INDEX_PATH):
        if not path.exists():
            print(f"Error: {path} not found. Run Phase 1 first.", file=sys.stderr)
            sys.exit(1)

    cards = _load(CARDS_PATH)
    fields = _load(FIELDS_PATH)
    field_index = _load(FIELD_INDEX_PATH)

    print(f"Loaded {len(cards)} cards, {len(fields)} fields")

    # Compute per-type counts proportionally
    active_fractions = {t: TYPE_FRACTIONS[t] for t in active_types}
    total_weight = sum(active_fractions.values())
    type_counts = {
        t: max(1, round(count * w / total_weight))
        for t, w in active_fractions.items()
    }
    print(f"Generating: {type_counts}")

    items: list[dict] = []

    if "lookup" in active_types:
        print("\nBuilding lookup items (from fields.json)...")
        items += _build_lookup_items(fields, field_index, type_counts["lookup"])
        print(f"  {sum(1 for x in items if x['type'] == 'lookup')} lookup items")

    for qtype in ("structural", "relational", "procedural"):
        if qtype in active_types:
            print(f"\nGenerating {qtype} items via LLM...")
            items += _gen_llm_items(cards, qtype, type_counts[qtype], dry_run)
            n = sum(1 for x in items if x["type"] == qtype)
            print(f"  {n} {qtype} items")

    # Re-assign sequential IDs
    for i, item in enumerate(items):
        item["id"] = f"eval_{i+1:03d}"

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(items)} eval items to {output}")
    return items


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate NVMe spec QA eval set.")
    parser.add_argument("--count", type=int, default=60, help="Total items to generate")
    parser.add_argument(
        "--types",
        nargs="+",
        choices=list(TYPE_FRACTIONS),
        help="Question types to include (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan and build lookup items only, skip LLM calls",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    generate_eval_set(
        count=args.count,
        types=args.types,
        dry_run=args.dry_run,
        output=args.output,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
