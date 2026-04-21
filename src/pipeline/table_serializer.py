"""
Phase 2 — Step 2.1b: Table Serializer

Reads tables.json + cards.json from Phase 1.
Converts each structured table into a readable text representation,
wraps it as a chunk with the same schema as prose chunks.

Data flow:
  tables.json (717 structured tables)
  + cards.json (1036 metadata cards with summaries)
  → chunks_tables.json (one chunk per table)
"""

import json
import sys
from pathlib import Path


WORDS_PER_TOKEN = 0.75


def word_count(text: str) -> int:
    return len(text.split())

# lookup optimizer, returns { {"section#"}: card, {"section#"}: card}
def build_card_index(cards: list[dict]) -> dict[str, dict]:
    return {c["section_id"]: c for c in cards}

# anchors context into ever table through llm summary
def get_card_prefix(card: dict | None) -> str:
    if not card or not card.get("summary"):
        return ""
    return f"[{card['section_id']} — {card['title']}] {card['summary']}"


def serialize_table(table: dict) -> str:
    """
    Convert a structured table into readable text.

    Format:
      Figure {N} — {Caption}
      {header1} | {header2} | ...
      ---
      {row1_col1} | {row1_col2} | ...
      {row2_col1} | {row2_col2} | ...
    """
    lines = []

    caption = table.get("caption", "")
    fig = table.get("figure_number", "")
    if fig and caption:
        lines.append(f"Figure {fig} — {caption}")
    elif caption:
        lines.append(caption)

    headers = table.get("headers", [])
    rows = table.get("rows", [])

    if headers:
        lines.append(" | ".join(headers))
        lines.append("---")

    for row in rows:
        lines.append(" | ".join(str(cell) for cell in row))

    return "\n".join(lines)


def make_table_chunk(table: dict, card: dict | None, index: int) -> dict:
    """Build one chunk dict from a single table."""
    serialized = serialize_table(table)
    prefix = get_card_prefix(card)
    enriched_text = f"{prefix}\n\n{serialized}" if prefix else serialized

    section_id = table.get("parent_section", "")
    fig = table.get("figure_number", "")

    return {
        "chunk_id": f"fig{fig}__{section_id}" if fig else f"table{index}__{section_id}",
        "section_id": section_id,
        "section_title": table.get("caption", ""),
        "spec_document": card.get("spec_document", "NVM Express Base Specification") if card else "NVM Express Base Specification",
        "spec_version": card.get("spec_version", "2.1") if card else "2.1",
        "content_type": "table",
        "text": enriched_text,
        "text_raw": serialized,
        "word_count": word_count(serialized),
        "token_count_approx": int(word_count(serialized) / WORDS_PER_TOKEN),
        "pdf_pages": [table.get("pdf_page")] if table.get("pdf_page") else [],
        "chunk_index": 0,
        "has_normative": "shall" in serialized.lower(),
        "card_id": section_id,
        "figure_number": str(fig) if fig else None,
    }


def run(data_dir: str = "data") -> list[dict]:
    data = Path(data_dir)

    with open(data / "tables.json", encoding="utf-8") as f:
        tables = json.load(f)
    with open(data / "cards.json", encoding="utf-8") as f:
        cards = json.load(f)

    card_index = build_card_index(cards)

    all_chunks = []
    for i, table in enumerate(tables):
        section_id = table.get("parent_section", "")
        card = card_index.get(section_id)
        chunk = make_table_chunk(table, card, i)
        all_chunks.append(chunk)

    out_path = data / "chunks_tables.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    word_counts = [c["word_count"] for c in all_chunks]
    print(f"Tables processed:    {len(tables)}")
    print(f"Chunks produced:     {len(all_chunks)}")
    print(f"Avg words/chunk:     {sum(word_counts)/len(word_counts):.0f}")
    print(f"Max words/chunk:     {max(word_counts)}")
    print(f"Min words/chunk:     {min(word_counts)}")
    print(f"Chunks > 1000 words: {sum(1 for w in word_counts if w > 1000)}")
    print(f"Chunks > 5000 words: {sum(1 for w in word_counts if w > 5000)}")
    print(f"Output: {out_path}")

    return all_chunks


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "data")
