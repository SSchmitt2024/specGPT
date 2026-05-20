"""
Phase 2 — Step 2.1b: Table Serializer

Reads tables.json + cards.json from Phase 1.
Converts each structured table into embeddable text chunks.

Small tables (≤ SPLIT_ROW_THRESHOLD rows): one chunk per table.
Large tables (> SPLIT_ROW_THRESHOLD rows): split into row-group chunks of
ROWS_PER_CHUNK rows each. Every chunk includes the table caption + headers
so it is self-contained for embedding and retrieval.

All chunks from the same table share the same figure_number, letting you
reconstruct the full table or find siblings via:
  SELECT * FROM spec_chunks WHERE figure_number = '328' ORDER BY chunk_index

Data flow:
  tables.json (717 structured tables)
  + cards.json (1036 metadata cards with summaries)
  → chunks_tables.json (one or more chunks per table)
"""

import json
import sys
from pathlib import Path


WORDS_PER_TOKEN = 0.75
SPLIT_ROW_THRESHOLD = 30   # tables with more rows than this get split
ROWS_PER_CHUNK = 25        # rows per chunk when splitting


def word_count(text: str) -> int:
    return len(text.split())


def build_card_index(cards: list[dict]) -> dict[str, dict]:
    return {c["section_id"]: c for c in cards}


def get_card_prefix(card: dict | None) -> str:
    if not card or not card.get("summary"):
        return ""
    return f"[{card['section_id']} — {card['title']}] {card['summary']}"


def serialize_table(table: dict) -> str:
    """Serialize a full table to text (used by retriever.py structured lookup)."""
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


def _header_block(table: dict) -> str:
    """Caption + column headers — prepended to every chunk of a split table."""
    lines = []
    caption = table.get("caption", "")
    fig = table.get("figure_number", "")
    if fig and caption:
        lines.append(f"Figure {fig} — {caption}")
    elif caption:
        lines.append(caption)
    headers = table.get("headers", [])
    if headers:
        lines.append(" | ".join(headers))
        lines.append("---")
    return "\n".join(lines)


def _make_chunk(
    table: dict,
    card: dict | None,
    serialized: str,
    chunk_index: int,
    row_start: int,
    row_end: int,
    table_index: int,
) -> dict:
    fig = table.get("figure_number", "")
    section_id = table.get("parent_section", "")
    prefix = get_card_prefix(card)
    enriched = f"{prefix}\n\n{serialized}" if prefix else serialized

    base_id = f"fig{fig}__{section_id}" if fig else f"table{table_index}__{section_id}"
    chunk_id = f"{base_id}__{chunk_index}" if chunk_index > 0 else base_id

    return {
        "chunk_id":           chunk_id,
        "section_id":         section_id,
        "section_title":      table.get("caption", ""),
        "spec_document":      card.get("spec_document", "NVM Express Base Specification") if card else "NVM Express Base Specification",
        "spec_version":       card.get("spec_version", "2.1") if card else "2.1",
        "content_type":       "table",
        "text":               enriched,
        "text_raw":           serialized,
        "word_count":         word_count(serialized),
        "token_count_approx": int(word_count(serialized) / WORDS_PER_TOKEN),
        "pdf_pages":          [table.get("pdf_page")] if table.get("pdf_page") else [],
        "chunk_index":        chunk_index,
        "row_start":          row_start,
        "row_end":            row_end,
        "has_normative":      "shall" in serialized.lower(),
        "card_id":            section_id,
        "figure_number":      str(fig) if fig else None,
    }


def make_table_chunks(table: dict, card: dict | None, table_index: int) -> list[dict]:
    """
    Produce one or more chunks from a table.

    Small tables become a single chunk (chunk_index=0, same behaviour as before).
    Large tables are split into row groups of ROWS_PER_CHUNK, each prefixed
    with the table caption and headers so the chunk is self-contained.
    """
    rows = table.get("rows", [])

    if len(rows) <= SPLIT_ROW_THRESHOLD:
        serialized = serialize_table(table)
        return [_make_chunk(table, card, serialized, 0, 0, len(rows) - 1, table_index)]

    header_block = _header_block(table)
    chunks: list[dict] = []

    for chunk_index, start in enumerate(range(0, len(rows), ROWS_PER_CHUNK)):
        row_group = rows[start : start + ROWS_PER_CHUNK]
        end = start + len(row_group) - 1

        row_lines = [" | ".join(str(cell) for cell in row) for row in row_group]
        serialized = f"{header_block}\n" + "\n".join(row_lines)

        chunks.append(_make_chunk(table, card, serialized, chunk_index, start, end, table_index))

    return chunks


def run(data_dir: str = "data") -> list[dict]:
    data = Path(data_dir)

    with open(data / "tables.json", encoding="utf-8") as f:
        tables = json.load(f)
    with open(data / "cards.json", encoding="utf-8") as f:
        cards = json.load(f)

    card_index = build_card_index(cards)

    all_chunks: list[dict] = []
    split_count = 0
    for i, table in enumerate(tables):
        section_id = table.get("parent_section", "")
        card = card_index.get(section_id)
        chunks = make_table_chunks(table, card, i)
        if len(chunks) > 1:
            split_count += 1
        all_chunks.extend(chunks)

    out_path = data / "chunks_tables.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    word_counts = [c["word_count"] for c in all_chunks]
    print(f"Tables processed:    {len(tables)}")
    print(f"Tables split:        {split_count}")
    print(f"Chunks produced:     {len(all_chunks)}")
    print(f"Avg words/chunk:     {sum(word_counts)/len(word_counts):.0f}")
    print(f"Max words/chunk:     {max(word_counts)}")
    print(f"Chunks > 1000 words: {sum(1 for w in word_counts if w > 1000)}")
    print(f"Output: {out_path}")

    return all_chunks


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "data")
