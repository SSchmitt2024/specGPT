"""
Phase 2 — Step 2.1a: Chunking Engine

Reads prose.json + cards.json from Phase 1.
Produces enriched, overlapping text chunks ready for embedding.

Data flow:
  prose.json (914 sections, 6275 paragraphs)
  + cards.json (1036 metadata cards with summaries)
  → enriched_chunks.json (chunk text + metadata per chunk)
"""

import json
import os
import sys
from pathlib import Path


# ── tokenization ──────────────────────────────────────────────
# We approximate tokens as whitespace-split words.
# GPT/Claude tokenizers average ~0.75 words per token for technical English,
# so 500 tokens ≈ 375 words. We use word count with a conservative ratio.

WORDS_PER_TOKEN = 0.75
TARGET_TOKENS = 500
OVERLAP_TOKENS = 50

TARGET_WORDS = int(TARGET_TOKENS * WORDS_PER_TOKEN)   # 375
OVERLAP_WORDS = int(OVERLAP_TOKENS * WORDS_PER_TOKEN)  # 37


def word_count(text: str) -> int:
    return len(text.split())


# ── card lookup ───────────────────────────────────────────────
# Each chunk gets the card summary prepended so the embedding
# captures section-level context ("definition-enriched").

def build_card_index(cards: list[dict]) -> dict[str, dict]:
    """Map section_id → card for fast lookup."""
    return {c["section_id"]: c for c in cards}


def get_card_prefix(card: dict | None) -> str:
    """Build the summary line that gets prepended to every chunk."""
    if not card or not card.get("summary"):
        return ""
    return f"[{card['section_id']} — {card['title']}] {card['summary']}"


# ── chunking logic ────────────────────────────────────────────
# Paragraphs are small (avg 33 words), so we merge consecutive
# paragraphs until hitting the word budget, then start a new chunk.
# Overlap: the last N words of chunk[i] are prepended to chunk[i+1]
# so the embedding model sees continuity across boundaries.

def chunk_section(section: dict, card: dict | None) -> list[dict]:
    """
    Chunk one prose section into overlapping segments.

    Returns a list of chunk dicts ready for embedding.
    """
    paragraphs = section.get("paragraphs", [])
    if not paragraphs:
        return []

    # Flatten paragraphs into (text, page) pairs
    texts = [(p["text"].strip(), p.get("pdf_page")) for p in paragraphs]
    texts = [(t, pg) for t, pg in texts if t]
    if not texts:
        return []

    prefix = get_card_prefix(card)
    section_id = section["section_number"]
    section_title = section.get("title", "")

    # Merge paragraphs into chunks
    chunks = []
    current_words: list[str] = []
    current_pages: set[int] = set()
    start_para_idx = 0

    for i, (text, page) in enumerate(texts):
        words = text.split()

        # If adding this paragraph exceeds the budget, finalize current chunk
        if current_words and (len(current_words) + len(words)) > TARGET_WORDS:
            chunk_text = " ".join(current_words)
            chunks.append(_make_chunk(
                prefix=prefix,
                body=chunk_text,
                section_id=section_id,
                section_title=section_title,
                section=section,
                card=card,
                pages=sorted(current_pages),
                chunk_index=len(chunks),
            ))

            # Overlap: keep the last OVERLAP_WORDS from this chunk
            overlap = current_words[-OVERLAP_WORDS:] if len(current_words) > OVERLAP_WORDS else current_words[:]
            current_words = list(overlap)
            current_pages = {page} if page else set()
            start_para_idx = i

        current_words.extend(words)
        if page:
            current_pages.add(page)

    # Final chunk from remaining words
    if current_words:
        chunk_text = " ".join(current_words)
        chunks.append(_make_chunk(
            prefix=prefix,
            body=chunk_text,
            section_id=section_id,
            section_title=section_title,
            section=section,
            card=card,
            pages=sorted(current_pages),
            chunk_index=len(chunks),
        ))

    return chunks


def _make_chunk(
    prefix: str,
    body: str,
    section_id: str,
    section_title: str,
    section: dict,
    card: dict | None,
    pages: list[int],
    chunk_index: int,
) -> dict:
    """Assemble one chunk dict with all metadata needed downstream."""
    # The enriched text = card summary + chunk body.
    # This is what gets embedded. The summary gives the embedding model
    # section-level context so "this field" in a paragraph resolves to
    # the right concept.
    enriched_text = f"{prefix}\n\n{body}" if prefix else body

    normative_tags = []
    for n in section.get("normative", []):
        if any(w in body.lower() for w in [n.get("keyword", "").lower()]):
            normative_tags.append(n)

    return {
        "chunk_id": f"{section_id}__c{chunk_index}",
        "section_id": section_id,
        "section_title": section_title,
        "spec_document": card.get("spec_document", "NVM Express Base Specification") if card else "NVM Express Base Specification",
        "spec_version": card.get("spec_version", "2.1") if card else "2.1",
        "content_type": "prose",
        "text": enriched_text,
        "text_raw": body,
        "word_count": word_count(body),
        "token_count_approx": int(word_count(body) / WORDS_PER_TOKEN),
        "pdf_pages": pages,
        "chunk_index": chunk_index,
        "has_normative": len(normative_tags) > 0,
        "card_id": section_id,
    }


# ── main pipeline ─────────────────────────────────────────────

def run(data_dir: str = "data") -> list[dict]:
    data = Path(data_dir)

    with open(data / "prose.json", encoding="utf-8") as f:
        prose = json.load(f)
    with open(data / "cards.json", encoding="utf-8") as f:
        cards = json.load(f)

    card_index = build_card_index(cards)

    all_chunks = []
    for section in prose:
        sid = section["section_number"]
        card = card_index.get(sid)
        chunks = chunk_section(section, card)
        all_chunks.extend(chunks)

    # Write output
    out_path = data / "chunks_prose.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Stats
    word_counts = [c["word_count"] for c in all_chunks]
    print(f"Sections processed:  {len(prose)}")
    print(f"Chunks produced:     {len(all_chunks)}")
    print(f"Avg words/chunk:     {sum(word_counts)/len(word_counts):.0f}")
    print(f"Max words/chunk:     {max(word_counts)}")
    print(f"Min words/chunk:     {min(word_counts)}")
    print(f"Chunks > 400 words:  {sum(1 for w in word_counts if w > 400)}")
    print(f"Chunks < 20 words:   {sum(1 for w in word_counts if w < 20)}")
    print(f"Output: {out_path}")

    return all_chunks


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else os.getenv("SPEC_DATA_DIR", "data"))
