"""
Phase 2 — Step 2.2: Supabase Indexer

Reads chunks_embedded.json (1,905 chunks with embeddings) and upserts
them into the spec_chunks table in Supabase.

Data flow:
  chunks_embedded.json → Supabase spec_chunks table
  (pgvector + tsvector + metadata indexes)

Requires env vars:
  SUPABASE_URL  — project URL (https://xxxxx.supabase.co)
  SUPABASE_KEY  — service_role key
"""

import json
import os
import sys
from pathlib import Path

try:
    from supabase import create_client
except ImportError:
    print("Missing dependency: pip install supabase")
    sys.exit(1)


BATCH_SIZE = 100


def load_embedded_chunks(data_dir: Path) -> list[dict]:
    path = data_dir / "chunks_embedded.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def chunk_to_row(chunk: dict) -> dict:
    return {
        "id": chunk["chunk_id"],
        "embedding": chunk["embedding"],
        "text": chunk["text"],
        "text_raw": chunk["text_raw"],
        "content_type": chunk["content_type"],
        "section_id": chunk.get("section_id"),
        "section_title": chunk.get("section_title"),
        "spec_version": chunk.get("spec_version"),
        "spec_document": chunk.get("spec_document"),
        "pdf_pages": chunk.get("pdf_pages", []),
        "chunk_index": chunk.get("chunk_index"),
        "card_id": chunk.get("card_id"),
        "has_normative": chunk.get("has_normative", False),
        "figure_number": chunk.get("figure_number"),
        "word_count": chunk.get("word_count"),
        "token_count_approx": chunk.get("token_count_approx"),
        "row_start": chunk.get("row_start"),
        "row_end": chunk.get("row_end"),
    }


def upsert_chunks(chunks: list[dict], supabase_url: str, supabase_key: str) -> int:
    client = create_client(supabase_url, supabase_key)
    total = len(chunks)
    batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    uploaded = 0

    print(f"Upserting {total} chunks in {batches} batches")

    for i in range(0, total, BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch = chunks[i : i + BATCH_SIZE]
        rows = [chunk_to_row(c) for c in batch]

        client.table("spec_chunks").upsert(rows).execute()

        uploaded += len(rows)
        print(f"  Batch {batch_num}/{batches} — {uploaded}/{total} rows upserted")

    return uploaded


def run(data_dir: str = "data") -> None:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("SUPABASE_URL="):
                    supabase_url = supabase_url or line.split("=", 1)[1].strip().strip('"').strip("'")
                if line.startswith("SUPABASE_KEY="):
                    supabase_key = supabase_key or line.split("=", 1)[1].strip().strip('"').strip("'")

    if not supabase_url or not supabase_key:
        print("ERROR: Set SUPABASE_URL and SUPABASE_KEY as env vars or in .env file")
        sys.exit(1)

    data = Path(data_dir)
    chunks = load_embedded_chunks(data)
    print(f"Loaded {len(chunks)} embedded chunks")

    if not chunks[0].get("embedding"):
        print("ERROR: First chunk has no embedding — run embedder.py first")
        sys.exit(1)

    uploaded = upsert_chunks(chunks, supabase_url, supabase_key)
    print(f"Done. {uploaded} rows in spec_chunks.")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else os.getenv("SPEC_DATA_DIR", "data"))
