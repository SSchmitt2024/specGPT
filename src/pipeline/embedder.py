"""
Phase 2 — Step 2.1c: Embedding Pipeline

Reads chunks_prose.json + chunks_tables.json from Phase 1.
Embeds the enriched text field of each chunk via Voyage AI.
Writes chunks_embedded.json — the same chunk dicts with an "embedding" field added.

Data flow:
  chunks_prose.json (1,188 chunks)
  + chunks_tables.json (717 chunks)
  → chunks_embedded.json (1,905 chunks with 1024-dim vectors)

Voyage AI config:
  Model: voyage-3-lite (1024 dims, 512 token context)
  Batch size: 128 (API max)
  Rate limit: ~300 RPM on free tier → sleep between batches
"""

import json
import os
import sys
import time
from pathlib import Path
import voyageai



MODEL = "voyage-3-lite"
BATCH_SIZE = 128
SLEEP_BETWEEN_BATCHES = 0.5

COST_PER_MILLION_TOKENS = 0.02
MAX_BUDGET_DOLLARS = 0.50
WORDS_PER_TOKEN = 0.75

# flattens data into one long pipeline of things to be embedded
def load_chunks(data_dir: Path) -> list[dict]:
    prose_path = data_dir / "chunks_prose.json"
    tables_path = data_dir / "chunks_tables.json"

    chunks = []
    for path in [prose_path, tables_path]:
        with open(path, encoding="utf-8") as f:
            chunks.extend(json.load(f))

    return chunks


def estimate_cost(chunks: list[dict]) -> float:
    total_words = sum(len(c["text"].split()) for c in chunks)
    total_tokens = total_words / WORDS_PER_TOKEN
    return (total_tokens / 1_000_000) * COST_PER_MILLION_TOKENS


def embed_chunks(chunks: list[dict], client: voyageai.Client) -> list[dict]:
    texts = [c["text"] for c in chunks]
    total = len(texts)
    all_embeddings: list[list[float]] = [None] * total

    batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Embedding {total} chunks in {batches} batches (model={MODEL})")

    for i in range(0, total, BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch_texts = texts[i : i + BATCH_SIZE]

        result = client.embed(batch_texts, model=MODEL, input_type="document")

        for j, emb in enumerate(result.embeddings):
            all_embeddings[i + j] = emb

        print(f"  Batch {batch_num}/{batches} — {len(batch_texts)} chunks embedded")

        if i + BATCH_SIZE < total:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    for idx, chunk in enumerate(chunks):
        chunk["embedding"] = all_embeddings[idx]

    return chunks


def run(data_dir: str = "data") -> list[dict]:
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("VOYAGE_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if not api_key:
        print("ERROR: Set VOYAGE_API_KEY as env var or in .env file")
        sys.exit(1)

    data = Path(data_dir)
    client = voyageai.Client(api_key=api_key)

    chunks = load_chunks(data)
    print(f"Loaded {len(chunks)} chunks ({sum(1 for c in chunks if c['content_type']=='prose')} prose, {sum(1 for c in chunks if c['content_type']=='table')} table)")

    est = estimate_cost(chunks)
    print(f"Estimated cost: ${est:.4f} (budget: ${MAX_BUDGET_DOLLARS:.2f})")
    if est > MAX_BUDGET_DOLLARS:
        print(f"ABORT: Estimated cost ${est:.4f} exceeds hard limit ${MAX_BUDGET_DOLLARS:.2f}")
        sys.exit(1)

    chunks = embed_chunks(chunks, client)

    dims = len(chunks[0]["embedding"])
    print(f"Embedding dimensions: {dims}")

    out_path = data / "chunks_embedded.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Output: {out_path} ({size_mb:.1f} MB)")
    print(f"Chunks with embeddings: {sum(1 for c in chunks if c.get('embedding'))}")

    return chunks


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "data")
