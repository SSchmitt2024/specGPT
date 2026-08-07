"""Submit batch_requests.jsonl to the Anthropic Batches API and collect results.

    python -m scripts.submit_batch requests.jsonl -o results.jsonl

Resumable: batch ids are appended to <out>.batches as they are created, so a
rerun with the same -o picks up already-submitted chunks instead of paying for
them twice. Delete that file to start over.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import anthropic

# 256 MB / 100k requests is the API cap; contexts here average ~15 KB, so size
# binds first. 1500 keeps each chunk well under with room for outliers.
CHUNK = 1500


def chunks(path: Path, n: int):
    buf = []
    for line in path.open():
        if line.strip():
            buf.append(json.loads(line))
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def submit(client, requests_file: Path, ledger: Path) -> list[str]:
    done = ledger.read_text().split() if ledger.exists() else []
    ids = list(done)
    with ledger.open("a") as f:
        for i, batch in enumerate(chunks(requests_file, CHUNK)):
            if i < len(done):
                continue
            bid = client.messages.batches.create(requests=batch).id
            f.write(bid + "\n")
            f.flush()  # a crash between create and flush orphans a paid batch
            ids.append(bid)
            print(f"submitted chunk {i} ({len(batch)} requests): {bid}")
    return ids


def wait(client, ids: list[str], poll: int) -> None:
    pending = set(ids)
    while pending:
        for bid in sorted(pending):
            b = client.messages.batches.retrieve(bid)
            if b.processing_status == "ended":
                pending.discard(bid)
                print(f"{bid}: ended {b.request_counts}")
        if pending:
            print(f"{len(pending)} batch(es) still processing")
            time.sleep(poll)


def collect(client, ids: list[str], out: Path) -> int:
    n = 0
    with out.open("w") as f:
        for bid in ids:
            for r in client.messages.batches.results(bid):
                f.write(r.model_dump_json() + "\n")
                n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("requests")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--poll", type=int, default=120, help="seconds between status checks")
    ap.add_argument("--collect-only", action="store_true",
                    help="skip submission, just fetch results for the ids in <out>.batches")
    a = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=key)

    out, ledger = Path(a.out), Path(a.out + ".batches")
    if a.collect_only:
        ids = ledger.read_text().split()
    else:
        ids = submit(client, Path(a.requests), ledger)
        wait(client, ids, a.poll)
    print(f"wrote {collect(client, ids, out)} results to {out}")
