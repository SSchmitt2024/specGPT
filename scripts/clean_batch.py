"""Clean a batch-context JSONL in place (offline) before sending it to a batch API.

Standalone: no Supabase, no specGPT imports, no network. Pure JSONL rewrite.

Fixes, in order:
  1. drop records with no sources / empty context
  2. drop duplicate questions (keep first)
  3. dedup chunks whose body repeats inside one context (null ids defeat the
     id-based dedup upstream), renumber the fences
  4. optional token cap: drop lowest-reranked chunks until under budget
  5. optional RAFT no-oracle slice: for a seeded random P% of records, strip the
     top-N reranked chunks so the answer is NOT derivable from what remains.
     Those records get "no_oracle": true. They are the examples that teach the
     student to refuse instead of answering from distractors.
  6. stamp a stable "id" (hash of the question) so batch-API responses can be
     joined back by key instead of by line order.
  7. optional held-out "split", grouped so an eval question's own oracle section
     never appears as the oracle of a train question.

`sources` is kept 1:1 with the chunks in `context` at every step.

Chunk ORDER is left alone: the live app serves chunks in reranked order, so the
96% "best chunk first" skew here is the real serving distribution. Shuffling it
would train the student on a layout production never sends.

Usage:
    python scripts/clean_batch.py IN.jsonl -o OUT.jsonl
    python scripts/clean_batch.py IN.jsonl -o OUT.jsonl --no-oracle-frac 0.25 --eval-frac 0.05
    python scripts/clean_batch.py --self-check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

PREAMBLE = (
    "Sources below are drawn from several specifications. Section numbers\n"
    "are per-document, so the same number in two of them is two different\n"
    "sections. The tag after each heading names the document:\n"
)
_CHUNK_RE = re.compile(
    r"===== CHUNK (\d+) =====\n(.*?)\n===== CHUNK END \1 =====", re.S
)

# ponytail: chars/4, same shape as the app's estimator. Only used to compare
# against a budget, so a few percent of drift costs nothing.
def est_tokens(s: str) -> int:
    return len(s) // 4


def split_chunks(context: str) -> list[str]:
    """Chunk bodies (header line + text), fences stripped."""
    return [m.group(2) for m in _CHUNK_RE.finditer(context)]


def build_context(sources: list[dict], bodies: list[str]) -> str:
    """Rebuild a context from the surviving chunks, byte-identical to what the
    upstream assembler emits: the spec legend appears only when more than one
    spec is present, lists just the specs present, tag-sorted, and falls back to
    the tag itself when a source carries no spec_document (3k sources do)."""
    if not bodies:
        return ""
    docs: dict[str, str] = {}
    for s in sources:
        if not s.get("spec"):
            continue
        tag = s["spec"].upper()
        # First source of a spec fixes its legend line, even if its
        # spec_document is null (upstream does not look ahead for a better one).
        docs.setdefault(tag, s.get("spec_document") or tag)
    head = ""
    if len(docs) > 1:
        head = PREAMBLE + "".join(f"  <{t}> = {docs[t]}\n" for t in sorted(docs)) + "\n"
    fences = [
        f"===== CHUNK {i} =====\n{b}\n===== CHUNK END {i} ====="
        for i, b in enumerate(bodies, 1)
    ]
    return head + "\n\n".join(fences)


def _norm(body: str) -> str:
    return re.sub(r"\s+", " ", body).strip().lower()


def clean_record(
    rec: dict, *, max_tokens: int | None, strip_top: int, make_no_oracle: bool
) -> tuple[dict | None, Counter]:
    st = Counter()
    sources, bodies = rec.get("sources") or [], split_chunks(rec.get("context") or "")
    if not sources or not bodies:
        st["dropped_empty"] += 1
        return None, st
    if len(sources) != len(bodies):
        # Never seen in practice (verified 0/6036), but a silent misalignment
        # would mislabel every oracle downstream. Refuse the record instead.
        st["dropped_misaligned"] += 1
        return None, st

    pairs = list(zip(sources, bodies))

    seen, deduped = set(), []
    for s, b in pairs:
        key = _norm(b)
        if key in seen:
            st["chunks_deduped"] += 1
            continue
        seen.add(key)
        deduped.append((s, b))
    pairs = deduped

    if make_no_oracle:
        # Strip the strongest evidence, keep the tail as distractors. Depth is
        # varied per record: a shallow strip leaves the context looking very
        # on-topic with the governing statement gone (the hard negative
        # production actually sees), a deep strip leaves nothing relevant at all
        # (the easy one). Uniform depth teaches only the easy discrimination.
        order = sorted(
            range(len(pairs)), key=lambda i: pairs[i][0].get("rerank_score") or 0, reverse=True
        )
        drop = set(order[:strip_top])
        kept = [p for i, p in enumerate(pairs) if i not in drop]
        if not kept:
            st["dropped_no_oracle_empty"] += 1
            return None, st
        st["chunks_stripped_for_no_oracle"] += len(pairs) - len(kept)
        pairs = kept

    if max_tokens:
        # Drop weakest chunks until under budget. Preserve original order.
        while pairs and est_tokens(build_context([s for s, _ in pairs], [b for _, b in pairs])) > max_tokens:
            worst = min(range(len(pairs)), key=lambda i: pairs[i][0].get("rerank_score") or 0)
            pairs.pop(worst)
            st["chunks_dropped_for_budget"] += 1
        if not pairs:
            st["dropped_budget_empty"] += 1
            return None, st

    out_sources = [s for s, _ in pairs]
    out = {
        # Stable across reruns and reordering, so the batch API's custom_id can
        # carry it and responses join by key. Joining by line order silently
        # mispairs every answer if either file is ever filtered or resorted.
        "id": hashlib.sha1(rec["question"].encode()).hexdigest()[:16],
        "question": rec["question"],
        "context": build_context(out_sources, [b for _, b in pairs]),
        "sources": out_sources,
    }
    if make_no_oracle:
        out["no_oracle"] = True
        # How deep the strip was, so negatives can be sorted by difficulty once
        # the teacher's responses come back.
        out["no_oracle_stripped"] = strip_top
        st["no_oracle_records"] += 1
        st[f"no_oracle_depth_{strip_top}"] += 1
    st["kept"] += 1
    return out, st


def oracle_key(rec: dict) -> tuple:
    """The (spec, section) the record's strongest chunk comes from. Used to keep
    a section's records on one side of the train/eval split."""
    srcs = rec.get("sources") or []
    if not srcs:
        return ("", "")
    top = max(srcs, key=lambda s: s.get("rerank_score") or 0)
    return (top.get("spec") or "", top.get("section_id") or top.get("figure_number") or "")


def assign_splits(staged: list[dict], frac: float, seed: int) -> dict[int, str]:
    """Hold out whole oracle sections, not individual records. A random split
    leaks: the section that answers an eval question would also be the oracle of
    several train questions, so eval would score memorisation as skill.

    ponytail: groups on the TOP chunk's section only. A held-out section can
    still appear as a weak distractor in a train context, which is harmless.
    Full isolation would need a connected-component split over every chunk and
    would strand most of the data in one giant component."""
    if not frac:
        return {}
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(staged):
        groups.setdefault(oracle_key(r), []).append(i)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target, out, n = int(len(staged) * frac), {}, 0
    for k in keys:
        if n >= target:
            break
        for i in groups[k]:
            out[i] = "eval"
        n += len(groups[k])
    return out


def run(inp: Path, outp: Path, *, max_tokens, strip_top, frac, seed, eval_frac=0.0) -> Counter:
    rows = [json.loads(l) for l in inp.open() if l.strip()]
    stats = Counter(total_in=len(rows))

    seen_q, staged = set(), []
    for r in rows:
        q = (r.get("question") or "").strip()
        if q in seen_q:
            stats["dropped_duplicate_question"] += 1
            continue
        seen_q.add(q)
        staged.append(r)

    # Choose the no-oracle slice from records that survive the cheap filters, so
    # the realised fraction matches what was asked for.
    eligible = [i for i, r in enumerate(staged) if (r.get("sources") and r.get("context"))]
    rng = random.Random(seed)
    picked = set(rng.sample(eligible, int(len(eligible) * frac))) if frac else set()
    splits = assign_splits(staged, eval_frac, seed)

    with outp.open("w") as f:
        for i, r in enumerate(staged):
            rec, st = clean_record(
                r, max_tokens=max_tokens, strip_top=rng.choice(strip_top),
                make_no_oracle=i in picked,
            )
            stats.update(st)
            if rec:
                if eval_frac:
                    rec["split"] = splits.get(i, "train")
                    stats[f"split_{rec['split']}"] += 1
                f.write(json.dumps(rec) + "\n")
    return stats


def verify(inp: Path) -> None:
    """Parse then rebuild every context with nothing removed. Any byte that
    changes is a parser/builder bug that would silently corrupt the output."""
    n = bad = 0
    for line in inp.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("sources"):
            continue
        n += 1
        if build_context(r["sources"], split_chunks(r["context"])) != r["context"]:
            bad += 1
    print(f"verify: {n - bad}/{n} contexts byte-identical after round-trip")
    assert bad == 0, f"{bad} contexts would be corrupted by the rewrite"


def self_check() -> None:
    """Round-trip: parsing a context and rebuilding it unchanged must be a no-op,
    otherwise every downstream context is silently corrupted."""
    src = [
        {"spec": "rdma", "spec_document": "NVM Express NVMe-over-RDMA Transport Specification", "rerank_score": 0.9},
        {"spec": "base", "spec_document": "NVM Express Base Specification", "rerank_score": 0.5},
        {"spec": "base", "spec_document": "NVM Express Base Specification", "rerank_score": 0.1},
    ]
    bodies = ["[Section 1.1] A  <RDMA>\nalpha text", "[Section 2.2] B  <BASE>\nbeta text", "[Section 2.2] B  <BASE>\nbeta text"]
    ctx = build_context(src, bodies)
    assert split_chunks(ctx) == bodies, "parse/build round-trip lost data"
    assert build_context(src, split_chunks(ctx)) == ctx, "rebuild is not idempotent"

    rec = {"question": "q", "context": ctx, "sources": src}
    out, st = clean_record(rec, max_tokens=None, strip_top=1, make_no_oracle=False)
    assert st["chunks_deduped"] == 1 and len(out["sources"]) == 2, "dedup failed"
    assert len(split_chunks(out["context"])) == 2, "sources/context drifted apart"
    assert "<BASE>" in out["context"] and "<RDMA>" in out["context"], "preamble lost a spec"

    out, st = clean_record(rec, max_tokens=None, strip_top=1, make_no_oracle=True)
    assert out["no_oracle"] and len(out["sources"]) == 1, "no-oracle strip failed"
    assert out["sources"][0]["rerank_score"] == 0.5, "stripped the wrong chunk"
    assert len(split_chunks(out["context"])) == 1, "sources/context drifted apart"

    assert clean_record({"question": "q", "context": "", "sources": []},
                        max_tokens=None, strip_top=1, make_no_oracle=False)[0] is None

    # ids must be stable and unique-per-question, or the batch join mispairs.
    a = clean_record(rec, max_tokens=None, strip_top=1, make_no_oracle=False)[0]
    b = clean_record({**rec, "sources": src[::-1]}, max_tokens=None, strip_top=1, make_no_oracle=False)[0]
    assert a["id"] == b["id"], "id must not depend on source order"
    assert a["id"] != clean_record({**rec, "question": "q2"}, max_tokens=None,
                                   strip_top=1, make_no_oracle=False)[0]["id"], "id collision"

    # split must hold out whole oracle sections, never straddle one.
    staged = [{"question": f"q{i}", "sources": [{"spec": "base", "section_id": f"{i % 4}", "rerank_score": 1}]}
              for i in range(40)]
    sp = assign_splits(staged, 0.25, 0)
    by_key: dict = {}
    for i, r in enumerate(staged):
        by_key.setdefault(oracle_key(r), set()).add(sp.get(i, "train"))
    assert all(len(v) == 1 for v in by_key.values()), "a section leaked across the split"
    assert 0 < sum(1 for v in sp.values() if v == "eval") < len(staged), "split is degenerate"
    print("self-check: ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?")
    ap.add_argument("-o", "--out")
    ap.add_argument("--max-tokens", type=int, default=None, help="cap context tokens per record")
    ap.add_argument("--no-oracle-frac", type=float, default=0.0, help="fraction of records to make unanswerable (RAFT)")
    ap.add_argument("--eval-frac", type=float, default=0.0, help="held-out fraction, grouped by oracle section")
    ap.add_argument("--strip-top", default="1,2,3,5",
                    help="comma-separated strip depths, chosen at random per no-oracle record")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--verify", action="store_true", help="round-trip the input, change nothing")
    a = ap.parse_args()

    if a.self_check:
        self_check()
    elif a.verify:
        if not a.input:
            ap.error("input is required")
        verify(Path(a.input))
    else:
        if not a.input or not a.out:
            ap.error("input and -o are required")
        depths = [int(x) for x in str(a.strip_top).split(",") if x.strip()]
        s = run(Path(a.input), Path(a.out), max_tokens=a.max_tokens,
                strip_top=depths, frac=a.no_oracle_frac, seed=a.seed,
                eval_frac=a.eval_frac)
        for k, v in sorted(s.items()):
            print(f"{k}: {v}")
