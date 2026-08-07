"""Join batch answers back to their contexts and emit the SFT/RAFT training file.

    python -m scripts.build_sft_from_batch final_batch.clean.jsonl results.jsonl \
        -o data/raft_finetune/sft.jsonl   # -> sft.train.jsonl + sft.eval.jsonl

Gate: every answer is re-parsed with generator._extract_citations against its OWN
chunks. A tag that doesn't resolve comes back hallucinated=True and the example
is dropped, same rule build_raft_dataset.py applies. Training on an answer that
cites a section it wasn't given teaches exactly the failure the citations exist
to prevent.

The system prompt written here is production's, WITHOUT the teacher's CoT rules,
and the target is the stripped answer. The student learns to answer directly;
CoT was a teacher-side device for better answers, not a behaviour to copy.
Pass --keep-cot to train the reasoning too (then serve rules 12-14 as well, or
the student emits <thinking> the live app never strips).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from scripts.build_batch_requests import COT_RULE, strip_cot
from src.pipeline.generator import DEFAULT_SYSTEM_PROMPT, _extract_citations

REFUSAL_RE = re.compile(
    r"does not (contain|specify|address|provide|include)|"
    r"not (specified|addressed|found|available|covered)|"
    r"no information (is )?(available|provided)|context does not",
    re.IGNORECASE,
)

# generator._extract_citations parses section tags ONLY, BY DESIGN: its _BID
# pattern excludes bare bracketed integers so "[0]" and bit positions aren't read
# as citations, and the live app surfaces figures as chips (_figures_from_sources)
# rather than citations. Don't "fix" generator.py for this. But a training gate
# has no chips, and 27% of sources here are citable only by figure, so without
# the check below every figure-cited answer would be binned as ungrounded.
_FIG_TAG = re.compile(r"\[Figure\s+(\d+)\]")


def figure_tags(text: str, sources: list[dict]) -> tuple[int, int]:
    """(resolved, unresolved) figure tags in the answer."""
    have = {str(s.get("figure_number")) for s in sources if s.get("figure_number")}
    tags = set(_FIG_TAG.findall(text))
    return len(tags & have), len(tags - have)


def load_results(path: Path) -> dict[str, str]:
    """Accepts raw batch results or the output of `build_batch_requests --strip`."""
    out = {}
    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if "answer" in r:
            out[r["id"]] = r["answer"]
            continue
        res = r.get("result") or {}
        if res.get("type") not in (None, "succeeded"):
            continue
        msg = res.get("message") or {}
        # A max_tokens stop is a mid-sentence answer that still carries a valid
        # citation, so every gate below would pass it. Drop it here instead.
        if msg.get("stop_reason") == "max_tokens":
            continue
        # type=="text" only: Sonnet's reasoning arrives as separate thinking
        # blocks, so the CoT is discarded here by construction.
        out[r["custom_id"]] = strip_cot(
            "".join(c.get("text", "") for c in msg.get("content") or [] if c.get("type") == "text")
        )
    return out


def gate(rec: dict, ans: str, *, keep_cot: bool) -> tuple[str | None, str, bool]:
    """(drop_reason | None, answer_text, is_refusal)."""
    if not ans or not ans.strip():
        return "missing_answer", "", False
    visible = ans if keep_cot else strip_cot(ans)
    is_refusal = bool(REFUSAL_RE.search(visible[:400]))
    cites = _extract_citations(visible, rec["sources"])
    fig_ok, fig_bad = figure_tags(visible, rec["sources"])

    if any(c.get("hallucinated") for c in cites) or fig_bad:
        return "dropped_hallucinated_tag", visible, is_refusal
    if not is_refusal and not cites and not fig_ok:
        return "dropped_ungrounded", visible, is_refusal
    if rec.get("no_oracle") and not is_refusal:
        # Oracle was removed on purpose; a confident answer means it was built
        # from distractors. Exactly the label RAFT must not learn.
        return "dropped_answered_without_oracle", visible, is_refusal
    return None, visible, is_refusal


def build(clean: Path, results: Path, out: Path, *, keep_cot: bool,
          max_refusal_frac: float = 0.0, seed: int = 0) -> Counter:
    answers, st = load_results(results), Counter()

    # Pass 1 gates only. The refusal share can't be capped while streaming: it
    # isn't known until every record has been judged, and the gates themselves
    # move it (a no_oracle record Sonnet answered anyway is dropped, so refusals
    # survive at a higher rate than the 25% that went in).
    keep: dict[str, bool] = {}
    for line in clean.open():
        rec = json.loads(line)
        st["total"] += 1
        reason, _vis, is_refusal = gate(rec, answers.get(rec["id"], ""), keep_cot=keep_cot)
        if reason:
            st[reason] += 1
            continue
        keep[rec["id"]] = is_refusal

    refusals = [i for i, r in keep.items() if r]
    if max_refusal_frac and refusals:
        # RAFT wants roughly 20-40% unanswerable. Well past that and the student
        # learns refusing is the safe default, which is worse than the failure
        # mode being trained away.
        # Solve r/(answers+r) <= frac. int(len(keep)*frac) is the wrong cap:
        # dropping refusals shrinks the denominator too, so it lands over target.
        answered = len(keep) - len(refusals)
        cap = int(max_refusal_frac * answered / (1 - max_refusal_frac)) if max_refusal_frac < 1 else len(refusals)
        if len(refusals) > cap:
            drop = set(random.Random(seed).sample(refusals, len(refusals) - cap))
            keep = {i: r for i, r in keep.items() if i not in drop}
            st["dropped_excess_refusal"] = len(drop)

    stem = out.with_suffix("")
    paths = {"train": Path(f"{stem}.train.jsonl"), "eval": Path(f"{stem}.eval.jsonl")}
    out.parent.mkdir(parents=True, exist_ok=True)
    files = {k: p.open("w") for k, p in paths.items()}
    try:
        for line in clean.open():
            rec = json.loads(line)
            if rec["id"] not in keep:
                continue
            _r, visible, is_refusal = gate(rec, answers[rec["id"]], keep_cot=keep_cot)
            system = DEFAULT_SYSTEM_PROMPT.format(context=rec["context"])
            if keep_cot:
                system = system.replace("\n<retrieved_context>", COT_RULE + "\n<retrieved_context>")
            split = rec.get("split", "train")
            files[split].write(json.dumps({
                "id": rec["id"],
                "conversations": [
                    {"from": "system", "value": system},
                    {"from": "human", "value": rec["question"]},
                    {"from": "gpt", "value": visible},
                ],
            }) + "\n")
            st[f"{split}_refusal" if is_refusal else split] += 1
    finally:
        for f in files.values():
            f.close()
    if not st["eval"] and not st["eval_refusal"]:
        paths["eval"].unlink(missing_ok=True)

    kept = st["train"] + st["train_refusal"]
    if kept:
        st["train_refusal_pct"] = round(100 * st["train_refusal"] / kept)
    return st


def _self_check() -> None:
    srcs = [{"section_id": "5.2.1", "section_title": "T", "figure_number": None, "content_type": "prose"}]
    rec = {"id": "a", "question": "q", "context": "[Section 5.2.1] T\nbody", "sources": srcs}
    assert not any(c.get("hallucinated") for c in _extract_citations("x [§5.2.1]", srcs))
    assert any(c.get("hallucinated") for c in _extract_citations("x [§9.9.9]", srcs)), \
        "gate would pass a tag absent from the context"
    assert REFUSAL_RE.search("The provided context does not specify the encoding.")
    assert strip_cot("<thinking>[§9.9.9]</thinking>\nAns [§5.2.1]") == "Ans [§5.2.1]"

    figs = [{"section_id": "", "section_title": "T", "figure_number": "564", "content_type": "table"}]
    assert figure_tags("A [Figure 564]", figs) == (1, 0), "valid figure tag must count as grounded"
    assert figure_tags("A [Figure 999]", figs) == (0, 1), "absent figure tag must be caught"
    assert not _extract_citations("A [Figure 564]", figs), \
        "section parser started matching figures; drop the figure_tags workaround"
    print("self-check: ok", rec["id"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clean", nargs="?")
    ap.add_argument("results", nargs="?")
    ap.add_argument("-o", "--out")
    ap.add_argument("--keep-cot", action="store_true")
    ap.add_argument("--max-refusal-frac", type=float, default=0.40,
                    help="cap the refusal share of train (0 disables)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()

    if a.self_check:
        _self_check()
        raise SystemExit
    if not (a.clean and a.results and a.out):
        ap.error("clean, results and -o are required")
    for k, v in sorted(build(Path(a.clean), Path(a.results), Path(a.out), keep_cot=a.keep_cot,
                             max_refusal_frac=a.max_refusal_frac, seed=a.seed).items()):
        print(f"{k}: {v}")
