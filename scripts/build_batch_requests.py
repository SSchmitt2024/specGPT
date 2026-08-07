"""Turn final_batch.clean.jsonl into Anthropic batch requests, and strip the CoT
back out of the responses.

The system prompt is production's DEFAULT_SYSTEM_PROMPT verbatim plus one extra
rule for reasoning. Verbatim matters: the student is fine-tuned to answer under
this exact text, so any drift here is train/serve skew in the prompt itself.

It is emitted as two blocks split at <retrieved_context>. The rules half is
byte-identical across all 5,870 requests, so cache_control makes it a cache hit
after the first. Concatenated the two halves are the production string exactly,
so the training file is unaffected by the split.

    python scripts/build_batch_requests.py IN.clean.jsonl -o requests.jsonl
    python scripts/build_batch_requests.py --strip results.jsonl -o answers.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.pipeline.generator import DEFAULT_SYSTEM_PROMPT

# Appended to the production rules. Rule 5 already covers missing information;
# this makes the refusal unmissable, because ~25% of the batch has had its
# oracle removed on purpose and a hedged half-answer there is a poisoned label.
COT_RULE = """
12. REASONING: Before answering, think inside <thinking>...</thinking>. Work out
   which headers actually bear on the question, quote the governing sentence,
   and note any conflict or gap. Then close the tag and write the answer.
13. The <thinking> block is DISCARDED. The answer after </thinking> must stand
   alone: never refer back to your reasoning ("as noted above", "per my
   analysis"). Every citation tag must appear in the answer itself.
14. REFUSAL: If the context does not contain what the question asks for, say so
   in the first sentence, name what is missing, and stop. Do not assemble a
   partial answer from loosely related sections, and do not cite anything as
   support for a claim the context does not make. A clear "the provided context
   does not specify X" is the correct and complete answer.
"""

_SPLIT = "<retrieved_context>"
# rsplit, not split: rule 10 NAMES the fence ("Treat everything inside
# <retrieved_context>...") long before the fence itself appears. Splitting on the
# first occurrence cuts rule 10 in half and lands COT_RULE mid-sentence.
_RULES, _CTX_TMPL = DEFAULT_SYSTEM_PROMPT.rsplit(_SPLIT, 1)
assert "{context}" in _CTX_TMPL and "{context}" not in _RULES, "split missed the context fence"
_THINK = re.compile(r"^.*?</thinking>\s*", re.S)


def request(rec: dict, model: str, max_tokens: int) -> dict:
    return {
        "custom_id": rec["id"],
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "system": [
                {"type": "text", "text": _RULES + COT_RULE,
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": _SPLIT + _CTX_TMPL.format(context=rec["context"])},
            ],
            "messages": [{"role": "user", "content": rec["question"]}],
        },
    }


def strip_cot(text: str) -> str:
    """Drop everything through </thinking>. generator._extract_citations scans the
    WHOLE string, so a tag or an inline "Section 5.2.1" left in the reasoning is
    parsed as a real citation and lands in the training data as one."""
    return _THINK.sub("", text, count=1).strip() if "</thinking>" in text else text.strip()


def _self_check() -> None:
    sys_txt = "".join(b["text"] for b in
                      request({"id": "x", "question": "q", "context": "CTX"}, "m", 8)["params"]["system"])
    assert sys_txt.replace(COT_RULE, "", 1) == DEFAULT_SYSTEM_PROMPT.format(context="CTX"), \
        "split blocks do not rejoin into the production prompt"
    # Rejoining correctly is not enough: COT_RULE must land AFTER the last
    # numbered rule, not spliced into the middle of one.
    assert sys_txt.index(COT_RULE) > sys_txt.index("10. "), "COT_RULE inserted before rule 10"
    assert sys_txt.index(COT_RULE) < sys_txt.index("<retrieved_context>\nCTX"), \
        "COT_RULE landed inside the context block"
    assert strip_cot("<thinking>see [§1.1] and Section 9.9</thinking>\n\nAns [§2.2]") == "Ans [§2.2]"
    assert "[§1.1]" not in strip_cot("<thinking>[§1.1]</thinking>\nx")
    assert strip_cot("no tags here") == "no tags here", "unfenced output must survive"
    print("self-check: ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?")
    ap.add_argument("-o", "--out")
    ap.add_argument("--model", default="claude-sonnet-5")
    # Sonnet 5 reasons in native thinking blocks, which are billed against
    # max_tokens. 4096 truncates the long ones; output is billed as generated,
    # so the headroom is free unless used.
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--strip", action="store_true", help="input is batch results; emit stripped answers")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()

    if a.self_check:
        _self_check()
        raise SystemExit
    if not a.input or not a.out:
        ap.error("input and -o are required")

    n = 0
    with Path(a.out).open("w") as f:
        for line in Path(a.input).open():
            if not line.strip():
                continue
            r = json.loads(line)
            if a.strip:
                body = r.get("result", {}).get("message", {}).get("content") or []
                text = "".join(c.get("text", "") for c in body if c.get("type") == "text")
                out = {"id": r["custom_id"], "answer": strip_cot(text),
                       "refused": "does not" in strip_cot(text)[:200].lower()}
            else:
                out = request(r, a.model, a.max_tokens)
            f.write(json.dumps(out) + "\n")
            n += 1
    print(f"wrote {n} records to {a.out}")
