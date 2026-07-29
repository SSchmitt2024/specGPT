"""Build a RAFT/SFT fine-tuning dataset (ShareGPT JSONL, for Unsloth) from qa_log.

Reuses the PRODUCTION prompt/context assembly (generator.DEFAULT_SYSTEM_PROMPT +
generator.assemble_context) so the fine-tuned model is trained on the exact
same system prompt / chunk-header / citation-tag format the live app sends and
parses. Context per example = the chunks actually cited (oracle) + random
same-spec distractor chunks (RAFT), shuffled so oracle position isn't fixed.

Usage:
    source venv/bin/activate
    python -m src.pipeline.build_raft_dataset [--out data/raft_finetune/gemma_raft_sft.jsonl]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.pipeline.generator import DEFAULT_SYSTEM_PROMPT, _extract_citations, assemble_context
from src.pipeline.orchestrator import _agentic_gap_analysis
from src.pipeline.search import supabase_client

CONCRETE_SPECS = ("base", "command", "pcie")

REFUSAL_RE = re.compile(
    r"does not (contain|specify|address|provide)|"
    r"not (specified|addressed|found|available|covered) in (the )?(provided )?"
    r"(spec|context)|no information (is )?(available|provided)|"
    r"context does not|i (don't|do not) have (enough|sufficient) "
    r"(information|context)|cannot find (this|that|the) information",
    re.IGNORECASE,
)

MAX_CONTEXT_TOKENS = 6000
DISTRACTOR_RANGE = (2, 4)


def _fetch_all(table: str, cols: str) -> list[dict]:
    """Page through a table (supabase default caps a single select at 1000)."""
    sb = supabase_client()
    out, start, page = [], 0, 1000
    while True:
        res = sb.table(table).select(cols).range(start, start + page - 1).execute()
        rows = res.data or []
        out.extend(rows)
        if len(rows) < page:
            break
        start += page
    return out


def _load_pools():
    chunks = _fetch_all(
        "spec_chunks",
        "id, spec, section_id, section_title, content_type, chunk_index, figure_number, pdf_pages, text_raw",
    )
    by_section: dict[tuple, list[dict]] = defaultdict(list)
    by_spec: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_section[(c["spec"], c["section_id"])].append(c)
        by_spec[c["spec"]].append(c)
    for key in by_section:
        by_section[key].sort(key=lambda c: c.get("chunk_index") or 0)

    fields = _fetch_all("spec_fields", "spec, name, description, section_id, figure_number")
    fields_by_section: dict[tuple, list[dict]] = defaultdict(list)
    for f in fields:
        if f.get("section_id"):
            fields_by_section[(f["spec"], f["section_id"])].append(f)

    enums = _fetch_all("spec_enum_index", "spec, concept, value_hex, name, label, sections")
    enums_by_section: dict[tuple, list[dict]] = defaultdict(list)
    for e in enums:
        for sid in e.get("sections") or []:
            enums_by_section[(e["spec"], sid)].append(e)

    return by_section, by_spec, fields_by_section, enums_by_section


def _resolve_oracle(citation: dict, row_spec: str, by_section, fields_by_section, enums_by_section) -> dict | None:
    """Turn a stored citation into a chunk dict assemble_context understands."""
    spec = citation.get("spec") or row_spec
    if spec not in CONCRETE_SPECS:
        spec = CONCRETE_SPECS[0]
    section_id = citation.get("section_id")
    if not section_id:
        return None

    matches = by_section.get((spec, section_id))
    if matches:
        text = "\n".join(m.get("text_raw") or "" for m in matches).strip()
        m0 = matches[0]
        return {
            "id": m0["id"], "spec": spec, "section_id": section_id,
            "section_title": m0.get("section_title") or citation.get("section_title") or "",
            "content_type": m0.get("content_type") or citation.get("content_type") or "prose",
            "figure_number": m0.get("figure_number"), "pdf_pages": m0.get("pdf_pages") or [],
            "text_raw": text,
        }

    # Fallback: structured lookup tables (fields / enums), not indexed spec_chunks.
    fmatches = fields_by_section.get((spec, section_id))
    if fmatches:
        text = "\n".join(f"{f['name']}: {f.get('description') or ''}" for f in fmatches)
        return {
            "id": f"field:{spec}:{section_id}", "spec": spec, "section_id": section_id,
            "section_title": citation.get("section_title") or fmatches[0]["name"],
            "content_type": "prose", "figure_number": fmatches[0].get("figure_number"),
            "pdf_pages": [], "text_raw": text,
        }

    ematches = enums_by_section.get((spec, section_id))
    if ematches:
        text = "\n".join(
            f"{e['value_hex']} = {e['name']}" + (f" ({e['label']})" if e.get("label") else "")
            for e in ematches
        )
        return {
            "id": f"enum:{spec}:{section_id}", "spec": spec, "section_id": section_id,
            "section_title": citation.get("section_title") or ematches[0]["concept"],
            "content_type": "prose", "figure_number": None, "pdf_pages": [], "text_raw": text,
        }
    return None


def _sample_distractors(by_spec, specs_used: set[str], exclude_sections: set[str], n: int) -> list[dict]:
    pool_specs = list(specs_used) or [CONCRETE_SPECS[0]]
    candidates = [c for s in pool_specs for c in by_spec.get(s, [])
                  if c["section_id"] not in exclude_sections and (c.get("text_raw") or "").strip()]
    if not candidates:
        return []
    seen_sections, picked = set(), []
    random.shuffle(candidates)
    for c in candidates:
        if c["section_id"] in seen_sections:
            continue
        seen_sections.add(c["section_id"])
        picked.append(c)
        if len(picked) >= n:
            break
    return picked


def build_dataset(out_path: Path) -> Counter:
    qa_rows = _fetch_all(
        "qa_log", "request_id, query, answer, citations, config, spec, llm_model, agentic"
    )
    flagged = _fetch_all("flagged_answers", "query, answer")
    flagged_set = {(f["query"], f["answer"]) for f in flagged}

    by_section, by_spec, fields_by_section, enums_by_section = _load_pools()

    stats = Counter()
    records = []

    for row in qa_rows:
        query, answer = row.get("query") or "", row.get("answer") or ""
        if not query.strip() or not answer.strip():
            stats["empty"] += 1
            continue
        if (query, answer) in flagged_set:
            stats["flagged"] += 1
            continue

        citations = row.get("citations") or []
        good_citations = [c for c in citations if not c.get("hallucinated")]
        is_refusal = bool(REFUSAL_RE.search(answer))

        if not good_citations and not is_refusal:
            stats["no_valid_citations"] += 1
            continue

        oracle_chunks, oracle_sections, specs_used = [], set(), set()
        for c in good_citations:
            resolved = _resolve_oracle(c, row["spec"], by_section, fields_by_section, enums_by_section)
            if resolved and resolved["section_id"] not in oracle_sections:
                oracle_chunks.append(resolved)
                oracle_sections.add(resolved["section_id"])
                specs_used.add(resolved["spec"])

        if not oracle_chunks and not is_refusal:
            stats["unresolvable_citations"] += 1
            continue

        n_distractors = random.randint(*DISTRACTOR_RANGE)
        distractors = _sample_distractors(by_spec, specs_used, oracle_sections, n_distractors)
        combined = oracle_chunks + distractors
        if not combined:
            stats["no_context_at_all"] += 1
            continue
        random.shuffle(combined)

        context_text, used_chunks = assemble_context(
            query, combined, max_context_tokens=MAX_CONTEXT_TOKENS, figure_reserve_tokens=0
        )
        used_sections = {u["section_id"] for u in used_chunks}
        if oracle_sections and not oracle_sections.issubset(used_sections):
            stats["context_budget_dropped_oracle"] += 1
            continue

        # Ground-truth gate: re-run the SAME parser the app uses to grade a
        # live answer. Any citation the answer makes that doesn't resolve to
        # a header actually in `used_chunks` comes back hallucinated=True —
        # drop the whole example rather than train on an answer that asserts
        # a source it doesn't have.
        resolved_cites = _extract_citations(answer, used_chunks)
        if any(c.get("hallucinated") for c in resolved_cites):
            stats["hallucinated_tag_in_answer"] += 1
            continue
        if not is_refusal and not resolved_cites:
            stats["ungrounded_no_citations"] += 1
            continue

        # Content-groundedness gate: citation IDs can resolve to real headers
        # while the claims next to them aren't actually supported by that
        # chunk's content (real citation, fabricated substance). Re-run the
        # same classifier the live agentic pipeline uses to catch this —
        # drop the example rather than train the student on it.
        if not is_refusal:
            followups, _reason, requested, _call = _agentic_gap_analysis(
                query=query,
                answer=answer,
                used_chunks=used_chunks,
                citations=resolved_cites,
                max_followups=3,
            )
            if followups or any(requested.values()):
                stats["gap_flagged"] += 1
                continue

        system = DEFAULT_SYSTEM_PROMPT.format(context=context_text)
        records.append({
            "conversations": [
                {"from": "system", "value": system},
                {"from": "human", "value": query},
                {"from": "gpt", "value": answer},
            ]
        })
        stats["kept_refusal" if is_refusal and not oracle_chunks else "kept"] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    stats["total_written"] = len(records)
    stats["total_qa_log_rows"] = len(qa_rows)
    return stats


# Recover minimal chunk stubs from the rendered "[Section X] Title" /
# "[Figure N] Title" headers baked into the system prompt's context block, so
# the self-check can re-run generator._extract_citations (the real parser,
# same hierarchy/title fallback logic the app uses) without needing the
# original chunk dicts.
_HEADER_LINE_RE = re.compile(r"^\[(?:Section ([^\]]+)|Figure (\d+))\]\s*(.*)$", re.MULTILINE)


def check(out_path: Path) -> None:
    """Grounding self-check: re-parse each answer's citations with the SAME
    matcher generator.py uses at request time; any example whose answer
    yields a hallucinated citation against its own context is a bug."""
    n, bad = 0, 0
    with out_path.open() as f:
        for line in f:
            rec = json.loads(line)
            system = rec["conversations"][0]["value"]
            answer = rec["conversations"][2]["value"]
            stub_chunks = [
                {"section_id": sec, "section_title": title, "figure_number": fig,
                 "content_type": "table" if fig else "prose"}
                for sec, fig, title in _HEADER_LINE_RE.findall(system)
            ]
            n += 1
            cites = _extract_citations(answer, stub_chunks)
            if any(c.get("hallucinated") for c in cites):
                bad += 1
    print(f"self-check: {n} examples, {bad} with a hallucinated citation")
    assert bad == 0, f"{bad} examples cite a tag missing from their own context"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raft_finetune/gemma_raft_sft.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    out_path = Path(args.out)
    stats = build_dataset(out_path)
    for k, v in stats.items():
        print(f"{k}: {v}")
    check(out_path)
