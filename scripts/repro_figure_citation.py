"""Repro / eval harness for the "figure referenced but never cited" bug.

The runtime pipeline appends figure-expansion chunks to the TAIL of the
context list (orchestrator: `retrieved_chunks + fig_expansion`) and then hands
the list to `generator.assemble_context`, which fills a token budget in order
and skips anything that no longer fits. When the ranked prose nearly fills the
budget, the tail figures are dropped -> the model never sees the table -> it
cannot emit a `[Figure N]` citation even though it references the figure in
prose.

This harness reproduces that boundary deterministically, with REAL corpus
chunks and REAL token sizes, and NO LLM/embedding calls:

  prose cluster (mimics final_rerank_topk hits)  +  _expand_referenced_figures
        -> assemble_context(budget)  ->  did the referenced figures survive?

Metric: figure survival rate = (figures present in used_chunks) / (figures the
expansion appended). On buggy `main` this is ~0% at the first-pass budget; the
fix should drive it to ~100% without dropping prose grounding.

Usage:
    python -m scripts.repro_figure_citation                 # first-pass budget (4000)
    python -m scripts.repro_figure_citation --budget 4000
"""
from __future__ import annotations

import argparse
import sys

from src.pipeline import generator
from src.pipeline.orchestrator import _expand_referenced_figures
from src.pipeline.search import supabase_client

# (name, query, parent_section) — clusters whose prose references figures and
# carry enough prose to fill the first-pass budget. Discovered from the corpus;
# the first is the exact PKAS family users flagged.
DATASET: list[tuple[str, str, str]] = [
    ("pkas_unfreeze", "how to unfreeze a personality using pkas step by step", "8.1.6"),
    ("rotational_media", "explain the rotational media operations and their data structures", "5.2.13"),
    ("sanitize_ops", "how does the sanitize operation work and what fields does it use", "8.1.9"),
    ("directives", "describe the directive operations and their data frames", "8.1.18"),
    ("fdp_events", "how are flexible data placement events structured", "8.1.26"),
]

PROSE_TOPK = 10          # mimic final_rerank_topk (orchestrator default ~10)
FIG_CAP = 6              # PipelineConfig.figure_ref_expansion_cap default
SPEC = "base"


def _fig_key(c: dict) -> str | None:
    f = c.get("figure_number")
    if f is None:
        return None
    return str(f).strip().lstrip("0") or "0"


def _fetch_prose_cluster(parent: str, topk: int) -> list[dict]:
    """Real prose chunks for a section subtree, document order, capped at topk.

    Stands in for the retrieval+rerank output: a topic query lands on the
    section cluster, and the cross-encoder keeps the top ~10 prose chunks.
    """
    sb = supabase_client()
    cols = "id, section_id, section_title, content_type, text_raw, figure_number, pdf_pages, spec"
    rows = (
        sb.table("spec_chunks").select(cols)
        .eq("spec", SPEC).eq("content_type", "prose")
        .like("section_id", f"{parent}.%")
        .order("section_id").limit(200).execute()
    ).data or []
    # Prefer prose that actually references figures (that's what a real answer
    # for these queries leans on), then fill with the rest in document order.
    import re
    fig_re = re.compile(r"\bFigure\s+\d{2,4}\b")
    referencing = [r for r in rows if fig_re.search(r.get("text_raw") or "")]
    other = [r for r in rows if not fig_re.search(r.get("text_raw") or "")]
    picked = (referencing + other)[:topk]
    for r in picked:
        r.setdefault("method", "rerank")
    return picked


def run_scenario(name: str, query: str, parent: str, budget: int) -> dict:
    prose = _fetch_prose_cluster(parent, PROSE_TOPK)
    fig_expansion = _expand_referenced_figures(prose, spec=SPEC, cap=FIG_CAP)
    retrieved = prose + fig_expansion  # EXACT orchestrator ordering

    expected = [k for k in (_fig_key(c) for c in fig_expansion) if k]
    expected_set = set(expected)

    _ctx, used = generator.assemble_context(query, retrieved, max_context_tokens=budget)
    used_figs = {k for k in (_fig_key(c) for c in used) if k}
    survived = sorted(expected_set & used_figs, key=int)
    dropped = sorted(expected_set - used_figs, key=int)

    prose_used = sum(1 for c in used if not _fig_key(c))
    prose_total = len(prose)

    return {
        "name": name,
        "parent": parent,
        "prose_total": prose_total,
        "prose_used": prose_used,
        "expected_figs": sorted(expected_set, key=int),
        "survived": survived,
        "dropped": dropped,
        "n_expected": len(expected_set),
        "n_survived": len(survived),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=generator.DEFAULT_MAX_CONTEXT_TOKENS,
                    help="max_context_tokens for assemble_context (first pass default 4000)")
    args = ap.parse_args()

    print(f"figure-citation repro  |  budget={args.budget}  prose_topk={PROSE_TOPK}  fig_cap={FIG_CAP}\n")
    total_exp = total_surv = 0
    rows = []
    for name, query, parent in DATASET:
        r = run_scenario(name, query, parent, args.budget)
        rows.append(r)
        total_exp += r["n_expected"]
        total_surv += r["n_survived"]
        rate = (r["n_survived"] / r["n_expected"] * 100) if r["n_expected"] else 100.0
        print(f"[{name:<16}] prose {r['prose_used']}/{r['prose_total']} used | "
              f"figs expected={r['expected_figs']} survived={r['survived']} dropped={r['dropped']} "
              f"({rate:.0f}%)")

    overall = (total_surv / total_exp * 100) if total_exp else 100.0
    print("\n" + "=" * 72)
    print(f"OVERALL figure survival: {total_surv}/{total_exp} = {overall:.1f}%")
    print("=" * 72)
    # Non-zero exit if any expected figure was dropped (useful for iterate loop).
    return 0 if total_surv == total_exp else 1


if __name__ == "__main__":
    sys.exit(main())
