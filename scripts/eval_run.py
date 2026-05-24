"""
Phase 2 — Step 2.6: Eval Runner

Runs each item in eval_set.json through the live pipeline and scores the output.

Scoring per item:
  answer_present   — answer is non-empty and not a "no context" refusal
  citation_hit     — at least one citation.section_id matches expected_sections
  field_mentioned  — (lookup only) expected field name appears in answer text

Pass = answer_present AND (citation_hit OR field_mentioned for lookup)

Full run of 60 items takes ~15-20 min (Gemini 6.5s pacing + Anthropic calls).
Use --limit N for quick smoke tests.

Output: data/eval_results.json  (per-item detail)
        stdout                  (summary report)

Run:
  python -m src.pipeline.eval_run
  python -m src.pipeline.eval_run --limit 10          # first 10 items only
  python -m src.pipeline.eval_run --types lookup      # one type only
  python -m src.pipeline.eval_run --eval-set path/to/eval_set.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.orchestrator import orchestrate, PipelineConfig


DEFAULT_EVAL_SET = Path("data/eval_set.json")
DEFAULT_OUTPUT = Path("data/eval_results.json")

# Phrases that indicate the pipeline had no relevant context
_NO_CONTEXT_PHRASES = [
    "not in the context",
    "not provided in the context",
    "context does not contain",
    "does not contain the answer",
    "information is not available",
    "cannot find",
    "no information",
]


# ---------------------------------------------------------------------------
# Scoring

def _answer_present(answer: str) -> bool:
    if not answer or len(answer.strip()) < 20:
        return False
    lower = answer.lower()
    return not any(phrase in lower for phrase in _NO_CONTEXT_PHRASES)


def _citation_hit(citations: list[dict], expected_sections: list[str]) -> bool:
    if not expected_sections:
        return False
    cited_ids = {c.get("section_id", "") for c in citations}
    for expected in expected_sections:
        for cited in cited_ids:
            # exact match or cited is a child section (e.g. "3.1.3.1" starts with "3.1.3")
            if cited == expected or cited.startswith(expected + "."):
                return True
    return False


def _field_mentioned(answer: str, expected_fields: list[str]) -> bool:
    if not expected_fields:
        return False
    lower = answer.lower()
    return any(f.lower() in lower for f in expected_fields)


def score_item(item: dict, result: dict) -> dict:
    answer = result.get("answer", "")
    citations = result.get("citations", [])

    ap = _answer_present(answer)
    ch = _citation_hit(citations, item.get("expected_sections", []))
    fm = _field_mentioned(answer, item.get("expected_fields", []))

    qtype = item.get("type", "unknown")
    if qtype == "lookup":
        passed = ap and (ch or fm)
    else:
        passed = ap and ch

    return {
        "answer_present": ap,
        "citation_hit": ch,
        "field_mentioned": fm,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Report

def _print_report(results: list[dict]) -> None:
    types = sorted({r["type"] for r in results})

    print("\n" + "=" * 60)
    print("EVAL RESULTS")
    print("=" * 60)

    overall_pass = 0

    for qtype in types:
        group = [r for r in results if r["type"] == qtype]
        if not group:
            continue

        n = len(group)
        passed = sum(1 for r in group if r["scores"]["passed"])
        ap = sum(1 for r in group if r["scores"]["answer_present"])
        ch = sum(1 for r in group if r["scores"]["citation_hit"])
        fm = sum(1 for r in group if r["scores"]["field_mentioned"])
        errors = sum(1 for r in group if r.get("error"))

        overall_pass += passed

        print(f"\n{qtype.upper()} ({n} items)")
        print(f"  Pass rate       : {passed}/{n}  ({100*passed/n:.0f}%)")
        print(f"  Answer present  : {ap}/{n}")
        print(f"  Citation hit    : {ch}/{n}")
        if qtype == "lookup":
            print(f"  Field mentioned : {fm}/{n}")
        if errors:
            print(f"  Errors          : {errors}")

    n_total = len(results)
    errors_total = sum(1 for r in results if r.get("error"))
    print(f"\nOVERALL: {overall_pass}/{n_total} passed ({100*overall_pass/n_total:.0f}%)")
    if errors_total:
        print(f"  ({errors_total} items errored — see eval_results.json for details)")

    avg_latency = sum(r.get("latency_ms", 0) for r in results) / max(n_total, 1)
    print(f"  Avg latency: {avg_latency/1000:.1f}s per query")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Runner

def run_eval(
    eval_set_path: Path = DEFAULT_EVAL_SET,
    output_path: Path = DEFAULT_OUTPUT,
    limit: int | None = None,
    types: list[str] | None = None,
    debug: bool = False,
) -> list[dict]:
    if not eval_set_path.exists():
        print(f"Error: {eval_set_path} not found. Run eval_gen.py first.", file=sys.stderr)
        sys.exit(1)

    with open(eval_set_path, encoding="utf-8") as f:
        eval_set: list[dict] = json.load(f)

    if types:
        eval_set = [item for item in eval_set if item.get("type") in types]

    if limit:
        eval_set = eval_set[:limit]

    print(f"Running {len(eval_set)} eval items...")
    config = PipelineConfig()

    results: list[dict] = []
    for i, item in enumerate(eval_set, 1):
        query = item["query"]
        qtype = item.get("type", "?")
        print(f"[{i}/{len(eval_set)}] ({qtype}) {query[:70]}...")

        start = time.time()
        error: str | None = None
        pipeline_result: dict = {}

        try:
            pipeline_result = orchestrate(query, config=config, debug=debug)
        except Exception as e:
            error = str(e)
            print(f"  ERROR: {e}", file=sys.stderr)

        latency_ms = (time.time() - start) * 1000
        scores = score_item(item, pipeline_result) if not error else {
            "answer_present": False,
            "citation_hit": False,
            "field_mentioned": False,
            "passed": False,
        }

        if scores["passed"]:
            print(f"  PASS — cited: {[c.get('section_id') for c in pipeline_result.get('citations', [])]}")
        else:
            flags = []
            if not scores["answer_present"]:
                flags.append("no answer")
            if not scores["citation_hit"] and item.get("expected_sections"):
                flags.append(f"missed sections {item['expected_sections']}")
            if qtype == "lookup" and not scores["field_mentioned"]:
                flags.append(f"field not mentioned: {item.get('expected_fields')}")
            print(f"  FAIL — {', '.join(flags) or 'unknown'}")

        results.append({
            "id": item["id"],
            "query": query,
            "type": qtype,
            "expected_sections": item.get("expected_sections", []),
            "expected_fields": item.get("expected_fields", []),
            "answer": pipeline_result.get("answer", ""),
            "citations": pipeline_result.get("citations", []),
            "scores": scores,
            "latency_ms": round(latency_ms),
            "error": error,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    _print_report(results)
    print(f"\nDetailed results saved to {output_path}")
    return results


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run NVMe spec eval set through live pipeline.")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, help="Run only the first N items")
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["lookup", "structural", "relational", "procedural"],
        help="Filter to specific question types",
    )
    parser.add_argument("--debug", action="store_true", help="Include pipeline trace in output")
    args = parser.parse_args(argv)

    run_eval(
        eval_set_path=args.eval_set,
        output_path=args.output,
        limit=args.limit,
        types=args.types,
        debug=args.debug,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
