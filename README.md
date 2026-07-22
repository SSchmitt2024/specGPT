# specGPT

A web app for intelligent querying of the public NVMe specification. Type a question, get a cited answer.

## What it is

An independent Q&A system over the NVMe spec (800+ pages, heavily cross-referential). Custom spec parser, self-built knowledge graph, hybrid retrieval, LLM generation with inline section citations.

Built on publicly available NVMe specifications from nvmexpress.org.

## How it works

1. **Parse** — tables, prose, cross-references, definitions, TOC, enriched summaries, and metadata extracted from the spec PDF
2. **Index** — definition-enriched chunks embedded into Supabase (pgvector + BM25 + tsvector), parsed tables, and parsed fields
3. **Retrieve** — first querey decomposition occurs then, hybrid search: BM25 + vector + graph expansion + cross-encoder rerank as well as RRF
4. **Generate** — A selected LLM will answer prompts with strict grounding and section citations
5. **Agentic** - A configurable agent loop that can: perfom gap analyis, request follow up data, and can recurse into another cycle.

## Stack

Python · Supabase · Multiple LLM API · FastAPI · Docker 

See [docs/BUILD_PLAN_FINAL.md](docs/BUILD_PLAN_FINAL.md) for phase-by-phase detail.


