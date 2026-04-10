# specGPT

A web app for intelligent querying of the public NVMe specification. Type a question, get a cited answer.

## What it is

An independent Q&A system over the NVMe spec (800+ pages, heavily cross-referential). Custom spec parser, self-built knowledge graph, hybrid retrieval, LLM generation with inline section citations.

Built on publicly available NVMe specifications from nvmexpress.org. No proprietary data.

## How it works

1. **Parse** — tables, prose, cross-references, and metadata extracted from the spec PDF
2. **Graph** — entities and relationships stored in a NetworkX knowledge graph
3. **Index** — definition-enriched chunks embedded into Supabase (pgvector + BM25)
4. **Retrieve** — hybrid search: BM25 + vector + graph expansion + cross-encoder rerank
5. **Generate** — Claude Sonnet answers with strict grounding and section citations

## Stack

Python · NetworkX · Supabase (pgvector) · Claude (Sonnet + Haiku) · FastAPI · Docker

## Status

Pre-prototype. Building toward a 1-month milestone: live website with cited answers and a documented eval score.

See [BUILD_PLAN_FINAL.md](BUILD_PLAN_FINAL.md) for phase-by-phase detail.

## Principle

AI-driven, not vibecoded. A project to learn from, not just direct.
