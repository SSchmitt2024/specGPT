# Pipeline Audit — Post-Fix Re-Audit

Re-audit of `app.py` → `orchestrator.py` → `query_processor` →
`search/retriever/bm25_index` → `reranker` → `generator` after the
swarm-driven fix sweep. Line numbers refer to the current files.

Smoke-tested locally via `venv/bin/python3` against pure-Python helpers
(`table_serializer._normalize_row`, `generator._extract_citations` /
`_table_header_line_count` / `assemble_context`, `bm25_index.tokenize`,
`query_processor._heuristic_type` / `_normalize_llm_output`,
`retriever.rrf_merge`, `reranker.rerank` empty-query path,
`search._is_empty_query`). All assertions passed. All nine pipeline
modules import cleanly.

## Fixed in this pass

### Critical bugs (every request was failing)

| Was | Now |
|-----|-----|
| `orchestrator.py:252` read `decomp.notes` — `AttributeError` on every call | `orchestrator.py:252` reads `decomp.rationale` (matches `QueryDecomposition`) |
| `orchestrator.py:441–454` duplicated the `"generation"` stage outside the `try/except`, referencing undefined `context_text` | Deleted; the in-band trace entry inside the `try` is the single source of truth |
| `generator.py:234` ran `system_prompt.format(context_text, "")`, blanking the `USER QUESTION:` line via the second `{}` | `generator.py:357` formats only `{context}`; user query is sent as the user message |
| `generator.py:255` returned a 4-tuple while the signature claimed 2-tuple | `generator.py:315` signature is now `tuple[str, list[dict], list[dict], dict]` |
| `search.py:101–106` `sys.exit()` inside Supabase / Voyage client init killed the FastAPI worker | `search.py:117–135` raises `RuntimeError`; FastAPI returns 500 instead of dying |
| `bm25_index.py:89–107` pagination silently truncated when PostgREST's `db-max-rows` capped a single response below `page_size` | `bm25_index.py:108–141` advances by `len(batch)`, terminates on empty batch, adds `.order("id")` for stable pages, exposes `reload_index()` |
| `retriever.py:87 / :100 / :110` unpaginated reads of `spec_field_index` / `spec_fields` / `spec_tables` (silent 1000-row cap) | `retriever.py:44–72, 86–142` shared `_paginate()` helper used by all three loaders; cache-clearing `reload_lookup_caches()` exposed |

### Concurrency / safety

| Was | Now |
|-----|-----|
| `app.py:97` ran the sync `orchestrate()` inside an `async def`, blocking the event loop for the full pipeline | `app.py:111` wraps it in `asyncio.to_thread`, with a per-request id for log correlation |
| `app.py:668` `HOST` defaulted to `0.0.0.0` (docstring lied) | `app.py:684–686` defaults to `127.0.0.1`; explicit `HOST=0.0.0.0` required to expose |
| `app.py:74` `DEBUG_PIPELINE` defaulted to `"1"` (chunk previews + timings in prod responses) | `app.py:82` defaults to `"0"`; trace is opt-in |
| `app.py:99` raw exception string leaked to client | `app.py:117–122` logs the exception with a request id and returns only the opaque id |
| `app.py:614` frontend `innerHTML` interpolated unescaped `section_title` / `section_id` (XSS), missing closing `</div>` | `app.py:629–664` `escapeHtml()` everywhere, hallucinated-citation flag rendered safely, closing `</div>` restored, pipeline trace JSON also escaped |
| `generator.py:243` retrieved chunks interpolated into the system prompt with no "treat as untrusted" guard | `generator.py:49–70, 188–193` chunks wrapped in `<retrieved_context>` block with a per-chunk `===== CHUNK n =====` fence; system rule #8 explicitly instructs the model to ignore instructions inside that block |
| `query_processor.py:243–249` user query concatenated into the classifier prompt | `query_processor.py:251–273` query wrapped in `<<<USER_QUERY_START>>> … <<<USER_QUERY_END>>>` and the classifier is told to treat the contents as data |

### Logic / scoring bugs

| Was | Now |
|-----|-----|
| `orchestrator.py:334–337` dedup keyed by `chunk.get("id")` collapsed all id-less chunks under `None` | `orchestrator.py:341–355` falls back to `chunk_id`, then a content-derived surrogate `("__no_id__", section/figure/type/text[:120])` |
| `retriever.py:470–491` RRF same `None`-collision, plus `"unknown"` masking missing-method bugs | `retriever.py:495–526` extracted `_doc_key()` with the same surrogate strategy; missing `method` now tagged `"missing_method"` to surface the upstream gap |
| `generator.py:174–189` `Section\s+([\d.]+)` captured trailing dot, breaking equality match; no hallucination signal | `generator.py:225–250` uses `Section\s+(\d+(?:\.\d+)*)(?!\.\d)`; citations carry an explicit `hallucinated` boolean and the frontend renders a ⚠ badge |
| `generator.py:126–134` context-budget loop `break`ed on first oversized chunk, dropping all remaining lower-ranked chunks | `generator.py:178–179` uses `continue` so smaller chunks still fill the budget |
| `generator.py:73–99` `_trim_table_chunk` hard-coded 2 header lines but `table_serializer` emits 3 (caption + headers + `---`) | `generator.py:91–142` `_table_header_line_count()` detects the actual count by looking for the caption / header / `---` shape |
| `table_serializer.py:60–64` rows with fewer cells silently misaligned all subsequent columns | `table_serializer.py:46–87` `_normalize_row()` pads short rows and joins overflow into the last cell; applied in both `serialize_table` and `make_table_chunks` |
| `bm25_index.py:113–114` title boost via token repetition inflated `|d|` and skewed `avgdl` across the whole corpus | `bm25_index.py:74–84, 122–144` body-only index, per-doc additive title bonus (`_TITLE_BOOST_PER_MATCH` × `|q ∩ title_tokens|`) applied at score time |
| `reranker.py:75–76` empty-query early return dropped `rerank_score` / `method` keys | `reranker.py:53–64, 84–87` `_ensure_shape()` always attaches `rerank_score` (None when skipped), `prior_method`, and `method="rerank"` |
| `reranker.py:78–80` no exception handling around `model.predict`; empty `text_field` chunks could outrank real hits | `reranker.py:104–117, 121–126` `try/except` falls back to prior order with `rerank_score=None`; empty-text rows scored `-inf` so they sort last |

### Gaps / silent failures

| Was | Now |
|-----|-----|
| `orchestrator.py:317` sub_queries replaced the original query in retrieval | `orchestrator.py:315–326` builds `search_queries = [query, …deduped sub_queries]` so RRF always sees the verbatim phrase |
| `orchestrator.py:264` `decomp.type == "lookup"` was case-sensitive exact match | `orchestrator.py:264` `(decomp.type or "").lower() == "lookup"` |
| `query_processor.py:265–278` no sub-query dedup or length sanity | `query_processor.py:281–301` collapses whitespace, drops `<3` or `>400` char strings, dedupes case-insensitively, caps at `max_subqueries` |
| `query_processor.py:301` `generate_json` had no retry | `query_processor.py:329–354` two attempts with exponential backoff; original exception re-raised on final failure |
| `query_processor.py:308–316` heuristic fallback labeled any entity-bearing query as `lookup` (relational/procedural quietly mis-routed on LLM failure) | `query_processor.py:386–429` `_heuristic_type()` matches relational / procedural / structural surface cues and uses entity-count as a relational signal |
| `search.py:173–188, 208–218` Voyage + Supabase calls had no try/except | `search.py:62–80, 207–249, 260–278, 290–296` `_retry()` helper wraps the network calls; failures degrade the path (return `[]`) rather than crashing the request |
| `search.py:170, 206, 235` empty-query guard only checked `.strip()` (punctuation-only queries still hit Voyage and tsquery) | `search.py:39–46, 209, 256, 287` `_is_empty_query()` requires `[A-Za-z0-9]` after stripping |
| `bm25_index.py:146–148` `get_index` was lru_cached with no escape hatch | `bm25_index.py:175–184` `reload_index()` clears and rebuilds |
| `retriever.py:83–114` lookup data loaders were lru_cached for process lifetime | `retriever.py:148–155` `reload_lookup_caches()` exposed |
| `generator.py:238–244` no retry / timeout / `stop_reason` handling | `generator.py:266–302, 359–381` retry on transient (429/5xx/timeout) failures, per-request timeout, `stop_reason="max_tokens"` surfaced in `tokens_used` and appended to the answer text |
| `generator.py:246` assumed first content block is text | `generator.py:253–263` `_extract_text()` concatenates every `type=="text"` block and ignores `tool_use`/empty content |

### Nits

| Was | Now |
|-----|-----|
| `app.py:36` unused `StaticFiles` import | Removed |
| `bm25_index.py:138–139` float `== 0` on scores | `bm25_index.py:55–58, 167–170` `_SCORE_EPS = 1e-9`; check is `abs(score) <= eps` |
| `bm25_index.py:55–59` stopwords misaligned with Postgres `english` config | `bm25_index.py:32–53` expanded to the Snowball English stoplist used by Postgres' `english` text-search config |
| `search.py:15–28` method-field docstring missing `rrf` / `structured_lookup` / `rerank` | `search.py:15–35` now lists every method tag added by downstream stages |

## Second-pass fixes (after the first re-audit)

| Was | Now |
|-----|-----|
| `orchestrator.py:411–448` synthesized `"Generation failed: ErrType: msg"` into the user-facing answer, leaking internals and suppressing 5xx | New `GenerationError(message, cause, trace, retrieved_chunks)` raised from the orchestrator; `app.py` catches it specifically and returns 502 with `{error, request_id, cause_type}` (plus the trace if debug is on). 500 still covers all other exceptions. |
| Generation model + token budgets were hardcoded inside `orchestrate()` | New `PipelineConfig` fields: `llm_model`, `llm_max_context_tokens`, `llm_max_output_tokens`. Surfaced through `/api/config`. |
| `assemble_context` could return empty `used_chunks` if every chunk exceeded budget, then send an empty `<retrieved_context>` block to the model → hallucinated answers | `generate()` raises `ValueError("All retrieved chunks exceeded max_context_tokens; …")` before the API call. |
| Total hybrid-search wall time wasn't traced (only per-sub-query timings) | New `hybrid_search.total` stage entry. |
| `retriever._paginate` `except Exception: pass` swallowed Supabase load failures silently | Now logs `"Supabase lookup load failed (…); falling back to local JSON snapshot"`. |
| `reranker` exception handler used `logger.error` (no traceback — painful for CUDA OOM / model-load diagnosis) | Now `logger.exception` (captures traceback). |
| `client._pace()` mutated module-level `_last_call_ts` without a lock; FastAPI's thread pool let concurrent requests race past the configured min-delay and burst-trip provider rate limits | `_pace_lock = threading.Lock()` plus reserve-then-sleep so the second thread paces against the first's *reservation*, not its previous timestamp. |
| Unused imports in `app.py` (`json`, `Any`, `JSONResponse`) and `orchestrator.py` (`Any`) | Removed. |
| Dead `struct_found` local in `orchestrator.py` | Removed. |
| No persisted regression tests | `tests/test_pipeline_units.py` — 26 unit tests covering every behavioral fix from this and the previous pass; runs via `pytest` or `venv/bin/python3 tests/test_pipeline_units.py`. |

## Still open (deferred — needs more context or a design call)

- **`query_processor.py:240–245`** — classifier tie-breaker `lookup > structural > relational > procedural` still biases toward the `lookup` path that skips decomposition. The heuristic fallback now sidesteps this on LLM failure, but the LLM itself is still nudged the same way. Changing the prompt order is a behavioral change worth A/B testing on a labelled query set before shipping.
- **`generator.py:117–142`** — `_trim_table_chunk` token-budgeted trim still keeps only contiguous leading rows. For a row-bit lookup where the relevant row is mid-table, the model loses the row. Needs query-aware row selection (probably wired through `_trim_table_rows` in retriever) — same logic the structured-lookup path already has.
- **`reranker.py:53–55`** — cross-encoder model load is still lazy + cached for process lifetime via `_load_model` (`lru_cache(maxsize=2)`). No invalidation, no warm-up at startup. Acceptable but worth noting if model swaps become common.
- **`retriever.py:477–480`** — `notes` only populated on miss paths; relational queries that surface a partial table still don't get a "confidence: medium because only one of three entities matched" signal.
- **`bm25_index.py` / `retriever.py`** — `reload_index()` and `reload_lookup_caches()` are exposed in-process but not wired to an admin endpoint or SIGHUP. After a re-embed, callers still need a restart.
- **`scripts/supabase_schema.sql`** — RPC bodies were not re-audited. If `search_spec_chunks_text` / `match_spec_chunks` themselves have ranking quirks (e.g. `ts_rank_cd` weight tuples, normalization flags), they're outside this pass.
- **`src/llm/client.py:296–304`** — JSON-mode parse only handles a trailing ``` fence; Gemini occasionally emits stray text before the JSON. Worth tightening with a `json.JSONDecoder.raw_decode` walk if we see real failures.

## New observations

- `generator._call_with_retry` retries any exception (not just Anthropic SDK errors); a malformed local pre-call exception will be retried 3× with backoff. Low-risk because the wrapping `try` inside `_call_with_retry` is specifically around `client.messages.create`, but worth tightening if we ever add pre-call validation that should fail fast.
- `assemble_context` now emits `===== CHUNK N =====` / `===== CHUNK END N =====` fences that count against the model's input tokens (~10 tokens per chunk). At `final_rerank_topk=7` that's ~70 tokens — comfortable inside the 4000-token context budget.
- `bm25_index._fetch_corpus` and `retriever._paginate` both hard-cap at 10M rows to guard against pathological loops. Above that they log a warning and stop — fine for the current ~1.9k-row corpus, revisit if the spec grows two orders of magnitude.
- The frontend's `escapeHtml()` is duplicated inline; if more user-derived strings start flowing into the trace UI, factor it out (and consider switching to `textContent` setters where layout allows).

## Suggested next fix order

1. Query-aware row selection in `_trim_table_chunk` so mid-table rows survive trimming (biggest open correctness gap).
2. Wire `bm25_index.reload_index()` and `retriever.reload_lookup_caches()` into an admin endpoint or signal handler so re-ingests don't need a process restart.
3. A/B the classifier tie-breaker (`lookup` vs. `structural`) on a labelled query set; whichever wins becomes the new default in `_SYSTEM_PROMPT`.

---

# Deploy-Readiness Evaluation

**Verdict: NO-GO for public production deploy. Conditional GO for private/internal preview.**

Code correctness is in good shape after the two audit passes (26/26 unit tests
green, every critical bug from the original audit closed, structured 502 on
generation failure, prompt-injection fences, XSS escape, async-safe pipeline,
thread-safe LLM pacing). What's still missing is the operational and
deployment-hygiene work — without it, the first attacker (or first scraper, or
first SaaS-credential-leaker) burns the project's Anthropic/Voyage budget.

## What is good ✅

- **Correctness**: 26 unit tests cover every behavioural fix; all 9 pipeline modules + `src/llm/client.py` import cleanly.
- **Web**: FastAPI app is async-safe (`asyncio.to_thread`), debug trace off by default, host defaults to `127.0.0.1`, XSS escaping on the inline frontend, hallucinated-citation badge.
- **Error model**: typed `GenerationError` separates retrieval failure from generation failure; `app.py` returns 502 vs. 500 accordingly with an opaque `request_id`.
- **Resilience**: retry/backoff on Voyage embed, Supabase RPC, Anthropic call, and Gemini/OpenAI classifier; degrades-not-crashes on transient hybrid-path failures.
- **Containerization basics**: `Dockerfile` exists, pre-downloads cross-encoder weights (no first-request stall), Railway config (`railway.toml`) with a healthcheck against `/api/config`.
- **No secrets tracked**: `.env` is in `.gitignore`, `git ls-files` confirms nothing sensitive committed.
- **Schema**: `scripts/supabase_schema.sql` is idempotent (`IF NOT EXISTS` / `OR REPLACE`), tsvector + pgvector + btree indexes covered, lookup tables defined.

## Deploy-blocking issues 🚫

| # | Issue | Where | Fix |
|---|-------|-------|-----|
| 1 | **No auth + no rate limit on `/api/query`** — every public request bills against your Anthropic + Voyage account. The single biggest cost-blast-radius risk for a publicly reachable deploy. | `app.py:88` | Add an API-key header check (read from env), `slowapi` or `fastapi-limiter` rate limit per IP/key. |
| 2 | **Unpinned production deps** — `voyageai`, `supabase`, `sentence-transformers`, `rank_bm25`, `anthropic`, `google-genai`, `openai`, `fastapi`, `uvicorn`, `pydantic`, `psycopg2-binary` have no version pins. Two builds a week apart can ship different code paths. | `requirements.txt:10–20` | Pin everything, ideally via `pip-compile` / `uv pip compile` into a `requirements.lock` checked in. |
| 3 | **No request-size limit on `/api/query`** — a 10 KB query is fine, a 10 MB query embeds + ships to Anthropic and costs $$$. Pydantic has no max length on `QueryRequest.query`. | `app.py:52–56` | `query: Annotated[str, StringConstraints(min_length=1, max_length=2000)]` or equivalent validator. |
| 4 | **Required env vars are not enumerated anywhere** — Dockerfile doesn't `ENV` them, there is no `.env.example`, README doesn't list them. Deploy comes up healthy (`/api/config` doesn't need them), then 500s on the first real request. | repo root | Add `.env.example` listing `SUPABASE_URL`, `SUPABASE_KEY`, `VOYAGE_API_KEY`, `ANTHROPIC_API_KEY`, and one of `GEMINI_API_KEY` / `OPENAI_API_KEY` (plus optional `LLM_PROVIDER`, `LLM_MODEL`). Make the healthcheck actually probe the backend (a `/api/health` that hits Supabase + Voyage + Anthropic with cheap no-ops). |
| 5 | **`docs/RUNNING_THE_APP.md` is stale on critical defaults** — claims `DEBUG_PIPELINE` defaults to `"1"` (it's now `"0"`), claims output shows "Debug Mode: True" (no longer true), doesn't mention `ANTHROPIC_API_KEY` at all. Will mislead anyone setting up the deploy. | `docs/RUNNING_THE_APP.md:121–137` | Sync with current defaults + add the missing API keys. |
| 6 | **Frontend contract drift** — `frontend/src/api.js` documents `citations: ["3.1.2", ...]` (array of strings) and a `confidence` field. Backend now returns `citations: [{section_id, section_title, content_type, hallucinated}]` and no `confidence`. If the Vite/React app is deployed, it breaks. | `frontend/src/api.js:22–37` | Either delete `frontend/` (the embedded HTML in `app.py` is what's actually served) or update the React app and add a CORS middleware if it'll run on a different origin. |
| 7 | **No CI** — there's no `.github/workflows`, no `pre-commit`, nothing that runs `tests/test_pipeline_units.py` before merge. Easy to regress one of the 26 covered behaviors. | repo | Add a workflow that runs `python tests/test_pipeline_units.py` on push + PR. |

## Should-fix-before-public ⚠️

| Issue | Where | Why |
|-------|-------|-----|
| **No top-level timeout on `orchestrate()`** — only the Anthropic call is bounded (60 s). A slow Voyage/Supabase combo can hang the request indefinitely. | `app.py:113` | Wrap the `asyncio.to_thread` call in `asyncio.wait_for(...)`. |
| **Dockerfile runs as root**, no `HEALTHCHECK` directive, no multi-stage build (image is ~1 GB after model pre-download + torch wheels). | `Dockerfile` | `USER app` + `HEALTHCHECK CMD curl -fsS http://127.0.0.1:$PORT/api/config`. Multi-stage to strip pip cache. |
| **No CORS middleware** — if the React frontend ever deploys to a separate origin, every request 4xxs. | `app.py:73–77` | `CORSMiddleware` with explicit allowlist. |
| **`ruvector.db` (1.5 MB) lives in repo root, untracked, no `.gitignore` entry** — easy to commit by accident. `Backups/` also unlisted in `.gitignore`. | `.gitignore` | Add `ruvector.db`, `Backups/`. |
| **No structured logging** — `logging.basicConfig(level=...)` only. Production needs JSON logs for ingest. | `app.py:44–45` | `python-json-logger` or `structlog`. |
| **No metrics / tracing** — no Prometheus exporter, no OTel spans. Hard to diagnose latency without per-stage traces beyond what's in `pipeline_trace`. | — | `prometheus-fastapi-instrumentator` is a one-liner. |
| **`bm25_index.get_index()` loads the entire ~1.9 k row corpus into memory on first call.** That's fine, but it happens at first-request time, not startup, so cold-start latency is high. | `bm25_index.py:199–201` | Call `bm25_index.get_index()` at FastAPI startup (`@app.on_event("startup")` or lifespan handler) so the corpus warms before traffic. Same for cross-encoder. |
| **`Anthropic()` reads `ANTHROPIC_API_KEY` implicitly** — no early validation. If the var is missing, the first request 502s instead of failing fast at boot. | `generator.py:360` | Validate at startup. |
| **Generic `except Exception:` on the embedded `runQuery()` frontend** logs the error message to a red banner. Fine for dev, leaks 500 detail (`request_id={...}`) to the user in prod. Today acceptable because `app.py` only sends `{"error", "request_id"}`. | `app.py:599–625` | Just be aware; no change needed unless we add detailed error envelopes. |

## Nice-to-have (post-launch) 💡

- `/api/health` that actually exercises Supabase + Voyage + Anthropic vs. the current `/api/config` which only returns static config.
- Admin endpoint hitting `bm25_index.reload_index()` + `retriever.reload_lookup_caches()` so a re-ingest doesn't need a redeploy.
- Cost meter: log `tokens.prompt + tokens.completion` per request, expose as Prometheus counter.
- Query-aware mid-table row trimming in `generator._trim_table_chunk` (still open from the audit).

## Repo / supabase hygiene 🧹

- **Supabase tables have no RLS configured** — for a read-only public-spec app this is acceptable, but make sure the key in `.env` is `anon` (not `service_role`); `service_role` in a public-facing FastAPI is a major risk.
- **`docs/IOL_NVMe_Benchwork_Questions.md`** (33 KB) is in the repo. Make sure that's intentional — IOL benchwork material is usually licensed/embargoed.
- **`Backups/`** directory present. Should be `.gitignore`d.

## Go / no-go matrix

| Audience | Verdict | Conditions |
|----------|---------|------------|
| **Internal demo / colleague preview** | ✅ GO | Lock the URL behind a tunnel/VPN or a single static header token. Set `DEBUG_PIPELINE=1` so you can see the trace. |
| **Friends-and-family beta with a known userlist** | ⚠️ CONDITIONAL | Fix blockers 1, 3, 4 first (add an API key check, request-size cap, `.env.example`). Pin dependencies (#2). |
| **Public production deploy** | 🚫 NO-GO | All of blockers 1–7 must close. Plus: top-level timeout, CORS if FE is split, CI, `Anthropic` startup validation, request-cost logging, image hardening. |

