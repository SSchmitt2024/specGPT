# specGPT0: Chat Interface + IOL Test Plan Context Injection (rough plan)

## Why

specGPT today is one-shot Q&A: every `/api/query` call is fully stateless (no
session/conversation id anywhere in `qa_log`, `flagged_answers`, or
`PipelineConfig`; `/api/refine` only re-runs deeper agentic search on the
*same* query via an in-process `_REFINE_CACHE`, it does not carry
conversation history). Goal: evolve specGPT from "efficient retrieval
system" into a tool that works the bug with you, by:

1. **Multi-turn chat** — full history carried forward, token-budgeted, but
   any chunk actually cited in a prior turn gets pinned so it can't be
   evicted from context on later turns.
2. **IOL INTERACT test-plan picker** — IOL INTERACT test plans are public,
   structured docs (Step 1/2/3 + Observable Result 1/2/3 per test case).
   Download/parse them the same way the NVMe/PCIe specs are ingested
   (confirm no ToS issue first — public docs, but verify), store in
   Supabase, load at runtime like spec chunks. A UI popup lets you pick a
   test case + step/observable and inject that exact text as context, so
   the model is instantly caught up on what the customer-facing script
   checks instead of you re-explaining it by hand.

Work happens on branch **`specGPT0`, off `deepthought-vm`** (has the 8-model
menu this builds on). Chat mode is a **toggle** next to the existing
one-shot UI, not a replacement.

## Guardrails (apply to every phase)

- Stay token-budgeted like today's `assemble_context()`
  (`generator.py:312-450`, `max_context_tokens=4000` /
  `agentic_max_context_tokens=16000`). Pinned/cited chunks are a *reserved*
  bucket (same shape as existing `figure_reserve_tokens=3000`), not
  unbounded.
- Citation extraction depends on the exact fenced-chunk format
  (`_CHUNK_FENCE`, `generator.py:395-398`) carrying
  `section_id`/`spec`/`pdf_pages`. Injected test-plan text and prior-turn
  history are **not spec chunks** — fence them separately and explicitly
  exclude from citation parsing, so the model can't "cite" a test plan step
  as a spec section.
- `app.py` is 6454 lines, ~5150 of which is `FRONTEND_HTML`. New chat
  JS/CSS and the test-plan modal go in `src/pipeline/static/` (currently
  just `favicon.png`) as real `.js`/`.css` files, not more inline Python
  string.

## Phase 1 — IOL test plan ingestion

- Confirm ToS allows downloading/parsing IOL INTERACT test plans before
  building anything.
- New script `scripts/ingest_iol_testplans.py` (mirrors
  `scripts/load_lookup_data.py`/`indexer.py`): download → parse into
  `{test_id, title, category, steps: [{step_num, action_text}],
  observables: [{obs_num, text}]}` → `data/iol_testplans/` artifact →
  Supabase.
- New table `test_plans` in `scripts/supabase_schema.sql` (follow
  `spec_fields`/`spec_tables` conventions): `id, test_id, title, category,
  steps jsonb, observables jsonb, source_url, updated_at`.
- New read path in `search.py`: `fetch_test_plans()` /
  `fetch_test_plan(test_id)`, reusing `supabase_client()`.
- New endpoints: `GET /api/testplans` (list, for the picker),
  `GET /api/testplans/{test_id}` (steps/observables, for the popup).

## Phase 2 — Test plan picker + context injection

- New `src/pipeline/static/testplan-picker.js` (+ scoped CSS). Reuse the
  existing `preset-select` dropdown pattern (`app.py:2079`) for the
  test-plan dropdown, and the existing `.stage-popup` modal pattern
  (`app.py:1800-1822`, already used by field-definition popovers and the
  figure-render popup) for "pick step/observable → show text."
- Selected text goes on the query payload as its own field
  (`injected_context: {source: "iol_testplan", test_id, label, text}`), not
  concatenated into `query`, so the backend can fence it separately.
- `orchestrate()` gains an optional `injected_context` passthrough →
  `assemble_context()` gets a new reserved-token bucket (same shape as
  `figure_reserve_tokens`) wrapping the text in its own labeled fence
  (`[Test Plan Context — not a spec citation]`); `_extract_citations` skips
  fences of this kind.

## Phase 3 — Chat interface

- **Frontend**: replace the clear-and-replace `#answer-section`
  (`app.py:2221-2228`, `displayResults()` at `app.py:5733`) with an
  append-only message list, behind the toggle. New JS lives in
  `static/chat.js`. Existing streaming (`/api/query/stream` NDJSON) and
  citation-chip rendering (`app.py:5029-5119`) stay the same, just appended
  per turn instead of overwritten.
- **Backend**: lightweight `conversation_id` (client-generated UUID, no
  auth needed) mapping to turn history:
  `[{role, query, answer, citations, used_chunks}]`. Each new turn's prompt
  = (a) budgeted prior-turn summaries within the normal token budget, plus
  (b) a **pinned bucket** of chunk fences for any `used_chunks` cited in a
  *previous* turn (reserved-bucket pattern again, so pinned citations can't
  be silently evicted by new retrieval). If the pinned bucket alone would
  blow the budget, surface that back to the UI rather than silently
  truncating.
- New table `chat_sessions`, or simpler: extend `qa_log` with
  `conversation_id` + `turn_index` columns (cheaper, follows the existing
  fire-and-forget logging pattern) so history survives a page refresh.

## Security — rate limiting & spam protection

Still internal-use (session-cookie gate stays as the primary control), but
the new endpoints add cost/storage surface that the existing gate doesn't
throttle. Current state: `LoginThrottle` (`auth.py:186-211`) only covers
`/login` — per-IP exponential backoff on failed passwords. Nothing throttles
authenticated traffic once a session cookie is valid.

- **Per-session/IP request throttle on expensive endpoints.** `/api/query`,
  `/api/query/stream`, `/api/refine`, `/api/refine/stream` each trigger a
  full retrieval + LLM generation (real $ cost, latency). Add a simple
  in-memory token-bucket or sliding-window limiter (same shape as
  `LoginThrottle` — single-process/single-worker is still true, so no Redis
  needed) keyed by session cookie value, falling back to IP. Return 429 with
  a `Retry-After` header past the limit.
- **Query length cap.** `QueryRequest.query` (`app.py:177-179`) has no max
  length today — an oversized query inflates embedding cost and can blow
  the context budget before `assemble_context()` even gets to trim it.
  Add a `pydantic` `max_length` (a few thousand chars is generous for a
  question) — one line, no new dependency.
- **Conversation growth cap (Phase 3).** `conversation_id` is
  client-generated with no auth beyond the session cookie
  (`Phase 3` above). Unbounded turns per conversation, or unbounded
  conversations per session, grow `qa_log` and the pinned-citation bucket
  without limit. Cap turns per conversation (e.g. 20-30) and surface "start
  a new conversation" in the UI past that, rather than enforcing server-side
  rejection alone.
- **`/api/flag-answer` abuse.** Currently anyone with a session can write
  arbitrary rows to `flagged_answers`. Low risk internally, but a stuck
  client retry-loop can spam it — cheap to fold into the same per-session
  limiter above rather than build a separate one.
- **New `/api/testplans*` endpoints (Phase 1).** Read-only, but still put
  them behind the same auth gate as everything else (don't accidentally
  expose them unauthenticated) and behind the shared rate limiter — no
  reason to let a script hammer Supabase for the picker's sake.
- **What NOT to build:** no CAPTCHA, no WAF, no per-user accounts/roles.
  Single shared password + a lightweight in-memory limiter matches the
  actual threat model (internal tool, casual link leaks) called out in
  `auth.py`'s own threat-model comment. Revisit only if this ever goes
  public.

## Suggested build order

1. Branch `specGPT0` off `deepthought-vm`.
2. This doc.
3. Phase 1 (IOL ingestion) — independent of UI work.
4. Phase 2 (test plan picker) — smaller, validates the pinned/reserved
   fence-bucket pattern Phase 3 also needs.
5. Phase 3 (chat interface) — largest phase, reuses the pattern proven in
   Phase 2.

## Verification

- Phase 1: ingest a couple of real IOL test plans, confirm rows land in
  `test_plans`, confirm `fetch_test_plans()` returns them.
- Phase 2: pick a step in the UI, submit a query, confirm the injected text
  shows in `DEBUG_PIPELINE=1` traces as its own fence and is NOT present in
  `citations`/`used_chunks`.
- Phase 3: run a 4-5 turn conversation citing different sections each turn,
  confirm earlier citations stay answerable without ballooning
  `tokens_used`.
- Toggle off still works unmodified.
