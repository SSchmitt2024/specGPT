# Flag Answer — Implementation Plan

A feature that lets a user flag a low-quality or incorrect answer and
optionally describe the problem. Flagging is a one-click action via a
**sticky flag icon pinned to the bottom-right** of the viewport. Submitting a
flag writes one row to a new Supabase table capturing everything needed to
reproduce and triage the issue: the prompt, the generated answer, the full
pipeline config, the citations, the pipeline trace (the "critical steps"), and
timing/token metadata.

This doc is the build plan only — no code has been changed yet.

---

## 0. Context (how the app is wired today)

| Concern | Where it lives |
| --- | --- |
| FastAPI app + embedded single-page HTML/CSS/JS UI | `src/pipeline/app.py` |
| Supabase client (`create_client`, cached) | `src/pipeline/search.py` → `supabase_client()` |
| Schema + migrations (idempotent SQL) | `scripts/supabase_schema.sql` |
| Pipeline config dataclass | `src/pipeline/orchestrator.py` → `PipelineConfig` |
| Auth dependency for API routes | `src/pipeline/app.py` → `require_auth` |
| Response model returned to the UI | `src/pipeline/app.py` → `QueryResponse` |

**Key insight that makes this cheap to build:** `QueryResponse`
(`src/pipeline/app.py:173`) already contains every field we want to persist:

```python
class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[dict]
    config: dict
    pipeline_trace: list[dict] | None = None   # ← the "critical steps"
    latency_ms: float
    tokens_used: dict | None = None
    agentic: bool = False
    gap_hint: dict | None = None
    request_id: str | None = None
```

The UI already holds this object after `displayResults(data)` runs
(`src/pipeline/app.py` ~line 4043). So the flag flow does **not** need to
re-run anything — it just snapshots the last response plus an optional note and
POSTs it.

> **"Critical steps" = `pipeline_trace`.** Each entry is a `PipelineStage`
> (`stage` name, `input`, `output`, `took_ms`, `metadata`). Persisting the
> trace is what lets a reviewer see exactly which retrieval/rerank/generation
> steps produced the flagged answer.

---

## 1. Supabase table

Add to `scripts/supabase_schema.sql` (keep it `IF NOT EXISTS` so the file stays
re-runnable, matching the existing convention in that file).

```sql
-- ── Flagged answers (user-reported answer quality issues) ───────────────────
CREATE TABLE IF NOT EXISTS flagged_answers (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at    timestamptz NOT NULL DEFAULT now(),

    -- What the user asked and what we answered
    query         text  NOT NULL,
    answer        text  NOT NULL,

    -- Full reproduction context
    config         jsonb NOT NULL,          -- PipelineConfig as sent/used
    pipeline_trace jsonb,                    -- the "critical steps" (stage list)
    citations      jsonb,                    -- sources shown to the user
    spec           text,                     -- convenience: config->>'spec'
    llm_model      text,                     -- convenience: config->>'llm_model'
    agentic        boolean NOT NULL DEFAULT false,

    -- Timing / cost
    latency_ms     double precision,
    tokens_used    jsonb,

    -- User-supplied
    reason         text,                     -- optional free-text explanation

    -- Triage workflow
    status         text NOT NULL DEFAULT 'open',   -- open | reviewing | resolved | wontfix
    flagged_by     text                            -- session id / email if available
);

CREATE INDEX IF NOT EXISTS flagged_answers_created_idx ON flagged_answers (created_at DESC);
CREATE INDEX IF NOT EXISTS flagged_answers_status_idx  ON flagged_answers (status);
```

**Notes**
- `spec` and `llm_model` are denormalized out of `config` purely so triage
  queries / dashboards don't have to dig into JSON every time. They're optional.
- `status` gives a minimal triage lifecycle; default `'open'`.
- **RLS:** the server uses the Supabase **service key** (`SUPABASE_KEY` in
  `search.py`) which bypasses RLS, and all inserts go through the authenticated
  `/api/flag-answer` route — so no RLS policy is strictly required. If RLS is
  enabled project-wide, add a policy permitting the service role to insert.

**To apply:** run the updated `scripts/supabase_schema.sql` in the Supabase SQL
editor (or via `psql`), same as every other schema change in this repo.

---

## 2. Backend — request model + endpoint

In `src/pipeline/app.py`.

### 2a. Request model (near the other `BaseModel`s, ~line 190)

```python
class FlagAnswerRequest(BaseModel):
    """Request body for /api/flag-answer."""
    query: str
    answer: str
    config: dict
    pipeline_trace: list[dict] | None = None
    citations: list[dict] | None = None
    latency_ms: float | None = None
    tokens_used: dict | None = None
    agentic: bool = False
    reason: str | None = None          # optional user explanation
```

### 2b. Endpoint (alongside the other `/api/*` routes)

```python
@app.post("/api/flag-answer")
async def flag_answer_endpoint(
    req: FlagAnswerRequest,
    _: bool = Depends(require_auth),
) -> dict:
    from src.pipeline.search import supabase_client

    config = req.config or {}
    row = {
        "query": req.query,
        "answer": req.answer,
        "config": config,
        "pipeline_trace": req.pipeline_trace,
        "citations": req.citations,
        "spec": config.get("spec"),
        "llm_model": config.get("llm_model"),
        "agentic": req.agentic,
        "latency_ms": req.latency_ms,
        "tokens_used": req.tokens_used,
        "reason": (req.reason or "").strip() or None,
    }
    try:
        supabase_client().table("flagged_answers").insert(row).execute()
    except Exception as exc:  # don't 500 the UI over a logging-style write
        logger.exception("flag insert failed")
        raise HTTPException(status_code=502, detail={"error": "flag_failed"}) from exc
    return {"ok": True}
```

**Decisions**
- Reuse `require_auth` so only signed-in sessions can write (consistent with
  every other `/api/*` route).
- Reuse the cached `supabase_client()` from `search.py` — no new client/config.
- Size guard (optional): truncate `answer`/`pipeline_trace` if you want a hard
  cap; JSONB handles large payloads fine for expected volume, so skip unless
  abuse becomes a concern.

---

## 3. Frontend — sticky flag icon + modal

All in the embedded HTML/CSS/JS in `src/pipeline/app.py`. The UI is vanilla
JS with CSS variables (`--surface`, `--accent`, `--border`, `--text`, …) and a
light/dark theme — match those tokens, don't introduce a framework.

### 3a. Capture the last response

`displayResults(data)` already receives the full `QueryResponse`. Stash it so
the flag button has something to send:

```javascript
window._lastResponse = data;   // add inside displayResults(...)
```

(The codebase already does similar with `window._pipeTrace = data.pipeline_trace`.)

### 3b. Sticky flag button (HTML)

Add once, near the end of `<body>`:

```html
<button id="flag-fab" class="flag-fab" type="button"
        aria-label="Flag this answer" title="Flag this answer" hidden>
  <!-- inline flag SVG icon -->
  <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
    <path d="M5 3v18M5 4h11l-2 4 2 4H5" fill="none"
          stroke="currentColor" stroke-width="2"
          stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
</button>
```

### 3c. Styling (sticky bottom-right)

```css
.flag-fab {
  position: fixed;
  right: 20px;
  bottom: 20px;
  z-index: 1000;
  width: 44px; height: 44px;
  display: inline-flex; align-items: center; justify-content: center;
  border-radius: 999px;
  background: var(--surface);
  color: var(--text);
  border: 1px solid var(--border);
  box-shadow: 0 2px 10px rgba(0,0,0,.18);
  cursor: pointer;
  transition: transform .12s ease, color .12s ease, border-color .12s ease;
}
.flag-fab:hover { transform: translateY(-1px); color: var(--accent); border-color: var(--accent); }
.flag-fab.flagged { color: #c0392b; border-color: #c0392b; }  /* confirmation state */
```

Use `position: fixed` (truly viewport-pinned regardless of scroll); `bottom:20px;
right:20px`. Keep it `hidden` until the first answer renders, then unhide in
`displayResults`.

### 3d. Modal for the optional explanation

A lightweight modal (reuse existing modal/overlay styles if present, otherwise a
simple fixed overlay) with:
- A short prompt: *"What was wrong with this answer? (optional)"*
- A `<textarea id="flag-reason">`
- **Submit** and **Cancel** buttons.

Submitting calls `submitFlag()`; the explanation is optional (empty is allowed).

### 3e. Submit logic (JS)

```javascript
async function submitFlag(reason) {
  const d = window._lastResponse;
  if (!d) return;
  const res = await fetch("/api/flag-answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    body: JSON.stringify({
      query: d.query,
      answer: d.answer,
      config: d.config,
      pipeline_trace: d.pipeline_trace || null,
      citations: d.citations || null,
      latency_ms: d.latency_ms ?? null,
      tokens_used: d.tokens_used || null,
      agentic: !!d.agentic,
      reason: reason || null,
    }),
  });
  if (res.ok) {
    document.getElementById("flag-fab").classList.add("flagged");
    // show a brief "Thanks — flagged" toast, then close the modal
  } else {
    // show an inline error in the modal
  }
}
```

**Wiring**
- Click `#flag-fab` → open modal.
- Modal Submit → `submitFlag(textarea.value)` → on success close modal + toast.
- Reset the `.flagged` state when a new query starts (in `runQuery`).

---

## 4. UX details / decisions

- **Visibility:** hide the FAB until there's an answer to flag; show it in
  `displayResults`. Hide again (or reset) when a fresh query begins.
- **Optional explanation:** allowed to be empty — one click is enough to flag.
- **Confirmation:** turn the icon red + brief toast ("Thanks — flagged") so the
  user knows it landed.
- **Failure handling:** the endpoint returns `502 {error:"flag_failed"}` on a
  DB error; surface that in the modal rather than silently dropping it.
- **Don't block the answer UI:** flagging is fire-and-forget from the user's
  perspective; never let a flag failure interfere with normal Q&A.

---

## 5. Build checklist (in order)

1. [ ] **Schema** — add `flagged_answers` table + indexes to
   `scripts/supabase_schema.sql`; run it against Supabase.
2. [ ] **Backend model** — add `FlagAnswerRequest` to `src/pipeline/app.py`.
3. [ ] **Backend route** — add `POST /api/flag-answer` (guarded by
   `require_auth`, inserts via `supabase_client()`).
4. [ ] **Frontend capture** — set `window._lastResponse = data` in
   `displayResults`.
5. [ ] **Frontend FAB** — add the sticky flag button HTML + `.flag-fab` CSS.
6. [ ] **Frontend modal** — add the optional-explanation modal + open/close
   wiring.
7. [ ] **Frontend submit** — add `submitFlag()`, confirmation state, error
   handling; reset state on new query.
8. [ ] **Manual test** — run a query, click the flag, submit with and without a
   note, confirm a row lands in `flagged_answers` with `config` +
   `pipeline_trace` populated.
9. [ ] **(Optional) Triage view** — a simple authenticated
   `/api/flags` list endpoint or a Supabase dashboard query ordered by
   `created_at DESC` for reviewing flags later.

---

## 6. Out of scope (possible follow-ups)

- Admin UI for browsing/resolving flags (the `status` column is ready for it).
- Rate limiting / dedupe (e.g. one flag per `request_id`).
- Notifying maintainers (email/Slack) on new flags.
- Linking a flag back to a reproducible `request_id` for one-click re-run.
