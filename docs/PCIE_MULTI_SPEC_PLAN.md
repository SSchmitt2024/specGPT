# Multi-Spec Plan — Add the NVMe PCIe Transport Specification alongside the Base Spec

**Goal.** Let specGPT serve **two** NVMe specifications side by side:

- **Base** — *NVM Express Base Specification* (today's corpus), and
- **PCIe** — *NVM Express PCIe Transport Specification*.

At the **top of the UI** the user picks **Base** or **PCIe**; every query is then
scoped to that spec's data so search only ever returns rows from the selected
spec. The data-pipeline **scripts ask first** which spec to build/ingest.

This document is the implementation plan. **All phases below are now implemented
on branch `claude/nvme-pcie-spec-plan-kIuUR`** using **Option A** (one tagged
corpus): identical parsing, identical data schema, every row tagged with a
`spec` discriminator, and every retriever filtered on it. What remains is
**operational**, not code: apply the schema migration and ingest the PCIe PDF.

> **To go live:**
> 1. Apply `scripts/supabase_schema.sql` (adds the `spec` column + widens lookup
>    PKs; existing rows backfill to `spec='base'`).
> 2. Obtain `nvme_spec/NVMe_PCIe_transport.pdf`, confirm its page offset, then
>    run `NVME_SPEC=pcie ./scripts/rerun_pipeline.sh` and
>    `NVME_SPEC=pcie ./scripts/run_phase2.sh`.
> 3. The UI **Spec** picker (top-left of the header) then flips all retrieval
>    between Base and PCIe.

---

## 1. Why the PCIe spec needs its own corpus

Since NVMe 2.0 the specification was split into a transport-agnostic **Base
Specification** (+ Command Set specs) and per-transport binding specs: **PCIe**,
**RDMA**, and **TCP**. The PCIe Transport Specification is a distinct document
that defines how NVMe binds to PCI Express:

- **PCIe register map** — the NVMe controller register set mapped into PCIe
  memory space (BAR0/BAR1), plus the **doorbell** registers.
- **PCI/PCIe configuration space** — PCI header, **MSI / MSI-X** capability
  structures, the PCI Express Capability, AER, power management, etc.
- **Interrupt mechanisms** — pin-based, MSI, and MSI-X behavior on PCIe.
- **Transport-specific behaviors** — function-level reset, SR-IOV/VFs, link
  states, and other PCIe-only semantics.

These are *different figures, fields, registers, and definitions* from the Base
spec. They share the same parsing pipeline and retrieval machinery, but the
**content must not be co-mingled**: a question about "MSI-X table" should only
surface PCIe rows, and "Identify Controller data structure" should surface Base
rows. Hence a **separate, spec-tagged corpus** per document.

> **Inputs the maintainer must supply** (the spec PDFs are gitignored; see
> `.gitignore`):
> - `nvme_spec/NVMe_spec_full.pdf` — Base (already used today).
> - `nvme_spec/NVMe_PCIe_transport.pdf` — PCIe Transport (new). Filename is
>   configurable via `SPEC_PDF_PATH`.
> - The PCIe PDF's **page offset** (`pdf_page − printed_page`). The Base value
>   is 23/24; the PCIe document's cover+TOC length differs, so it is prompted /
>   set via `SPEC_PAGE_OFFSET` (placeholder default `12` — **verify** against
>   the real PDF before trusting citations).

---

## 2. What already exists (multi-spec readiness audit)

The codebase was partly built for this. Inventory before changing anything:

| Layer | Already multi-spec? | Notes |
|---|---|---|
| `spec_chunks` table | **Partial** | Has `spec_version` + `spec_document` columns. |
| RPCs `match_spec_chunks` / `search_spec_chunks_text` | **Yes** | Both accept a `spec_version` filter (`scripts/supabase_schema.sql`). |
| `search.py` `_FILTER_KEYS` | **Yes** | Includes `spec_version`; coerces it into the jsonb filter. |
| `bm25_index.py` | **Yes** | `_matches_filter()` filters the in-memory corpus by `spec_version`. |
| Lookup tables `spec_fields` / `spec_field_index` / `spec_tables` | **No** | No spec discriminator column. Structured lookup is **not** spec-scoped. |
| `retriever.py` loaders | **No** | `load_fields/_field_index/_tables_by_figure` read whole tables. |
| `orchestrator.py` | **No** | Never *passes* a spec filter to `search.*` on the main hybrid path. |
| `PipelineConfig` | **No** | No `spec` field. |
| `/api/query` + frontend | **No** | No spec selector; one global Base corpus assumed. |
| Phase-1 parser modules / Phase-2 scripts | **Done on this branch** | Now read `src/spec_env.py` (see §4). |

**Design consequence:** the cleanest path reuses the discriminator that already
threads through vector + tsvector + BM25 — but `spec_version` alone is a poor
key (Base "2.1" vs PCIe "1.1" happen to differ today, but versions collide
across reissues). We therefore standardize on an explicit **`spec`** tag
(`"base"` | `"pcie"`) carried *alongside* `spec_document`/`spec_version`.

---

## 3. Architecture decision — one tagged corpus vs separate tables

Two ways to "search the appropriate tables":

**Option A — shared tables + a `spec` discriminator column (RECOMMENDED).**
Add a `spec text` column to `spec_chunks` and the three lookup tables, filter on
it everywhere. Reuses all existing infra (one ANN index, one BM25 corpus build,
one set of RPCs). Smallest change, easy to add a third transport (TCP/RDMA)
later by just tagging rows. Isolation is enforced by an always-applied filter.

**Option B — physically separate tables per spec** (`spec_chunks_pcie`, …).
Stronger isolation, independent ANN tuning per spec, but multiplies table names,
RPCs, indexer targets, and retriever code paths, and re-introduces the
table-name routing everywhere. Overkill at this corpus size (~1.9k Base rows).

> **Recommendation: Option A.** It matches the grain the code already has
> (`spec_version` filter) and keeps the BM25/vector/lookup paths single. The
> rest of this plan assumes Option A. (If strict physical isolation is ever
> required, Option B is a localized swap of table names behind a
> `table_for(spec)` helper.)

---

## 4. Phase 0 (DONE) — pipeline + scripts parameterized

Implemented on branch `claude/nvme-pcie-spec-plan-kIuUR`:

- **`src/spec_env.py`** — single source of truth that resolves per-spec values
  from the environment, **defaulting to today's Base behavior** when unset:

  | Env var | Default (Base) | Meaning |
  |---|---|---|
  | `NVME_SPEC` | `base` | Logical spec id (`base`/`pcie`). |
  | `SPEC_DATA_DIR` | `data` | JSON artifact dir (PCIe → `data/pcie`). |
  | `SPEC_PDF_PATH` | `nvme_spec/NVMe_spec_full.pdf` | Source PDF. |
  | `SPEC_PAGE_OFFSET` | *(per-module)* | `pdf_page − printed_page`. Unset for Base so each module keeps its historical value (toc=24, others=23). |
  | `SPEC_DOCUMENT` | `NVM Express Base Specification` | `spec_document` tag. |
  | `SPEC_VERSION` | `2.1` | `spec_version` tag. |

- **Phase-1 modules** now resolve paths/offsets/metadata through `spec_env`
  (`toc_rebuild`, `deep_sections`, `prose`, `tables`, `fields`,
  `relationships`, `llm/extract_relationships`, `llm/reconcile`,
  `llm/generate_cards`). With no env set, output is byte-for-byte the old Base
  behavior (verified: defaults resolve to `data/*.json`, Base PDF, offsets
  23/24).

- **Phase-2 ingest scripts** (`chunker`, `embedder`, `indexer`,
  `load_lookup_data`) default their data dir to `$SPEC_DATA_DIR` (else `data`).

- **`scripts/rerun_pipeline.sh` and `scripts/run_phase2.sh` ask first** which
  spec to run (Base/PCIe), export the env above, prompt for the PCIe page
  offset, write per-spec backup dirs (`Backups/pipeline_<spec>_<ts>`), and
  record the spec in the run manifest. A pre-set `NVME_SPEC` (env/.env) skips
  the prompt for non-interactive runs.

**Result today:** `NVME_SPEC=pcie ./scripts/rerun_pipeline.sh` produces a fully
independent `data/pcie/` corpus from the PCIe PDF, with cards tagged
`spec_document="NVM Express PCIe Transport Specification"`.

---

## 5. Phase 1 (DONE) — Supabase schema (spec discriminator)

Add to `scripts/supabase_schema.sql` (all `IF NOT EXISTS` / idempotent):

1. **Columns**
   ```sql
   ALTER TABLE spec_chunks      ADD COLUMN IF NOT EXISTS spec text NOT NULL DEFAULT 'base';
   ALTER TABLE spec_fields      ADD COLUMN IF NOT EXISTS spec text NOT NULL DEFAULT 'base';
   ALTER TABLE spec_field_index ADD COLUMN IF NOT EXISTS spec text NOT NULL DEFAULT 'base';
   ALTER TABLE spec_tables      ADD COLUMN IF NOT EXISTS spec text NOT NULL DEFAULT 'base';
   ```
   `DEFAULT 'base'` backfills the existing Base rows automatically.

2. **Primary-key widening.** `spec_tables` (PK `figure_number`) and
   `spec_fields` (PK `name`) can collide across specs (both have a "Figure 1",
   etc.). Change PKs to composite `(spec, figure_number)` / `(spec, name)`. The
   `id` text PK on `spec_chunks` should be prefixed at write time (see §6) so
   no PK change is needed there; if you prefer, add `(spec, id)`.

3. **Indexes**
   ```sql
   CREATE INDEX IF NOT EXISTS spec_chunks_spec_idx       ON spec_chunks (spec);
   CREATE INDEX IF NOT EXISTS spec_field_index_spec_idx  ON spec_field_index (spec, field_name);
   ```

4. **RPC filter.** Add a `spec` predicate to both RPCs:
   ```sql
   AND (filter->>'spec' IS NULL OR c.spec = filter->>'spec')
   ```
   Keep the existing `spec_version` predicate for back-compat.

---

## 6. Phase 2 (DONE) — ingest writes the spec tag

- **`scripts/indexer.py`** — set `"spec": spec_env.spec()` (or read
  `NVME_SPEC`) on every chunk row, and **prefix chunk ids** with the spec
  (`f"{spec}:{chunk_id}"`) so Base/PCIe ids never collide.
- **`scripts/load_lookup_data.py`** — set `spec` on every `spec_fields`,
  `spec_field_index`, `spec_tables` row.
- These already read `$SPEC_DATA_DIR`; they only need the tag added.

Re-run for each spec: `NVME_SPEC=base ./scripts/run_phase2.sh` then
`NVME_SPEC=pcie ./scripts/run_phase2.sh`.

---

## 7. Phase 3 (DONE) — backend retrieval scoping

The contract: **every retrieval call carries the active `spec`**, and structured
lookup loads only that spec's tables.

1. **`PipelineConfig`** (`src/pipeline/orchestrator.py`) — add:
   ```python
   spec: str = "base"   # "base" | "pcie"
   ```
   It's already serialized via `to_dict()` and round-trips through
   `/api/query`'s `config`.

2. **Thread `spec` into the search filter.** In `orchestrator.py`, build a base
   filter `{"spec": config.spec}` and pass it to **every** `search.vector_search
   / tsvector_search / bm25_search` call (the main hybrid path at lines ~195–203
   currently passes no filter), merging with any per-call filter
   (`section_prefix`, etc.).

3. **`retriever.py` lookup loaders** — make `load_fields`,
   `load_field_index`, `load_tables_by_figure` **spec-aware**: key the
   `lru_cache` by `spec`, and add `.eq("spec", spec)` to the `_paginate`
   queries (and filter the local-JSON fallback by the active data dir).
   `structured_lookup()` gains a `spec` argument that the orchestrator passes
   from `config.spec`.

4. **`bm25_index.py`** — the corpus is built once across all specs; either
   (a) include `spec` in the fetched columns and rely on the existing
   `_matches_filter` (extend it to check `spec`), or (b) build one index per
   spec. (a) is simplest and already 90% there.

5. **Agentic targeted-fetch** (`orchestrator.py` ~line 442) — pass `spec` so
   direct figure/field/section fetches stay in-spec.

---

## 8. Phase 4 (DONE) — API surface

- **`/api/query`** and **`/api/refine`** (`src/pipeline/app.py`) — already accept
  a free-form `config` dict, so `{"spec": "pcie"}` flows through with **no
  signature change**. Add light validation (reject unknown spec ids) and ensure
  `spec` is echoed back in the response `config`.
- **New `GET /api/specs`** (gated) — returns the available specs for the UI to
  populate the selector, e.g.
  ```json
  [{"id":"base","label":"Base Specification","version":"2.1"},
   {"id":"pcie","label":"PCIe Transport","version":"1.1"}]
  ```
  Back it with a small constant (or `SELECT DISTINCT spec, spec_document,
  spec_version FROM spec_chunks`).

---

## 9. Phase 5 (DONE) — UI selector (top of the page)

The frontend is server-rendered as `FRONTEND_HTML` in `src/pipeline/app.py`
(there is no separate `frontend/` build despite the README diagram). Wiring
points already in place: the header at `app.py:~2202` (next to the
`global-model-select`), and the query POST at `app.py:~3952` that sends
`{query, config, ...}`.

1. **Markup.** Add a spec picker in the header, mirroring the
   `.global-model-picker` styling:
   ```html
   <label class="global-model-picker" title="Which NVMe spec to search">
     <span class="global-model-picker-label">Spec</span>
     <select id="global-spec-select"></select>
   </label>
   ```
   Place it left of the Model picker so "pick the spec, then ask" reads
   top-to-bottom.

2. **Populate.** On load, `fetch("/api/specs")` and fill the `<select>`;
   default to `base`. Persist the choice in `localStorage` so it survives
   reloads.

3. **Send.** In the query handler, include the selection in the request:
   `config: { ...buildConfig(), spec: document.getElementById("global-spec-select").value }`.
   Do the same in the `/api/refine` call so refinement stays in-spec.

4. **Affordance.** Show the active spec near the answer header and in the
   sources list (the chunk's `spec_document` is already available) so it's
   obvious which document a citation came from. Optionally update the page
   subtitle ("Ask questions about the NVMe **PCIe Transport** spec…").

---

## 10. Migration, testing, rollout

1. **Schema first** — apply §5 to Supabase (`scripts/apply_schema.py`). Existing
   Base rows auto-tag `spec='base'`. No data loss.
2. **Backend with default** — ship §7–8 with `spec` defaulting to `base`;
   confirms Base behavior is unchanged before any PCIe data exists.
3. **Ingest PCIe** — obtain the PDF, set the page offset, run Phase-1 then
   Phase-2 for `pcie`. Spot-check `data/pcie/` artifacts (TOC depth, figure
   captions, a known register like the doorbell / MSI-X table).
4. **UI** — ship §9; verify the selector flips result provenance.
5. **Tests** — extend `tests/` with: `spec_env` default/override resolution;
   a retriever test asserting a PCIe-tagged query returns no Base rows (and
   vice-versa); an `/api/query` test passing `config.spec`.

**Acceptance:** with the selector on **PCIe**, every citation's
`spec_document` is the PCIe Transport spec; on **Base**, every citation is the
Base spec; structured lookups (e.g. a register/field name) resolve from the
selected spec's tables only.

---

## 11. Open questions / decisions for the maintainer

- **Corpus strategy** — confirm Option A (shared tables + `spec` tag). This plan
  assumes it.
- **PCIe page offset & exact PDF filename** — must be verified against the real
  document; the script uses a placeholder default until then.
- **Default spec on first load** — `base` (assumed). Change in §9 step 2 if
  PCIe should lead.
- **Re-embedding cost** — PCIe is a much smaller document than Base; one Voyage
  embedding pass, well within the existing per-run budget cap.
