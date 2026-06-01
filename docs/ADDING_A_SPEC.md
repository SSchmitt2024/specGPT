# Adding a new specification

specGPT is multi-spec: every pipeline step, lookup row, and retriever read is
scoped by a logical **spec id** (`base`, `pcie`, `command`, …). Adding a corpus
is wiring, not new code — the Python modules already read the spec from the
environment via `src/spec_env.py`. See `docs/PCIE_MULTI_SPEC_PLAN.md` for the
underlying design.

The `command` (NVM Command Set) spec is the reference example; copy its pattern.

## Steps

1. **Drop the PDF** into `nvme_spec/` (e.g. `nvme_spec/NVMe_<name>_full.pdf`).

2. **Find the page offset** = `pdf_page - printed_page`, expressed in the
   0-indexed page-iteration convention. Open the PDF, find a body page, and
   subtract its printed page number from its 0-indexed position. Example: NVM
   Command Set printed p.8 sits at 0-indexed pdf idx 7 → offset `-1`.
   (`toc_rebuild` auto-adds `+1`; don't apply it yourself — see `src/spec_env.py`.)

3. **`scripts/rerun_pipeline.sh`** (Phase 1) — in `select_spec()` add the menu
   line and a `case` branch exporting `NVME_SPEC`, `SPEC_DATA_DIR`
   (`data/<id>`), `SPEC_PDF_PATH`, `SPEC_DOCUMENT`, `SPEC_VERSION`, and a
   prompted `SPEC_PAGE_OFFSET` (default from step 2). End with `mkdir -p "$SPEC_DATA_DIR"`.

4. **`scripts/run_phase2.sh`** (Phase 2 — indexing) — in `select_spec()` add the
   matching menu line and `case` branch. Phase 2 only consumes JSON, so it needs
   `NVME_SPEC`, `SPEC_DATA_DIR`, `SPEC_DOCUMENT`, `SPEC_VERSION` only (no
   PDF/offset).

5. **Frontend** — add one row to `AVAILABLE_SPECS` in `src/pipeline/app.py`:
   `{"id": "<id>", "label": "<UI label>", "version": "<x.y>"}`. The spec picker
   populates itself from `/api/specs`, and `<id>` is what every `spec_*` row is
   tagged with and what the retrievers filter on — keep it identical to
   `NVME_SPEC` above.

6. **Run it.** `./scripts/rerun_pipeline.sh` → pick the new spec → `all` (builds
   `data/<id>/*.json`). Then `./scripts/run_phase2.sh` → same spec → chunk,
   embed, apply schema, index, load lookup data. The new spec is now selectable
   in the UI.
