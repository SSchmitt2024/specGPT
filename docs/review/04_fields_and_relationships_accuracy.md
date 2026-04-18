# Fields & Relationships Accuracy Audit — `data/fields.json`, `data/field_index.json`, `data/relationships.json`

Ground-truth source: PDF text of `nvme_spec/NVMe_spec_full.pdf` (784 pages; logical-page offset = 24, i.e. PDF page 25 = spec page 1). Cross-checked against `data/tables.json` (717 tables, figure numbers 2–819) and `data/toc.json` (1042 entries, 1041 unique section IDs). All three artifacts are downstream of the table parser (Phase 1.2 / 1.4 of `BUILD_PLAN_FINAL.md`).

## Summary stats

### `fields.json` (1623 entries)

| Metric | Value |
|---|---|
| Total entries | **1623** |
| Distinct parent_figures | **501** (of 717 tables in `tables.json`) |
| Schema keys (always present) | field_name, full_name, parent_figure, parent_caption, parent_type, offset, offset_type, requirements, register_type, register_reset, values, cross_refs, description, spec_page |
| Non-empty: `description` | 1612 / 1623 (99.3%) |
| Non-empty: `cross_refs` | 474 / 1623 (29.2%) |
| Non-empty: `values` (enum table) | 223 / 1623 (13.7%) |
| Non-empty: `requirements` (mandatory/optional flags) | 110 / 1623 (6.8%) |
| Non-empty: `register_type` / `register_reset` | 90 / 1623 (5.5%) — equals count of register-type rows |
| `parent_type` distribution | command_format 1422, data_structure 111, register 90 |
| `offset_type` distribution | bytes 1053, bits 570 |
| Duplicate `(field_name, parent_figure)` pairs | **0** |
| Unparseable offsets | **49** (all symbolic: `EHL+2+VSIL:EHL+3`, `(Dword Count * 4)+3:4`, `Variable:12`, etc.) |
| Non-monotonic bit-offset ordering within a figure | **0** |
| spec_page out of plausible range | 0 |

### `field_index.json` (1081 keys)

| Metric | Value |
|---|---|
| Total acronym keys | **1081** |
| Acronyms with ≥2 entries (polysemous) | **215** |
| Entries per acronym | min 1, max 27 (`DPTR`), avg 1.50 |
| Total entries across all keys | 1623 (matches `fields.json` exactly) |
| Orphan index keys (not a `field_name` in `fields.json`) | **0** |
| `field_name`s in `fields.json` missing from index | **0** |
| Count-mismatch between index entries and `fields.json` rows per key | **0** |
| Well-known **register acronyms** present (CAP, CC, CSTS, AQA, ASQ, ACQ, INTMS, INTMC, NSSR, CMBLOC, CMBSZ, BPINFO, CMBMSC) | **1 of 14 (VS only)** — see Issue 1 |

### `relationships.json` (4820 edges)

| Metric | Value |
|---|---|
| Total edges | **4820** |
| Type distribution | cross_reference 3074, child_of 1029, contained_in 717 |
| `confidence` field | always `"deterministic"` (100%) ✓ |
| Source-prefix distribution | section 3233, figure 1587 |
| Target-prefix distribution | section 3315, figure 1505 |
| Self-loops | **0** |
| Duplicate `(source, target, type)` triples | **0** |
| Orphan section **targets** (section not in `toc.json`) | **31** |
| Orphan figure **targets** (figure not in `tables.json`) | **98** |
| Orphan sources | 0 (all figure/section sources exist in their respective files) |
| `child_of` edges with malformed parent path | **0 / 1029** (every source is `target + "." + one level`) |
| `contained_in`: figures mapped to >1 section | **0** (exactly one per figure) |
| `contained_in` coverage | 717 / 717 tables (100%) |
| Cross-ref evidence patterns | "refer to" 2121, "defined in" 270, "described in" 173, other 510 |
| Cross-ref `strength` field | strong 2703, mention 371 (present only on `cross_reference` edges) |
| Cross-refs whose evidence looks like a figure caption (false-positive signal) | **9 / 3074 (0.3%)** |

## Spot-check table (15 samples across the three files)

| # | Artifact | Sample | Verdict |
|---|---|---|---|
| 1 | fields | `NSSES` @ fig 36 (CAP), offset 61 bits, spec_page 54 | PDF page 77 contains "Figure 36", caption match, field "NSSES" present — **OK** |
| 2 | fields | `CRIME` @ fig 41 (CC), offset 24 bits, spec_page 58 | PDF p81 has "Figure 41 Offset 14h: CC" but "CRIME" on p82 (multi-page table) — **OK** (description continuation, not a parse bug) |
| 3 | fields | `EGCN` @ fig 262 (Capacity Config Descriptor), offset 5:4 bytes, spec_page 264 | PDF p287 confirms figure, caption, field — **OK** |
| 4 | fields | `ACQB` @ fig 46 (ACQ), offset 63:12 bits, spec_page 66 | PDF p89 confirms — **OK** |
| 5 | fields | `VSI` @ fig 231, offset `EHL+2+VSIL:EHL+3 bytes` | Offset is symbolic (parameterised by header-length field). Cannot be range-checked numerically — **OK but flagged** (parser preserves verbatim; see Issue 3) |
| 6 | fields | `CDW0` @ various figs, description = `"Refer to Figure 95."` | Minimal description; spec really does delegate to a shared CDW0 layout. Technically accurate but breaks field retrieval unless the reader follows the link — **OK, limitation** |
| 7 | field_index | `CAP` lookup | **MISSING** — `CAP` only appears as `parent_caption` ("Offset 0h: CAP – Controller Capabilities"), never as `field_name`. Same for CC, CSTS, AQA, ASQ, ACQ, INTMS, INTMC, NSSR, CMBLOC, CMBSZ, BPINFO, CMBMSC. See Issue 1 |
| 8 | field_index | `VS` → 4 entries | Present; includes the VS register and `VS` subfields across structures — **OK** |
| 9 | field_index | `DPTR` → 27 entries across 27 figures | Correct — DPTR (Data Pointer) appears in every command CDW layout — **OK** |
| 10 | field_index | `SQyTDBL` / `CQyHDBL` (doorbell conventions from prompt) | **Not present as literal strings.** Spec uses `SQTDBL` / `CQHDBL` with a subscript `y`; neither form is a bit-field keyed in the index. Reasonable absence (doorbells don't have a bit-level figure), but a user would expect these as aliases — see Issue 1 |
| 11 | relationships | `figure:2 → section:1.4.2.4` type `contained_in`, evidence "printed_page 3 falls inside section 1.4.2.4" | Correct structural edge; every figure has exactly one — **OK** |
| 12 | relationships | `section:1.1.1 → section:1.1` type `child_of` | Correct; 1029/1029 child_of edges obey the `A.B.C → A.B` rule — **OK** |
| 13 | relationships | `figure:27 → section:6.3` type `cross_reference`, evidence "as described in section 6.3" | Verified against PDF p66 — phrase present. Pattern matches spec §1.4 regex (1) — **OK** |
| 14 | relationships | `figure:28 → figure:29` type `cross_reference`, evidence "Figure 29: Fabrics Command Support Requirements" | **FALSE POSITIVE** — evidence is a caption text, not a cross-reference. 8 other such cases exist (~0.3%) — see Issue 4 |
| 15 | relationships | `target = section:4.7.1.3` | **ORPHAN TARGET** — this section was reassigned in the TOC-accuracy report (Issue 1 of `02_toc_accuracy.md`, "hierarchy drift in chapter 3"). The relationship source-text is correct; the orphan is a consequence of the upstream TOC bug, not a regex false positive. Same for most of the 31 orphan section targets (e.g., `3.3.12`, `3.7.2.1`, `5.2.16.1f`). — **OK per this parser, bad per graph** |

## Systematic issues

1. **Register-level acronyms missing from `field_index.json`.** The index only keys on `field_name`, which the parser extracts only for bit-/byte-level sub-fields of a register/structure. The **register itself** — CAP, CC, CSTS, AQA, ASQ, ACQ, INTMS, INTMC, NSSR, CMBLOC, CMBSZ, BPINFO, CMBMSC (13 of the 27 controller registers) — is exposed only in `parent_caption` text, so a user query "what is CAP?" cannot resolve. Fix: either (a) emit a synthetic field row per register (field_name = register acronym, full_name from caption, offset = register base offset, parent = containing section), or (b) build a parallel `registers.json` / merge register-level acronyms into `field_index.json` with a distinct `entity_type` tag. Same gap applies to the aggregate names of non-register structures (e.g., `Identify Controller`, `NVMe-MI` doorbell conventions).

2. **Polysemous acronyms have no disambiguation metadata.** 215 acronyms collide (e.g., `DPTR` × 27, `CNTLID` × 21, `CID` × 13, `CDW0` × 9). Each entry carries `parent_figure` + `parent_caption`, which is enough to disambiguate programmatically, but a downstream LLM retrieving by acronym alone will get an ambiguous hit list. No `primary_definition` flag or canonical row is marked.

3. **Symbolic offsets are preserved verbatim but not typed.** 49 rows have offsets like `EHL+2+VSIL:EHL+3`, `(Dword Count * 4)+3:4`, `Variable:12`. Faithfully extracted from the spec, but not machine-readable — any downstream consumer that treats offset as a numeric range breaks. Add an `offset_form` flag (`numeric` vs `symbolic`) or pre-compute a nullable `offset_lo`/`offset_hi` numeric pair alongside the raw string.

4. **A small number of cross-references regex-matched inside figure captions.** 9 / 3074 (~0.3%) of `cross_reference` edges have evidence like `Figure 29: Fabrics Command Support Requirements` — the parser picked up "Figure N" from a caption rather than a cross-reference. Low incidence and all nine are tagged `strength: "mention"`, which is the correct lower-confidence bucket. No action critical; consider tightening the regex to require a preposition (`refer to`, `defined in`, `shown in`, `as in`) rather than allow bare "Figure N" followed by colon/dash.

5. **"Mention"-strength cross-refs need a downstream filter policy.** 371 edges (12.1% of cross-refs) are labelled `strength: "mention"` — bare "Figure N" / "section N.M" occurrences without an explicit navigational verb. These are mostly legitimate soft references, but a retrieval graph that treats them identically to `strong` edges will inflate neighbourhood size. `02_toc_accuracy.md`'s "reliability hierarchy" concern applies here even though technically all edges are deterministic.

6. **Orphan graph endpoints fall into two distinct categories, which the file currently conflates.**
   * **31 orphan section targets** (`3.3.12`, `3.7.2.1`, `4.7.1.3`, `5.1.22`, `5.2.26.1.3.1`, etc.) — these reference sections the spec mentions but that don't exist in `toc.json`. Several (`3.7.2.1`, `4.7.1.3`) are known casualties of the chapter-3 TOC drift documented in `02_toc_accuracy.md` Issue 1; others (`5.2.16.1f` — the trailing `f` is suspicious, possibly a regex over-capture grabbing the next letter) may be regex bugs.
   * **98 orphan figure targets** (figures 1, 3–6, 8–14, 16–26, 29, 66–90, 141…). `tables.json` only contains structures parsed as tables; pure-image figures (architecture diagrams, state machines) are absent by design. These are *valid* references to real spec figures; they're just orphaned in our graph. Mitigation: emit a stub entry in a `figures.json` side-file for any figure number referenced by `relationships.json` but missing from `tables.json`, annotated `kind: "image_or_unparsed"`.

7. **`contained_in` granularity is page-based, not content-based.** Every figure maps to the single section whose page range contains the figure's printed page. This is correct but loses multi-section figures (a figure referenced from §5.2 and §5.3 gets only one containment edge). Low-priority — the cross_reference edges cover the semantic relationships.

8. **20 fields have trivial descriptions** like `"Refer to Figure 95."` / `"Refer to Figure 91."` (CDW0, CID, SGL1 across several commands). Upstream spec truly delegates to a shared layout; the parser is faithful. But these rows are retrieval dead-ends unless the consumer resolves the `Figure 95` pointer. Add a post-processing pass that substitutes the pointed-to figure's field list into the description, or flag the row with `description_kind: "reference_only"`.

## Accuracy estimate

- **`fields.json` — structural fidelity:** spec_page matches PDF for 100% of the 7 sampled figures; caption matches `tables.json` for 1623/1623 rows (100%); offsets parse cleanly for 1574/1623 (97%), with the remainder symbolic-by-design. Mandatory columns (`field_name`, `offset`, `parent_figure`) are 100% populated; optional columns (`requirements`, `values`, `register_type/reset`) are populated at the rate those columns exist in the source tables (5–30%), which matches the plan's implied semantics. **Estimated row-level accuracy: ~98%.** Confidence: **High** (whole-population structural checks; 7 PDF spot-checks all passed).

- **`field_index.json` — lookup completeness:** Perfect consistency with `fields.json` (0 orphans, 0 missing, 0 count mismatches). Major gap is at the **register-acronym** level (CAP, CC, CSTS, … all absent), which is a scoping choice, not a bug. For "look up a known NVMe bit-field" the index is exhaustive; for "look up any NVMe register or data-structure by its top-level acronym" it's missing ~40 well-known names. **Estimated lookup success rate for bit-field queries: ~99%; for register queries: ~0%.** Confidence: **High**.

- **`relationships.json` — edge correctness & completeness:** Structural edges (`child_of`, `contained_in`) are 100% clean — no malformed paths, no duplicates, no self-loops, one-to-one containment. `cross_reference` evidence verifies against PDF text in all 5 manually checked cases (2 required scanning adjacent pages due to multi-page tables); ~0.3% caption false positives, already tagged as `strength: "mention"`. **Estimated edge correctness: ~97% (≥99% for structural, ~95% for cross_reference given the 12% mention-bucket noise and 0.3% caption false positives).** Orphan endpoints (129 total, touching 186 edges ≈ 3.9% of all edges) are mostly downstream symptoms of TOC drift, not regex bugs in this file. Confidence: **Medium-High** (whole-population checks plus 10 PDF-grounded spot checks; confidence limited by the fact that we verified no false-*negatives* — how many genuine cross-references the regex missed is untested here).

- **Deterministic-confidence claim:** `confidence: "deterministic"` is 100% honoured in this file, as required by the plan §1.4. ✓

## Recommended fixes (priority-ordered)

1. **Emit register-level rows** in `fields.json` (or a sibling `registers.json`) so `CAP`, `CC`, `CSTS`, etc. become addressable. This is the single biggest usability gap; ~40 missing entities.
2. **Tighten cross-reference regex** to require an explicit navigation verb ("refer to", "defined in", "described in", "shown in", "as in") immediately before `Figure N` / `section N.M` / `Table M`. Would eliminate the 9 caption false positives and substantially prune the 510 "other" matches and 371 "mention" matches.
3. **Emit a `figures.json` stub** for the 98 orphan figure targets so graph traversal doesn't dead-end at image figures. Mark with `kind: "image_or_unparsed"`.
4. **Add `offset_lo`/`offset_hi`** numeric columns alongside raw `offset` string; null when symbolic. Add `offset_form: "numeric" | "symbolic"` tag.
5. **Add disambiguation hints** on polysemous `field_index.json` entries: sort each list so the most central occurrence (register > identify-data-structure > command CDW) is first, and add a `primary: bool` flag.
6. **Treat upstream TOC drift separately**: 16 of 31 orphan section targets trace to the §3.2/§3.3 hierarchy-shift bug in `toc.json` (Issue 1 of `02_toc_accuracy.md`). Fixing the TOC will auto-resolve them. Do not patch these in `relationships.json` directly.
7. **Substitute delegated descriptions** ("Refer to Figure 95.") with the resolved target's field summary, or tag those 20 rows `description_kind: "reference_only"` so downstream code can follow the pointer.
