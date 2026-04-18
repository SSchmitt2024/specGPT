# Tables Accuracy Audit — `data/tables.json`

Ground-truth source: every `Figure N:` caption in `nvme_spec/NVMe_spec_full.pdf` (extracted with `pymupdf`), plus direct page-by-page content comparison for ~20 spot-check tables. Reference: BUILD_PLAN_FINAL.md §1.2. Page offset note: `pdf_page` is intended to be a direct PDF page index (not printed + 24). Section-type coverage classified by regex on caption text.

## Summary stats

| Metric | Value |
|---|---|
| Tables parsed (`tables.json`) | **717** |
| Distinct `Figure N:` captions in PDF body (pp. 25–784) | **820** |
| Missing figures (in PDF, not parsed) | **103** (of which ~80 are diagrams, not tabular; ~1–15 are genuine data tables — most notably **Fig 199**) |
| Extra figures (in parsed, not in PDF) | 0 |
| Duplicate `figure_number` values in parsed | 0 |
| `figure_number` present | 717 / 717 (100%) |
| `caption` present | 717 / 717 (100%) |
| `printed_page` present | 717 / 717 (100%) |
| `pdf_page` present | 717 / 717 (100%) |
| `headers` non-empty | 716 / 717 (99.9%) |
| `rows` non-empty | 716 / 717 (99.9%) |
| `raw_text` present | 717 / 717 (100%) |
| **`parent_section` / `section_id` present** | **0 / 717 (0%)** — field does not exist in schema |
| Row-arity mismatches (row length ≠ header length) | **1 008 / 4 491 rows (22.4%)**, across **184 / 717 tables (25.7%)** |
| Single-cell "orphan" rows (group headers / merged cells) | **341** |
| Tables where first header token re-occurs >2× in `raw_text` (multi-page suspects) | 22 |
| Tables with "Notes:" footnote text in `raw_text` | 89 |
| Tables where those "Note(s):" footnotes also appear in parsed `rows` | 84 (94% of above) |

### Coverage by section type (heuristic on caption)

| Section type | Parsed tables | Notes |
|---|---|---|
| Command / CDW layouts | **155** | e.g., `Command Dword 0`, `Submission Queue Entry Format`, `Opcodes for…` |
| Log pages | **65** | `Get Log Page …`, `Log Page Identifiers`, `Log Page Support Requirements` |
| Register maps | **39** | `Offset Xh: …`, property definitions |
| Identify structures | **8** | Identify Controller, Power State Descriptor, Namespace Identification Descriptor, etc. |
| Feature definitions | **3** | `Get Features – Feature Identifiers`, `Set Features – Feature Identifiers`, Boot Partition state defs |
| Other / support requirement tables / status tables / descriptors | 447 | |

All five plan-required section types are represented. The low count for "identify" (8) and "feature" (3) is an artifact of my caption regex — many identify sub-structures are captioned just with the data-structure name (e.g., Fig 329 "Power State Descriptor Data Structure") and count correctly; and feature attribute tables (Figs 404–465) were counted in "other." Actual coverage of these categories is good.

## Spot-check table (20 entries)

`rows` column shows parsed row count vs. authoritative PDF row count (where a discrepancy was observable); `pdf_page Δ` is parsed value minus actual first-occurrence of `Figure N:` in the body.

| Fig | Caption (parsed) | Type | pdf_page Δ | rows parsed / PDF | Verdict |
|---|---|---|---|---|---|
| 2 | Decimal and Binary Units | other | −1 | 5 / ~2 | **WRONG** — headers inflated to spurious 2nd line ("(base-10)", "(base-2)"), rows collapsed into 4 malformed entries |
| 28 | Admin Command Support Requirements | other | −1 | 71 / ~63 entries | extra rows are single-cell "group" headers (e.g., "Logical Block Device Management Commands") — captured but not distinguished |
| 31 | Log Page Support Requirements | log_page | −1 | 53 | OK (spot-checked 3 rows — opcodes, M/O columns align) |
| 32 | Feature Support Requirements | feature | −1 | 49 | OK |
| 36 | Offset 0h: CAP – Controller Capabilities | register | −1 | 18 / 18 bit-ranges | OK (multi-page: PDF pp 78–81 correctly stitched; bit ranges 63:62 → 15:00 all present) |
| 41 | Offset 14h: CC – Controller Configuration | register | −1 | 10 / 10 | OK |
| 42 | Offset 1Ch: CSTS – Controller Status | register | −1 | 9 / 9 | OK |
| 91 | Command Dword 0 | command_cdw | −1 | 5 / 5 | OK (pp 159–160 stitched) |
| 94 | Fabrics Command – Submission Queue Entry Format | command_cdw | −1 | 5 | OK |
| 103 | Status Code – Command Specific Status Values | other | −1 | 69 | **WRONG for ≥1 row**: row 5 = `['04h', '', 'Reserved']`; correct per PDF is Description="Reserved", Commands Affected="" — column mis-alignment on merged/empty cells (see issue #4) |
| 142 | Opcodes for Admin Commands | command_cdw | −1 | 55 | OK (spot-checked 5 opcodes) |
| 146 | Abort – Command Dword 10 | command_cdw | -- | 2 | OK |
| 198 | Get Features – Feature Identifiers | feature | −1 | 51 | **WRONG** — last 4 rows (`['Bits', 'Description'], ['31:3', 'Reserved'], …`) are actually Figure **199** ("Completion Queue Entry Dword 0 when Select is set to 11b") silently concatenated. Fig 199 is absent from `tables.json`. |
| 199 | — | — | — | — | **MISSING** — merged into Fig 198 |
| 200 | Get Log Page – Data Pointer | log_page | −1 | 1 | OK (small table) |
| 202 | Get Log Page – Command Dword 11 | log_page | −1 | 2 | OK |
| 206 | Get Log Page – Log Page Identifiers | log_page | -- | 49 | OK (spot-checked 5 LIDs) |
| 328 | Identify – Identify Controller Data Structure, I/O Command Set Independent | identify | −1 | 193 | OK on row count; M/O flags for I/O, Admin, Disc preserved as separate columns — strong example |
| 329 | Identify – Power State Descriptor Data Structure | identify | −1 | 33 / 33 bit-ranges | OK (pp 377–380 stitched) |
| 403 | Set Features – Feature Identifiers | feature | 0 | 61 | OK |
| 625 | Set Features Boot Partition Write Protection State Definitions | feature | 0 | 4 / 3 | **WRONG HEADERS** — parsed as 5 cols `["State", "Definition", "Power Cycles", "Persistent Across", "Controller"]`, actual 4 cols `["State", "Definition", "Persistent Across Power Cycles", "Controller Level Resets"]`; the row `['Level Resets']` is a garbled header remnant |

## Systematic issues

1. **No parent section / section_id field at all.** BUILD_PLAN §1.2 explicitly requires "Each table captures: figure number, caption, **parent section ID**, column headers." The only attributes in every record are `figure_number`, `caption`, `printed_page`, `pdf_page`, `headers`, `rows`, `raw_text` — there is no linkage back to the TOC (`toc.json` section number). Containment per §1.4 ("figures belong to sections") is therefore not expressible from this file alone; downstream code will need to synthesize it from `pdf_page` or from adjacency scanning. **High-impact gap.**

2. **Multi-page tables that got wrong-figure contamination.** The parser's multi-page stitching logic is mostly correct (Figs 36, 41, 42, 91, 328, 329 all stitched cleanly across 2–15 pages), but at least one boundary is mis-detected: **Fig 199** was silently subsumed into Fig 198, inflating Fig 198's row count and completely losing Fig 199 as a standalone entry. This pattern should be checked systematically by diffing expected-vs-parsed row counts against every Figure header reprinted on a page break.

3. **Garbled / split headers.** When header cells themselves wrap over two text lines in the PDF (e.g., "Persistent Across / Power Cycles"), the parser splits them into two or more separate header cells and pushes the second line into the first data row. Confirmed for Fig 625; likely also the cause of `headers` with anomalous widths 1, 3, 5, 7, 8, 9, 17 (see Header-widths distribution). **~10–20 tables probably affected.**

4. **Row/column mis-alignment on blank or merged cells.** In Fig 103 row 5, "Reserved" was pushed into the `Commands Affected` column because the `Description` cell for opcode `04h` is the merged/empty value "Reserved" with a blank `Commands Affected`. The parser has no anchor for which column is empty when a value collapses. A full sweep would likely find dozens of such silent mis-assignments in reserved-slot rows. Corpus-level signal: **22.4 % of rows (1 008 / 4 491) have length ≠ their table's header count**, across **25.7 % of tables**.

5. **Bit-field tables collapse "valid values" and "mandatory/optional" into the Description blob.** The BUILD_PLAN row schema demands `{field name, byte/bit offset, size, valid values, mandatory/optional, description}`. In `tables.json`, 260 / 288 bit-field tables have only 2 columns (`Bits`, `Description`). Inline enumerations (e.g., Fig 36 row 60:59 has an embedded `Bits/Description` sub-table for CRMS) live as unstructured prose inside the `Description` string — 126 rows have at least one such inline Value/Definition sub-table. Valid-values and M/O are extractable only by downstream parsing. Register-map tables (width-4 with `Bits, Type, Reset, Description`) handle this better; identify-structure tables (width-5 with separate I/O, Admin, Disc M/O columns, e.g., Fig 328) handle it best.

6. **Single-cell "group header" rows are not distinguished from data rows.** Tables like Fig 198 and Fig 328 contain rows with one string ("Attributes Returned", "Controller Capabilities and Features") that act as spec-level section dividers inside the table. There are **341** such orphan rows. They contribute to the arity-mismatch count and will create confusion in any downstream field index (they look like rows but carry no offset/size).

7. **`pdf_page` is systematically one page early.** For ~80 % of spot-checked tables, the stored `pdf_page` points to the page where the figure is *referenced* in prose ("…as specified in Figure 36…"), not the page where `Figure 36:` actually renders. Magnitude: typically −1. Not catastrophic, but any "open PDF at this page" UX will land the user on the wrong page for most tables.

8. **1 empty table.** Fig 817 ("Write after Write") has `headers=[]` and `rows=[]` but a populated `raw_text`. Fig 15 ("Complex NVM Storage Hierarchy…") is a diagram, not a table, but was still captured with empty headers — questionable inclusion.

9. **~1 real table missing outside Fig 199.** Fig 29 ("Fabrics Command Support Requirements", p70) appears in the PDF with 7+ rows (opcodes 7Fh, 01h, 04h, 05h, 06h, 08h, …) and is absent from `tables.json`. A slower content-aware sweep could turn up a handful more from the 103 missing-figure set — I counted ~1–15 candidates, the rest are diagrams. This is a low-severity but worth-fixing cluster.

## Accuracy estimate

- **Schema coverage (required per §1.2):**
  - figure_number, caption, column headers, rows, raw_text: **100 %** (all records)
  - parent section ID: **0 %** (missing by design)
  - Per-row (field name, bit/byte offset, size, valid values, M/O, description): **~10–20 %** of bit-field rows expose valid-values and M/O as structured columns; the remaining ~80 % embed them inside the `Description` string. This matches the plan only for register-map and identify-structure tables; bit-field (CDW/log-page) tables fall short of the schema promise.
- **Row-level structural integrity:** `77.6 %` of rows have length equal to their table's header width; the remaining 22.4 % are either group-header orphans (acceptable but un-tagged) or outright column mis-alignments (real errors).
- **Figure-level coverage:** **717 / ~730 real data tables present** (≈ 98 % if the ~90 missing-but-actually-diagram figures are excluded). Only ~1–2 genuine content tables are known missing (Fig 29, Fig 199).
- **Multi-page stitching:** **~5 of 6 spot-checked multi-page tables stitched correctly**; ≥1 boundary failure (198/199) confirmed.
- **Caption and identifier accuracy:** 100 % on spot-checks.
- **`pdf_page` accuracy:** 100 % present; **~20 % exactly correct, ~80 % off by 1**. Non-fatal but user-visible.

**Overall usable accuracy: ~85–90 % at the table level, ~75–80 % at the row level.** This is meaningfully below what the plan calls for ("every table field extracted correctly, every cross-reference captured, no misattributed content"). Tables will support retrieval well but should be treated as a *structured reliable base only for register-map and identify-structure tables*; CDW/log-page bit-field rows still require downstream parsing of the Description blob before fields.json can be fully populated.

Confidence: **Medium-high.** 15-table deep spot-check + whole-corpus statistics; did not exhaustively re-parse every figure, so issues #3 and #4 magnitudes are extrapolated rather than enumerated.

## Recommended fixes (priority order)

1. **Add `section_id` (or `parent_section`) to every record.** Compute by walking `toc.json` and selecting the deepest section whose `(page, level)` precedes the table's `pdf_page`. Blocks §1.4 containment and §1.6 card generation otherwise.
2. **Detect and split merged figures.** When a figure's row stream contains a second `"Figure N+1:"` token, cut there. The Fig 198/199 collision is the obvious example; a regex sweep over `raw_text` for each record will find any others.
3. **Fix multi-line header parsing.** When consecutive lines both look like column names (short, title-case, no digits), concatenate them before treating subsequent lines as data. Fixes Fig 625 class of issues.
4. **Classify bit-field tables into a richer schema.** For tables with `Bits`/`Description` only, post-process the Description string to extract: `field_abbr` (parenthesized all-caps), `valid_values` (embedded mini-tables), and mandatory/optional markers ("shall be set", "Impl Spec"). Emit these as explicit row attributes rather than leaving them in prose.
5. **Correct `pdf_page` to point at the figure's caption page.** Current value usually lands one page before the caption. Fix: after parsing, relocate each figure by searching for `Figure <n>:` starting at the stored page.
6. **Flag orphan rows.** Rows with length 1 that are not empty should get a `row_kind: "group_header"` annotation rather than being commingled with data rows.
7. **Reconcile the 103-figure gap.** Confirm which are diagrams (OK to skip), and add the ~1–15 genuine tables currently missing (at minimum: Fig 29 Fabrics Command Support Requirements, Fig 199 Completion Queue Entry Dword 0 when Select=11b).
8. **Drop or explicitly mark non-table figures.** Fig 15 and similar diagram-only figures should either be excluded from `tables.json` or carry a `is_diagram: true` flag.
