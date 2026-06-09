# NVMe Corpus Figure Coverage Audit

Audited: 2026-06-08. Live queries against Supabase plus PDF scans with PyMuPDF
against all three source PDFs in `nvme_spec/`.

---

## Summary

All three NVMe spec corpora were affected by a shared `FIRST_CONTENT` page-index
constant hardcoded to `24` in `src/tables.py`. The Base spec legitimately starts
body content at pdf index 24, so its cutoff was correct. The PCIe and Command
Set specs start body content at pdf index 5 and 7 respectively, so every figure
in those specs was silently dropped by the extractor.

After the fix (`SPEC_FIRST_CONTENT` env variable, now set to `7` for the Command
Set corpus), the Command Set corpus gained 17 newly tabular figures. The PCIe
corpus remains unre-extracted with `FIRST_CONTENT=24` still active (all 74
figures absent). The Base spec is unaffected by the fix.

Across all three specs, the largest categories of missing figures are:

- **Front-matter cutoff (bug):** 18 figures cut from Command Set pre-fix; all 74
  PCIe figures still cut today.
- **Non-tabular content (by design):** diagrams, flowcharts, and bit-field
  layouts that `pymupdf.find_tables()` cannot detect. 79 Base figures, 3 Command
  Set figures, and figures 1-2 plus 8 in PCIe fall into this category.
- **Caption-detection miss (extractor gap):** a small number of tabular figures
  on pages where `find_tables()` finds a table but `_find_caption_above()` does
  not match it to a Figure number. Observed in both Base (24 cases) and Command
  Set (12 cases above page 24).

The highest-impact single gap was Command Set Figure 11 (Protection Information
Field Definition), which was cut by `FIRST_CONTENT=24` while 75 downstream
chunks in the Command Set corpus cross-reference it by name. That figure is now
present in `spec_tables` post-fix.

---

## Root Cause

`src/tables.py` CLI hardcoded:

```python
FIRST_CONTENT = int(os.getenv("SPEC_FIRST_CONTENT", "24"))
```

The default `24` matched the Base spec, where the List of Figures occupies pdf
pages 9-22 and body content starts at page 24. For the PCIe spec (44 pages
total, body from page 5) and the Command Set spec (158 pages, body from page 7),
every single body page fell below the cutoff.

The fix introduced the `SPEC_FIRST_CONTENT` environment variable so each spec
can override the default. The Command Set re-ingestion used `SPEC_FIRST_CONTENT=7`.

---

## Per-Spec Coverage

### Base Spec (`NVMe_spec_full.pdf`, `spec='base'`)

| Metric | Value |
|--------|-------|
| PDF page count | 784 |
| Body content starts at pdf index | 24 (List of Figures occupies pages 9-22) |
| FIRST_CONTENT setting | 24 (correct) |
| Figures in PDF (body pages) | 820, range 1-820 |
| Figures in `spec_tables` | 717, range 2-819 |
| Total missing | 103 |
| Missing: non-tabular diagrams | 79 |
| Missing: caption-detection miss | 24 |
| Missing: front-matter cutoff | 0 |

The 103 missing Base figures are **all on body pages at or above index 24**. The
cutoff was not the cause. Breakdown:

- **79 non-tabular:** Figures on pages with zero tables detected. These are
  architecture diagrams (Figs 1, 4-5, 8-26), queue/flowchart illustrations
  (Figs 66, 69-81, 86-90), state machine models (Figs 619-688, 701, 706, 714,
  728-739, 755-820 range), and similar graphics. `pymupdf.find_tables()` finds
  no table because the "figure" is an embedded raster or drawn vector graphic,
  not a bordered table grid.
- **24 caption-detection miss:** Figures on pages where `find_tables()` does find
  tables, but `_find_caption_above()` did not match the Figure N caption to the
  correct table. Examples: Fig 3 (p.28, 3 tables), Fig 6 (p.44, 1 table), Fig 29
  (p.69, 3 tables), Figs 67-68 (p.103, 1 table), Fig 199 (p.229, 1 table).
  The most likely causes are captions more than `CAPTION_LOOKUP_DY=80pt` above
  the table, or the caption text block belonging to a sibling figure on the same
  page.

Notable cross-referenced gaps in Base: Figs 66-90 are architectural overview
figures cited in the §2.x introduction sections. Their absence does not break
structured lookups but reduces context richness for broad "how does NVMe work"
questions.

### PCIe Transport Spec (`NVMe_PCIe_full.pdf`, `spec='pcie'`)

| Metric | Value |
|--------|-------|
| PDF page count | 44 |
| Body content starts at pdf index | 5 |
| FIRST_CONTENT setting | 24 (incorrect, entire spec is cut) |
| Figures in PDF (body pages) | 74, range 1-74 |
| Figures in `spec_tables` | 31, range 44-74 |
| Total missing | 43 |
| Missing: front-matter cutoff (pages 5-23) | 43 |
| Missing: non-tabular diagrams | 2 (Figs 1, 8) |
| Missing: caption-detection miss | 0 |

The PCIe spec is 44 pages long. With `FIRST_CONTENT=24`, only pages 24-43 were
ever scanned. Those 20 pages contain Figures 44-74 (31 figures). Figures 1-43
all fall on body pages 5-23 and were completely skipped. The re-ingestion fix was
applied only to the Command Set corpus; the PCIe corpus has not yet been
re-extracted with the corrected cutoff.

All 43 missing PCIe figures do exist on body pages 5-23 with tables present.
Exceptions: Figure 1 (NVMe Family of Specifications, pdf page 5, diagram) and
Figure 8 (Command Processing, pdf page 12, flowchart) are non-tabular and will
remain absent even after re-extraction with the correct cutoff.

The 41 extractable missing PCIe figures include critical register tables:
Figs 3-7 (PCI Express registers and controller properties, Doorbell registers,
command processing), Figs 10-43 (the full PCI configuration space from ID
register through MSI-X). These are the figures most likely to matter for PCIe
transport questions.

### Command Set Spec (`NVMe_command_full.pdf`, `spec='command'`)

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| FIRST_CONTENT | 24 | 7 |
| Figures in PDF (body pages) | 179, range 1-179 | 179 |
| Figures in `spec_tables` | 133, range 19-179 | 149, range 2-179 |
| Total missing | 46 | 30 |
| Missing: front-matter cutoff | 20 | 0 |
| Missing: non-tabular (persistent) | 3 (Figs 1, 8, 41) | 3 |
| Missing: caption-detection miss | 12 | 12 |
| Missing: List-of-Figures only | 11 | 0 |

Figures newly added by the fix (17 tabular, gained from pages 7-24):
2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20.

Figures still absent after fix:

- **Non-tabular diagrams (3):** Fig 1 (NVMe Family of Specifications, pdf p.7),
  Fig 8 (Atomic Boundaries Example, pdf p.17), Fig 41 (Source LBA and
  Destination LBA Relationship Example, pdf p.37).
- **Caption-detection miss, on pages with tables (12):** Figs 149, 150, 151,
  153, 155, 158 (Protection Information format tables on pp.130-136), plus
  Figs 166-175 (PI processing flowchart/table pages on pp.140-150). These pages
  have 1-2 tables each, but `_find_caption_above()` matched a different figure
  caption on the same page, leaving these figures unattributed.
- **Non-tabular diagrams on body pages above 24 (15):** Figs 137-144 (namespace
  alignment diagrams, pp.120-125), Figs 147-148 (16b Guard PI format bit-layout
  illustrations, p.129), Figs 166-174 (PI processing flow diagrams, pp.140-146),
  Fig 175 (LBA Format List Structure, p.150).

Current gaps in `spec_tables` after fix: [1, 8, 20, 41, 137-144, 147-151, 153,
155, 158, 166-175].

---

## The Figure 11 Case

Figure 11 (Protection Information Field Definition) is the master definition of
the PRINFO / PRACT field used by Write, Read, and Copy commands. It appears at
pdf page 19 (body) in the Command Set spec.

Before the fix: Figure 11 was cut by `FIRST_CONTENT=24`. It was absent from
`spec_tables`. Yet 75 spec chunks in the Command Set corpus cross-referenced it
by name, and 33 chunks directly use `PRINFO` or `PRACT` field names. Every
structured lookup for PRINFO bit definitions silently retrieved no master
definition row, forcing the LLM to reconstruct semantics from prose alone. This
was the recurring blind spot for PRINFO/PRACT questions.

After the fix: Figure 11 is confirmed present in `spec_tables` with `figure_number=11`
and the correct tabular content. The 75 cross-referencing chunks now resolve
against the canonical field definition table.

---

## Fix Applied

File: `src/tables.py`, CLI block:

```python
# Before: hardcoded
FIRST_CONTENT = 24

# After: env-overridable
FIRST_CONTENT = int(os.getenv("SPEC_FIRST_CONTENT", "24"))
```

Correct `SPEC_FIRST_CONTENT` values per spec:

| Spec | PDF | Correct first_content_page_idx | Notes |
|------|-----|-------------------------------|-------|
| base | `NVMe_spec_full.pdf` | 24 | List of Figures is pages 9-22. Default is correct. |
| pcie | `NVMe_PCIe_full.pdf` | 5 | List of Figures is pages 3-4. Needs re-extraction. |
| command | `NVMe_command_full.pdf` | 7 | List of Figures is pages 3-6. Fix applied. |

The page offset (pdf_idx - printed_page) for each spec:

| Spec | page_offset |
|------|-------------|
| base | 23 |
| pcie | 4 (pdf p.5 = printed p.1) |
| command | 6 (pdf p.7 = printed p.1) |

---

## Recommendations

### 1. Re-extract PCIe corpus immediately (high priority)

Run the ingestion pipeline for `spec='pcie'` with `SPEC_FIRST_CONTENT=5`. This
recovers 41 extractable figures including the complete PCI configuration space
register tables (Figs 3-43) that underpin every PCIe transport compliance
question. Only Figs 1 and 8 will remain absent as non-tabular content.

### 2. Re-extract Base corpus with targeted re-scan (medium priority)

The Base spec's `FIRST_CONTENT=24` is correct, so no cutoff issue exists. The
24 caption-detection misses are an extractor accuracy gap. The most actionable
fix is to increase `CAPTION_LOOKUP_DY` beyond 80pt for pages where a figure
caption sits farther above its table, or to add a fallback that searches the
full page text for an unmatched Figure N caption when a table has no attribution.

The 79 non-tabular Base figures (diagrams, state machines) cannot be recovered
by the table extractor. Capturing them would require prose or image chunking:
either OCR the page into a prose chunk tagged `content_type='figure_prose'`, or
store the rendered page region as an image embedding. Both are future work.

### 3. Resolve Command Set caption-detection misses (medium priority)

Figures 149-151, 153, 155, 158 (PI format tables on pp.130-136) are tabular but
uncaptured because their figure captions map to the wrong table on a shared page.
Inspecting those pages shows that each page contains two figures: one whose
caption is within 80pt of a table, and one whose caption is farther away or
shared across a page break. Increasing `CAPTION_LOOKUP_DY` or adding
same-page caption disambiguation would recover these.

### 4. Non-tabular PI processing figures (informative)

Command Set Figs 137-144 and 166-174 are diagrams (namespace alignment
illustrations and PI processing flowcharts). They are legitimately non-tabular.
Their captions and surrounding prose are already present in `spec_chunks` as
prose content. No action is needed unless image-based retrieval is added.

### 5. Caption-detection miss watchlist

The following Base figures are tabular (tables on page) but not ingested and are
likely cross-referenced: Figs 3 (Byte/Word/Dword), 29 (Fabrics Command Support),
67-68 (NVM Sets), 199 (Completion Queue Entry Dword 0), 664 (FDP Model), 715,
738-739, 755, 758, 761, 763, 766. These are candidates for a targeted re-run
after the caption-detection fix.
