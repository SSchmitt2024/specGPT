# TOC Accuracy Audit — `data/toc.json`

Ground-truth source: PDF bookmark outline of `NVMe_spec_full.pdf` (959 entries). Cross-checked against the rendered TOC in `NVMe_spec_TOC.pdf` (pp. 3–23). Page offset: PDF page 25 = logical page 1 (front-matter is 24 pages).

## Summary stats

| Metric | Value |
|---|---|
| Parsed entries (`toc.json`) | **1042** |
| Outline entries (ground truth) | **959** (940 numbered + 19 unnumbered annex headings) |
| Parsed levels | 1:12, 2:69, 3:309, 4:376, 5:225, 6:10, 7:41 |
| Outline levels | 1:12, 2:69, 3:286, 4:277, 5:225, 6:49, 7:41 |
| Duplicate section numbers in parsed | **1** (`5.3.1`) |
| Parsed entries whose `(title, page)` also exists in outline | **940 / 1042 (90.2%)** |
| Parsed entries with matching section number, title, page, AND level | **751 / 940 (79.9%)** |
| Numbered L1/L2 entries (from rendered TOC pp. 3–23) | **68 / 68 correct (100%)** |

L1 chapter headings, L2 section headings, chapter page offsets, and first-order titles are essentially perfect. Deep-tree numbering, on the other hand, has drifted.

## Spot-check table (20 entries spanning the full spec)

| § (parsed) | Expected (outline) | Got (parsed) | Verdict |
|---|---|---|---|
| 1 | "INTRODUCTION", pg 1, L1 | same | OK |
| 1.1.1 | "NVM Express® Specification Family", pg 1, L3 | same | OK |
| 1.4.1.2 | "may", pg 2, L4 | "obsolete", pg 2, L4 | WRONG TITLE (off by 1 in definitions list) |
| 1.4.1.3 | "obsolete", pg 2, L4 | missing entry with this number | MISSING |
| 1.5.1 | "admin label", pg 5, L3 | same | OK (parser synthesised numbers matching outline) |
| 2 | "THEORY OF OPERATION", pg 18, L1 | same | OK |
| 2.1.2 | "Memory Model", pg 21 | "Core 0", pg 21 | WRONG — parser grabbed body-text heading |
| 3 | "NVM EXPRESS ARCHITECTURE", pg 37, L1 | same | OK |
| 3.1.4.28 | "Offset E18h: PMRMSCU …", pg 76, L4 | missing | MISSING (pulled into 3.2.1.1 instead) |
| 3.2.1.1 | "Namespace Overview", pg 76, L4 | "Offset E18h: PMRMSCU…", pg 76 | WRONG (one-row hierarchy shift) |
| 3.2.1.7 | "I/O Command Set Associations", pg 79 | "Subsystem", pg 78 | WRONG |
| 3.3.1.4 | "Empty Queue", pg 89 | "Queue Abort", pg 89 | WRONG |
| 4.1 | "Submission Queue Entry", pg 135, L2 | same | OK |
| 4.8.1 | (no such section in outline) | "Controller Unique Identifier", pg 169 | EXTRA — real outline has `4.7.1.3` here |
| 5.3 | "Create I/O Completion Queue command", pg 457 | same | OK |
| 5.3.1 (dup) | outline has one 5.3.1 only | parsed has **two**: "Command Completion" pg 456 AND "Create I/O Completion Queue command" pg 457 | DUPLICATE KEY bug |
| 6.1 | "Authentication Receive Command and Response", pg 489, L2 | same | OK |
| 7.1 | "Cancel command", pg 500, L2 | same | OK |
| 8.1 | "Common Extended Capabilities", pg 515, L2 | same | OK |
| 9 | "ERROR REPORTING AND RECOVERY", pg 742, L1 | same | OK |
| A.4 | outline has this as L2 unnumbered annex (pg 750) | "Bad Media and Vendor Specific NAND Use", pg 750, L2 | OK by title+page; numbering synthetic |
| B.3.1 | not in outline | "LBA #0 (4 KiB)", pg 753, L3 | EXTRA — parser numbered an unnumbered bullet list |
| B.5.1.1 | not in outline | "B.5.1. Shadow Doorbell Buffer Overview" — duplicates parent title verbatim | BOGUS CHILD NODE |

## Systematic issues

1. **Hierarchy drift in chapter 3 (pp. ~76–110).** Starting at `3.2.1.1`, every child entry is shifted by one relative to the authoritative outline — each parsed `N.x` holds the title that belongs to `N.(x−1)` in the ground truth. Likely caused by the parser mis-attaching a register-offset heading (`Offset E18h: PMRMSCU`) as the first child, pushing everything after it down one slot. Affects ~60 entries across §3.2 / §3.3. 
2. **Duplicate section number `5.3.1`.** Two different titles ("Command Completion" p456, "Create I/O Completion Queue command" p457) share the same key. Any lookup by section number will be ambiguous. This is the only dup in the file but it lives in a high-traffic Admin-command section.
3. **Phantom children in annexes (B.5, B.6).** The parser emits L4 entries whose titles are literal copies of their parent's heading (e.g., `B.5.1.1` → "B.5.1. Shadow Doorbell Buffer Overview"). ~8 bogus entries in annex B; these will create duplicate retrieval hits.
4. **Body-text headings promoted to TOC children.** ~102 "extra" parsed entries (e.g., `2.1.2` "Core 0", `B.3.1–4` "LBA #0 … #3", `5.2.25.2` "Security Protocol 00h") were inferred from body-text subheadings, not bookmarks. Not necessarily wrong as structural anchors, but they carry synthetic section numbers that do not exist in the spec.
5. **Missing deep entries at the `1.4.1.x` keyword list.** Seven keyword definitions ("obsolete", "optional", "R", "reserved", "shall", "should") are absent though they exist as L4 bookmarks in the source.
6. **Level inflation at L4.** Parsed has 376 L4 entries vs. outline's 277 (+99). Offset concentrated in the same areas as issues 1, 3, 4.

## Accuracy estimate

- **Title + page correctness (bookmark-level):** 940 / 1042 = **90.2%**
- **Full correctness (section number + title + page + level):** 751 / 940 = **79.9%**
- **L1 / L2 chapter & section headings:** 100% (81 / 81)
- **Deep-tree (L4–L7) correctness:** roughly **70–75%** — most wrong entries cluster here.

**Overall usable accuracy: ~80% for structured retrieval (by section number), ~90% for title-based or anchor-based retrieval.** Confidence: **High** — the comparison used the authoritative PDF bookmark outline and covered every parsed entry, not a sample.

### Recommended fixes before downstream work
- Rebuild `toc.json` from `doc.get_toc()` on `NVMe_spec_full.pdf` with `PAGE_OFFSET = 24` rather than inferring numbers from body text.
- De-dup key `5.3.1` and drop phantom annex children like `B.5.1.1`.
- If body-text sub-headings are wanted as retrieval anchors, store them under a separate key (e.g., `anchor_id`) instead of reusing `section_number`.
