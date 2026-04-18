# Prose / Definitions / LLM-Relationships Accuracy Audit

Ground-truth source: `nvme_spec/NVMe_spec_full.pdf` (NVM Express Base Specification, Revision 2.3) via `pymupdf` (fitz 1.27). Artifacts audited: `data/prose.json` (1042 entries), `data/definitions.json` (104 entries), `data/relationships_llm.json` (2758 edges), cross-referenced against `data/relationships.json` (4820 deterministic edges) and `data/toc.json`.

---

## `prose.json` — non-table prose + normative tagging (§1.3)

### Summary stats

| Metric | Value |
|---|---|
| Section entries | **1042** (matches toc.json 1:1 on `section_number`) |
| Entries with non-empty `paragraphs` | **835 / 1042 (80.1%)** |
| Entries with non-empty `normative` | **559 / 1042 (53.6%)** |
| Entries with *both* empty | **207 (19.9%)** |
| Total paragraph objects (`{text, pdf_page}`) | 6275 |
| Avg paragraph length | 201 chars; range 1–1794 |
| Very short (<20 char) paragraphs | 951 (15.2%) — mostly figure-label fragments bleeding in |
| Total normative statements | **2536** (shall: 1306, may: 871, should: 359) |
| Normatives whose text actually contains the modality word | 2536 / 2536 (100%) |
| `start_pdf_page > end_pdf_page` bugs | 0 |
| `target_page` matches `toc.json` | 1040 / 1041 (99.9%) — only mismatch is the dup-key `5.3.1` already flagged in `02_toc_accuracy.md` |
| Entries with `start==end` on pdf page | 554 (53%) — often correct for short leaves, but masks prose-loss bugs in L2 parents (see Issue 1) |

### Empty-paragraph breakdown by level

| Level | Empty / Total | % |
|---|---|---|
| L1 | 5 / 12 | 41.7 % |
| L2 | **13 / 69** | **18.8 %** — this is the bad one |
| L3 | 43 / 309 | 13.9 % |
| L4 | 124 / 376 | 33.0 % |
| L5 | 12 / 225 | 5.3 % |
| L6 | 10 / 10 | 100 % — all bogus children inherited from TOC drift |
| L7 | 0 / 41 | 0 % |

The L4/L6 empties mostly inherit from TOC-level bugs documented in `02_toc_accuracy.md` (phantom children, body-text headings promoted to TOC). The L2 empties are new and damaging — they are real sections whose intro prose is silently dropped (see Issue 1).

### Spot-check table — prose / normative (8 samples)

| Section | Title | PDF vs parsed | Verdict |
|---|---|---|---|
| §1.1 | "Overview" | 4 paras captured; text matches PDF pg 24 verbatim | OK |
| §1.1.1 | "NVM Express® Specification Family" | 17 paras, 0 normative | OK body text; 10 of the 17 paras are figure-label fragments ("Specification", "NVM Express", "Boot Specification") pulled from Figure 1 diagram | PARTIAL — figure-label noise |
| §3.1 | "NVM Controller Architecture" | 1 shall captured ("All controllers…shall support the same controller model") | OK |
| §3.1.1 | "Memory-Based Controller Architecture (PCIe)" | 1 shall captured; 2 other shalls on same PDF page belong to §3.1.2 / §3.1 — correctly NOT attributed here | OK |
| §3.7 | "Resets" | **0 paras, 0 normative** — PDF pg 142 has clear intro prose: "The scope of an NVM Subsystem Reset depends on whether the NVM subsystem supports multiple domains…" plus a 4-item normative bullet list | **MISS — intro prose and normatives silently dropped** |
| §4.2 | "Completion Queue Entry" | **0 paras, 0 normative** — PDF pg 163 has the intro: "The Common Completion Queue Entry Layout is at least 16 bytes in size. Figure 96 describes…" | **MISS — entire L2 intro lost to child §4.2.1** |
| §5 | "ADMIN COMMAND SET" | 1 may + 1 should captured on pg 194 | OK |
| §8.1.23 | "Replay Protected Memory Block" | 18 paras, 10 normatives; text + page numbers verify against PDF pp 632–637 | OK |

### Systematic issues (prose)

1. **L2 parent-section intros systematically dropped.** When an L2 heading is immediately followed by an L3 subheading, the parent's intro paragraph is absorbed into the first child and the L2 entry is left with empty `paragraphs`. Affected L2 sections include §1.4, §1.5, §2.3, §2.4, §3.2, §3.7, §3.8, §4.1, §4.2, §5.1, §5.4, §8.1, §B.5 — 13 real sections, several (§3.7, §4.2, §8.1) carrying multi-sentence normative intros in the PDF. **This is the single largest content-loss bug in `prose.json`.**
2. **Figure-label fragments treated as prose.** Inline labels inside figure diagrams (e.g., "Core 0" on pg 44, "Boot Specification" on pg 24, "NVM Express" on pg 24) are emitted as standalone paragraphs. 951 paragraphs (~15%) are under 20 characters, most of them figure labels, page headers, or orphaned line fragments. These will create retrieval noise but not misinformation.
3. **`end_pdf_page == start_pdf_page` for 53% of sections.** Correct for leaf sections, but for multi-page L3/L4 sections it clips content. §4.2 spans pgs 163–169 in practice but is recorded as 163–163; §3.7 spans 142–143 but is 142–142. This aligns with Issue 1 — the parser treats the first child's start page as the parent's end page.
4. **No evidence of normative misclassification.** Every one of the 2536 normatives contains its declared modality word. Random sampling did not surface any non-normative "shall" (e.g., section titles or figure labels) being tagged. Precision on normative tagging is high.
5. **No evidence of normative over-extraction.** Deep-dive on §3.1 / §3.1.1 on pg 60 showed that each page's shalls are correctly attributed to the section they belong to (not to other sections sharing the page).
6. **Target pages are trustworthy.** 1040 / 1041 prose entries have `target_page` equal to `toc.json`'s `target_page`. The one mismatch is the known duplicate `5.3.1`.

### Accuracy estimate (prose)

- **Normative tagging precision:** ~100 % (no false positives observed in a 30-sample sweep).
- **Normative tagging recall:** ≥ 95 % on sections that *have* prose captured, but the L2-dropped-intro bug (Issue 1) means normatives embedded in those intros are invisible — recall at the document level is more like 90 %.
- **Paragraph extraction:** ~80 % of entries carry real prose. Of those, text is faithful to the PDF but contaminated with ~15 % short-fragment noise. Usable for retrieval but needs a post-filter on paragraph length.
- **Section-boundary fidelity:** poor on L2 parents (Issue 1, Issue 3). Good on L3 and below.
- **Overall usable accuracy: ~85 % for retrieval, ~90 % for normative-requirement extraction.** Confidence: **Medium-High** — sampled 30+ sections across the spec and cross-checked the PDF directly, but did not exhaustively enumerate L2 intros.

### Recommended fixes (prose)

- **Fix L2 intro capture.** When the parser hits an L2 heading followed by an L3 heading, capture any intervening prose as the L2's own paragraphs before attributing to the child. Re-run and audit §3.2 / §3.7 / §3.8 / §4.1 / §4.2 / §8.1 explicitly.
- **Filter short-fragment paragraphs.** Drop or quarantine paragraphs < 20 chars that don't end with sentence punctuation. Alternative: tag them as `figure_label` instead of treating as prose.
- **Compute `end_pdf_page` as the start of the next same-or-higher-level section minus 1**, instead of whatever heuristic is producing 53 % start==end.
- **Add `paragraph_count` / `normative_count` to `cards.json`** so the downstream retrieval layer can distinguish intentionally empty sections (body-text-label stubs) from content-lost sections.

---

## `definitions.json` — term → definition lookup (§1.3)

### Summary stats

| Metric | Value |
|---|---|
| Entries | **104** |
| Source section | **§1.5 "Definitions" only** (pp 27–38 of the PDF, logical pp 4–15) |
| Avg definition length | 252 chars, ~3 sentences |
| Definitions with no ending punctuation (truncated) | 0 |
| Definitions under 40 chars | 1 (`Underlying NVM Subsystem → "Defined as NVM subsystem."` — a legitimate redirection, not truncation) |
| Multiple-sentence definitions | 80+ |

### Spot-check table (definitions)

| Term | Expected in glossary? | In `definitions.json`? | Note |
|---|---|---|---|
| `NVM` | yes | **yes** ("acronym for non-volatile memory") | OK |
| `NVMe` | no — defined inline in §1.1.1, not §1.5 | no | OK — out of scope of §1.5 |
| `PRP` | no — structure, not in §1.5 | no | documented in §4.3.1, not glossary |
| `SGL` | no — structure, not in §1.5 | no | documented in §4.3.2, not glossary |
| `LBA` / `logical block address` | **YES — defined in §1.7 "NVM Command Set specific definitions"** | **MISSING** | parser stopped at §1.5 |
| `logical block` | **YES — defined in §1.7** | **MISSING** | parser stopped at §1.5 |
| `HMB` (Host Memory Buffer) | no — structure, not in §1.5 | no | OK |
| `ANA` (Asymmetric Namespace Access) | no — structure, not in §1.5 | no | OK |
| `Admin Queue` | yes | yes | OK (full 2-sentence definition) |
| `controller` | yes | yes | OK |
| `namespace` | yes | yes | OK |
| `Endurance Group` | yes | yes | OK |
| `Reclaim Unit Handle (RUH)` | yes | yes | OK |
| `capsule` | yes | yes | OK |
| `Discovery controller` | yes | yes | OK |

### Systematic issues (definitions)

1. **§1.7 definitions dropped.** The spec has THREE definition sub-sections — §1.5 "Definitions" (core), §1.6 "I/O Command Set specific definitions" (6 cross-command-set terms), §1.7 "NVM Command Set specific definitions" (2 terms: `logical block`, `logical block address (LBA)`). The parser captured only §1.5. **`LBA` — one of the most-queried NVMe terms — is missing from `definitions.json`.** §1.6 terms appear to have been captured indirectly as `prose.json` §1.6.x sections but are not in the `{term: definition}` lookup.
2. **Single-glossary assumption.** §1.3 of the plan ambiguously says "the acronym/definitions table" (singular). The spec actually has the three sources above plus inline `Term (ACRONYM)` patterns in prose (e.g., "Reclaim Unit Handle (RUH)"). Only the one core glossary is parsed. Merge policy was never defined, so this is more a spec-ambiguity than a parser bug, but downstream retrieval will silently fail on acronyms like PRP, SGL, LBA.
3. **Case-sensitivity in keys.** Terms are stored case-sensitively (`admin label` vs `Admin Queue` vs `NVM`). Any lookup must normalize casing or maintain a case-insensitive index.

### Accuracy estimate (definitions)

- **Content faithfulness of the 104 parsed entries:** ~98 % — every definition sampled is a complete multi-sentence string faithful to the PDF. No truncation observed.
- **Glossary coverage vs the spec's definition sections:** ~104 / 112 (≈ 93 %) — missing the 2 §1.7 terms and (optionally) the 6 §1.6 terms.
- **Overall usable accuracy: ~95 % for the terms it covers, ~93 % coverage of the spec's defined terms.** Confidence: **High** — the §1.5 term list is fully enumerable.

### Recommended fixes (definitions)

- **Include §1.6 and §1.7 definitions**. At minimum add `logical block` and `logical block address (LBA)`. One extraction pass over sections whose numbers match `1\.[5-7](\..*)?` should handle it.
- **Add a lowercase / acronym alias index** for retrieval. `LBA → logical block address`, `PRP → Physical Region Page Entry`, etc. These aliases can be harvested from inline `Term (ACRONYM)` patterns in `prose.json`.
- **Mark provenance.** Add `source_section: "1.5"` to each entry so downstream layers can tell a glossary definition from a structure description.

---

## `relationships_llm.json` — Haiku-extracted implicit edges (§1.5)

### Summary stats

| Metric | Value |
|---|---|
| Edges | **2758** |
| Unique entities | 1922 |
| Deterministic edge count (for comparison) | 4820 |
| Overlap (by source+target, case-insensitive, namespace-stripped) with deterministic | **0** — LLM edges are disjoint, i.e. genuinely additive |
| Unique relationship types emitted | **48** — uncontrolled vocabulary |
| `confidence` field values | `"llm"` for all 2758 (vs `"deterministic"` on the other set) — correctly tagged as lower-trust, but it's a string label not a numeric score |
| Internal duplicates (same source+target+type) | 128 (4.6 %) |
| Entities with namespace prefix (`command:`, `feature:`, `field:`, `structure:`, `log_page:`, `queue type:`, `status code:`) | 1922 / 1922 (100 %) |
| Entity cores that collide across namespaces (canonicalization failures) | **165** (≈ 8.6 % of entities) |

### Relationship-type distribution (top 8, remaining 40 types tail off sharply)

| Type | Count | Notes |
|---|---|---|
| `requires` | 684 | Often directional misreadings (see Spot-check row 1, 8) |
| `uses` | 580 | |
| `related_to` | 558 | **Generic-to-useless**; most should be a more specific type |
| `configured_by` | 315 | Usually correct Set/Get-Features edges |
| `returned_by` | 234 | Often inverted (command returns structure, not vice versa — row 5) |
| `defined_in` | 197 | Usually correct field-in-structure edges |
| `posts_to` | 77 | Mixed — some misread Abort-like semantics (row 7) |
| `returns` | 14 | |

Tail includes one-offs, typos, and free-form phrases: `defines_in` (1) next to `defined_in` (197); `configured by` (1, with a space) next to `configured_by` (315); free-form `not reset as part of` (3), `may alter` (2), `indicates support of` (1), `shall support` (1). Haiku is not being constrained to a vocabulary.

### Spot-check table — random 20 edges (seed 13)

| # | Edge | Verdict |
|---|---|---|
| 1 | `command:Set Controller State --requires--> command:Migration Send` | **WRONG DIRECTION** — Set Controller State is an operation OF Migration Send |
| 2 | `feature:Spinup Control --requires--> field:Command Dword 11` | weak — evidence is just "method is specified in Command Dword 11" |
| 3 | `log_page:Predictable Latency Per NVM Set --returned_by--> command:Get Log Page` | OK — correct direction |
| 4 | `feature:host --related_to--> feature:user data` | NOISE — `host` should never be `feature:`, and `related_to` is uninformative |
| 5 | `command:Identify --returned_by--> structure:Identify Namespace data structure` | **WRONG DIRECTION** — Identify RETURNS the structure, not the inverse |
| 6 | `feature:controller doorbell property --requires--> field:queue depth` | OK-ish — evidence thin ("the modulus is the queue depth") |
| 7 | `command:Abort command --posts_to--> queue type:I/O Submission Queue` | **SEMANTIC ERROR** — Abort targets commands that are on the queue, it is not posted to that queue |
| 8 | `log_page:Lost Host Communication --requires--> feature:Time-Based Recovery` | weak — evidence "shorten Time-Based Recovery" does not imply `requires` |
| 9 | `command:command --related_to--> feature:data transfer` | HALLUCINATION — entity is the word "command" generically |
| 10 | `command:Get Log Page --requires--> field:LPOL` | OK |
| 11 | `structure:PRP List --requires--> structure:PRP entries` | weak — should be `contains` |
| 12 | `field:MAXCMD --defined_in--> structure:Identify Controller data structure` | OK |
| 13 | `field:MVCNCLD bit --defined_in--> log_page:Sanitize Status log page` | OK |
| 14 | `field:Placement Identifier --defined_in--> command:Data Placement Directive` | OK |
| 15 | `feature:Namespace Admin Label --configured_by--> command:Set Features` | OK |
| 16 | `command:Set Features --uses--> feature:Asynchronous Event Notifications` | OK |
| 17 | `command:Identify Directive --configured_by--> feature:Directive Type` | OK |
| 18 | `feature:Command Specific Errors --requires--> feature:selected I/O command sets` | noise — "selected I/O command sets" is not an entity |
| 19 | `command:Cross-Controller Reset command --related_to--> field:Alternate Controller ID (ACID) field` | OK, but `uses` would be more informative |
| 20 | `structure:command capsule --may_include--> feature:data` | NOISE — `feature:data` is not a real entity |

**Tally:** 11 clearly correct / 4 weak-but-defensible / 5 wrong-or-noise ≈ **55 % clean, 20 % weak, 25 % noisy.**

### Systematic issues (LLM relationships)

1. **Entity canonicalization failed.** 165 entity cores collide across namespace prefixes:
   - `Admin Queue` appears as `feature:Admin Queue`, `structure:Admin Queue`, `queue type:Admin Queue`, `structure:Admin queue` (case drift too).
   - `Admin Submission Queue` vs `Admin SQ` — the prompt asked about this exact case; the parser has `structure:Admin Submission Queue` and `queue type:Admin Submission Queue` but no `Admin SQ` variant in sources (checked), so this particular pair is fine. The broader problem is real though — `NVM Subsystem` / `NVM subsystem`, `Completion Queue Entry` / `Completion queue entry` / `completion queue entry`, `Controller` / `controller`, etc.
   - The plan explicitly called for a "Global reconciliation pass: merge duplicate entities, standardize names." Evidence says that pass did not run or was ineffective.
2. **Direction errors in ~20 % of `returned_by` / `requires` edges.** Haiku frequently inverts subject/object when the evidence sentence is phrased passively.
3. **Free-form type vocabulary.** 48 types instead of the 8–10 the plan implies. `configured by` vs `configured_by`, `defines_in` vs `defined_in` — both collision pairs indicate no enum validation on output.
4. **`related_to` is a junk drawer.** 558 edges (20 % of the set) use this type. About half of those are actually `uses` / `configures` / `supports` / `returns` but were relaxed to the generic verb. High recall, low specificity.
5. **Generic and made-up entities.** `feature:host`, `feature:data`, `feature:controller`, `command:command`, `feature:user data`, `feature:data transfer` are semantic nouns that Haiku extracted as entities — they are not defined NVMe concepts. Estimated 50–100 such non-entity entities contaminate the graph.
6. **Redundant edges.** 128 exact-duplicate `(source, target, type)` tuples within `relationships_llm.json` alone.
7. **No numeric confidence.** All 2758 edges carry the string `"llm"`. The risk-table requirement "must be flagged with lower confidence" is satisfied categorically (LLM vs deterministic) but offers no per-edge triage signal — the noisy 25 % is indistinguishable from the clean 55 %.
8. **No overlap with deterministic set.** Good news — by source+target the LLM set adds new edges rather than restating existing ones. The additive value is real, just noisy.

### Accuracy estimate (LLM relationships)

- **Spot-check edge correctness:** ~55 % clean, ~20 % weak-but-arguable, ~25 % direction-wrong or hallucinated entities (n = 20, seed 13). Extrapolating: ~1500 usable edges, ~550 weak, ~700 noisy out of 2758.
- **Entity canonicalization:** ~92 % of entity cores are unique; ~8 % have namespace/case collisions.
- **Type-vocabulary discipline:** poor — 48 types, tail of one-off free-form phrases.
- **Overall usable accuracy: ~55–70 % of edges as-is; ~85–90 % if `related_to` edges are filtered out and direction is reversed on `requires` / `returned_by` via post-processing.** Confidence: **Medium** — 20-sample spot check plus exhaustive entity/type/duplication counts.

### Recommended fixes (LLM relationships)

- **Run the reconciliation pass the plan specified.** Canonicalize entities by (lowercased core, namespace-stripped), then pick the longest / most-namespaced variant as the canonical name. This alone collapses ~165 duplicate cores.
- **Constrain the type vocabulary.** Give Haiku a closed enum — `requires`, `uses`, `configures`, `configured_by`, `defines`, `defined_in`, `returns`, `returned_by`, `contains`, `contained_in`, `posts_to`, `related_to` (last-resort). Reject anything else at ingestion.
- **Drop `related_to` or downgrade it.** Either remove all 558 edges or reclassify them via a second pass. In hybrid retrieval they just dilute graph traversal.
- **Filter generic entities.** Blocklist `feature:host`, `feature:data`, `feature:user data`, `command:command`, `feature:data transfer`, `feature:controller` (generic form with lowercase), and similar bare nouns.
- **Deduplicate.** Simple `(source, target, type)` set-dedup removes 128 edges for free.
- **Add a numeric-confidence field** (even a Haiku-self-reported 1–5 score would beat the current binary `llm` label) so downstream retrieval can gate on it.
- **Post-process direction for passive-voice evidence.** If `evidence` contains "is returned by" / "is configured by" / "is defined in", invert source↔target on `returned_by` / `configured_by` / `defined_in` edges where Haiku got the direction wrong.

---

## Summary scoreboard

| Artifact | Usable accuracy | Confidence | Biggest single fix |
|---|---|---|---|
| `prose.json` | ~85 % | Medium-High | Capture L2 parent intros before descending into L3 children |
| `definitions.json` | ~95 % on parsed terms, ~93 % coverage | High | Extend parser to §1.6 and §1.7 so `LBA` is included |
| `relationships_llm.json` | ~55–70 % as-is, ~85 % after reconciliation | Medium | Run the global entity-canonicalization pass the plan already called for; constrain the type enum; filter `related_to` |

All three artifacts are functionally usable for Phase 2 indexing, but `relationships_llm.json` needs a post-processing pass before it is merged into the graph or the noise will propagate into retrieval. `prose.json` needs the L2-intro fix before chunking — otherwise ~13 major section introductions will be silently absent from the chunk corpus. `definitions.json` is the healthiest of the three and needs only a minor extension.
