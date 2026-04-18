# specGPT Data Effectiveness Report

Synthesis of the per-artifact audits in this directory. Sources: `01_plan_goals.md`, `02_toc_accuracy.md`, `03_tables_accuracy.md`, `04_fields_and_relationships_accuracy.md`, `05_prose_definitions_llm_accuracy.md`, `06_cards_and_integrity.md`. Ground truth: `nvme_spec/NVMe_spec_full.pdf` (Base Spec 2.x) + `NVMe_spec_TOC.pdf`. Plan reference: `BUILD_PLAN_FINAL.md` Phase 1.

## The one-line verdict

**The parser clears the Week-1 bar but misses the §1.8 exit bar.** Data is usable enough to drive Phase-2 indexing experiments, but it is not yet "trustworthy across the spec" — misattribution, missing cross-references, and silent content loss are all present in measurable amounts. Ship Phase-1 fixes before locking the graph and embeddings.

## Scoreboard

| Artifact | Usable accuracy | Confidence | Plan bar met? |
|---|---|---|---|
| `toc.json` | 90% title+page / 80% full | High | **No** — hierarchy drift in §3, dup `5.3.1`, phantom children |
| `tables.json` | ~85–90% table-level, ~75–80% row-level | Med-High | **No** — no `parent_section`, row mis-alignment, Fig 199 lost inside Fig 198 |
| `fields.json` | ~98% | High | **Yes** |
| `field_index.json` | High internally; **misses 13/27 controller registers** (CAP, CC, CSTS, AQA, ASQ, ACQ, …) | High | **Partial** — bit-level only |
| `prose.json` | ~85% retrieval / ~90% normative | Med-High | **No** — L2 parent intros systematically dropped (§3.7, §4.2, §8.1 etc.); 15% short-fragment noise |
| `definitions.json` | ~95% on parsed terms / 93% coverage | High | **No** — §1.7 dropped, so `LBA` is missing |
| `relationships.json` (deterministic) | ~97% | Med-High | **Yes** — 0 self-loops, 0 dup, 3.9% orphan endpoints |
| `relationships_llm.json` | 55–70% as-is, ~85% after reconciliation | Medium | **No** — no canonicalization, 48 type vocabs, `related_to` junk-drawer |
| `cards.json` | 100% schema / **35% empty summaries** | High | **No** — summary/keywords missing on 364/1041; `cards_state` claims done |

## The five findings that matter most

1. **TOC bugs propagate downstream into 20–25% of cards.** The §3.2/§3.3 hierarchy drift, duplicate `5.3.1`, and phantom annex children documented in `02_toc_accuracy.md` are copy-forwarded into `cards.json` titles, break `section_id` uniqueness, and pollute parent/child trees. The single highest-leverage fix in the entire parser is rebuilding `toc.json` from `fitz.get_toc()` with `PAGE_OFFSET = 24`.

2. **`tables.json` has no `parent_section` field at all.** The plan explicitly requires every table to carry a parent section ID. 0/717 records have one. This blocks §1.4 containment and §1.6 card generation from being derivable from `tables.json` alone — `cards.tables` is currently populated only for 203/717 figures (71.7% undercount) because the joining is being guessed from pages.

3. **Content loss in L2 parent sections.** When an L2 heading is followed by an L3, the parent's intro prose gets absorbed into the first child and the L2 entry is left empty. Confirmed silent drops: §3.7 Resets, §4.2 Completion Queue Entry, §8.1, §B.5, and ~10 more L2s. Several of these carry multi-sentence *normative* intros — those `shall`s are invisible to the normative index.

4. **LLM relationship extraction skipped reconciliation.** The plan asked for a "global reconciliation pass: merge duplicate entities, standardize names." It did not run, or ran ineffectively: 165 entity cores collide across namespaces/case, 48 relationship types are in use (including typos like `defines_in` next to `defined_in` and `configured by` with a literal space), and 20% of `returned_by`/`requires` edges are direction-inverted. The deterministic edges (`relationships.json`) are fine; the LLM set needs a post-processing pass before it feeds the graph.

5. **`cards.json` summaries are 35% empty but `cards_state.processed` claims 100%.** 364 cards have `summary: ""` and identical `keywords: []`, concentrated at L3/L4/L6. L6 is 10/10 empty. The state file is lying — this will look "done" to any resume-from-checkpoint logic. The cost to finish is small (free tier is fine) but until it is run, ~35% of retrieval context is impoverished.

## Effectiveness by Phase-1 exit criteria

Plan §1.8 demands three things. Current state:

| §1.8 requirement | Status | Evidence |
|---|---|---|
| "every table field extracted correctly" | **Partial** | 77.6% of rows align with their header; 22.4% are arity-mismatched (group headers 341, column mis-alignment unknown) |
| "every cross-reference captured" | **Partial** | Deterministic set covers 4820 edges with ~97% correctness and 3.9% orphan endpoints; LLM set adds 2758 edges but at 55–70% usable quality. No corpus-level recall metric exists. |
| "no misattributed content" | **No** | Fig 199 subsumed into Fig 198; §3.7 / §4.2 prose dropped; §3.2.1.1 title shows the register-offset; dup 5.3.1 collapses content; 20% LLM direction errors |

The bar is "parser is not done until output is trustworthy across the spec." By this definition, the parser is not done.

## Effectiveness for Phase 2 (Knowledge Graph + Indexing)

Despite the Phase-1 gaps, Phase-2 work can productively start on what *is* solid:

**Safe to index as-is:**
- `fields.json` — clean, parents resolve, offsets plausible.
- `definitions.json` — small, high-fidelity, worth embedding immediately.
- `relationships.json` (deterministic) — well-formed, confidence-tagged, ready to load into NetworkX.
- `tables.json` raw_text and register/identify structures — strong enough to embed.

**Needs fixes before indexing or it will poison retrieval:**
- `toc.json` — rebuild from bookmarks; fixes propagate into `cards.json`, `prose.json`, `relationships.json` orphans.
- `prose.json` — capture L2 intros, filter <20 char paragraphs, recompute `end_pdf_page`.
- `tables.json` — attach `parent_section`, split the Fig 198/199 merge, handle multi-line headers, classify bit-field rows.
- `cards.json` — generate the missing 364 summaries; fix `cards.tables` to reflect all 717 figure containments.
- `relationships_llm.json` — run reconciliation, constrain type enum, drop/downgrade `related_to`, invert passive-voice direction errors.

## Recommended sequencing (highest ROI first)

1. **Rebuild `toc.json` from PDF bookmarks** (2–3 hours). Unblocks downstream integrity and eliminates ~20–25% card corruption.
2. **Add `parent_section` to every row in `tables.json`** (trivial once toc is solid). Unblocks `cards.tables` population and §1.4 containment.
3. **Fix L2 intro capture in `prose.json`** (1–2 hours). Recovers dropped normatives from major section intros.
4. **Finish card summary generation** (cheap — Haiku on 364 cards, a few dollars). Closes the 35% empty-summary gap.
5. **Post-process `relationships_llm.json`**: canonicalize entities, dedup, filter junk-drawer types, invert passive-voice directions. Lifts usable accuracy from ~60% to ~85% without re-calling the LLM.
6. **Split merged figures (Fig 199) and fix multi-line headers** in `tables.json`. Row-level accuracy climbs from ~78% to ~90%+.
7. **Extend `definitions.json` to cover §1.6 + §1.7.** One regex pass. Recovers `LBA` and 7 others.
8. **Expand `field_index.json` to include register-level acronyms** from `parent_caption`. Recovers CAP, CC, CSTS, AQA, ASQ, ACQ, INTMS, INTMC, NSSR, CMBLOC, CMBSZ, BPINFO, CMBMSC, SQyTDBL, CQyHDBL.

Items 1–4 are the critical path. Items 5–8 are polish that meaningfully improves retrieval quality but does not block a Phase-2 prototype.

## Overall effectiveness rating

**B− / 72%** — weighted average of artifact accuracies, penalized for the §1.8 exit-bar misses and for content that is silently lost rather than explicitly flagged.

The foundation is real. The parser understood the spec's structure well enough to extract 717 tables, 1623 fields, 4820 deterministic relationships, and 2536 normatives with >90% fidelity on the happy path. What's missing is the *validation-and-reconciliation* half of Phase 1 — the part the plan flagged as "validate obsessively" and "global reconciliation pass." Finishing items 1–5 above would push this to an A− and genuinely clear the plan's exit bar.

Confidence in this assessment: **Medium-High.** Per-artifact audits used deep spot-checks (15–30 samples each) plus whole-corpus stats; cross-artifact integrity was computed on full sets. No exhaustive re-parse of the 800-page PDF was performed, so recall-style metrics ("every cross-reference captured") are extrapolated from samples.

## Is this mission-critical for the MVP?

**Partially.** The MVP per the plan is Phase 3: a live site where you type a question and get a cited answer with eval scores. Here's the honest split between what blocks that and what just degrades it.

### Mission-critical for the MVP (will break the demo or destroy credibility)

1. **TOC rebuild** — the core value prop is *cited* answers. If citations link to "§3.2.1.1 Offset E18h: PMRMSCU" when the real title is "Namespace Overview," the demo looks broken the moment anyone cross-checks. ~60 drifted entries in §3, plus dup 5.3.1, plus phantom annex children. **Must fix before the demo.**
2. **`cards.tables` regeneration** — 514 of 717 figures are not linked into any card. Any question of the form "show me the fields in CAP" relies on this lookup. Without it, the retrieval path from section → figure → fields is broken for 72% of tables. **Must fix before the demo.**
3. **Dup `5.3.1`** — lookups by section number are ambiguous on a high-traffic Admin-command path. Cheap one-line fix; leaving it means random wrong answers. **Must fix before the demo.**

Total work for the three blockers: **~1 day.** Do these and the data is demo-safe.

### Degrades MVP quality but does NOT block the demo

- **Empty card summaries (364 cards, 35%)** — retrieval still works from prose chunks + tables; summaries sharpen recall but aren't the carrier. Nice to fix, cheap to fix, not blocking. Eval scores will be a few points lower.
- **L2 parent intros dropped in `prose.json`** — affects ~13 major sections. Children still carry the content; the intro's `shall`s are silently absent from normative queries. Will surface as specific wrong answers on a small number of questions; eval will catch them. Fix in Phase 2 polish.
- **Fig 199 lost into Fig 198** — one specific feature (Completion Queue Entry when Select=11b) will answer wrong. Bounded blast radius.
- **LLM relationship noise (~25% direction/hallucination)** — filterable. For MVP, either (a) only load the deterministic set into NetworkX or (b) downgrade `related_to` edges. Don't block on a full reconciliation pass.
- **`LBA` missing from `definitions.json`** — BM25 + vector search still hit the §1.7 prose that defines it. Users get a correct answer, just not from the glossary lookup path.
- **Register-level acronyms missing from `field_index.json`** (CAP, CC, CSTS, …) — same story; BM25 catches them in `tables.json` captions.
- **Row-level mis-alignment in 22% of table rows** — most are single-cell group-header orphans that are acceptable noise in retrieval context. The ~dozens of genuine column-misalignment errors are localized.

### Not needed for MVP at all

- LLM entity canonicalization and type-vocab constraint
- §1.6 / §1.7 definition expansion
- Symbolic-offset typing in `fields.json`
- Multi-line header reparse in `tables.json`

### Recommended MVP-cut path

Do the three blockers (TOC rebuild, cards.tables regeneration, 5.3.1 dedup), then ship Phase 2 indexing. Capture the remaining items as post-MVP eval-driven fixes — let the eval set in §2.3 tell you which ones actually hurt answer quality on real questions instead of pre-optimizing. Plan's own risk table frames this correctly: "Focus on lookup and structural questions first — high accuracy with good parsing. Procedural accuracy improves in Phase 4."

**Bottom line:** The data quality is not mission-critical for a *working* MVP, but the three TOC-level bugs are mission-critical for a *credible* one. Fix those three things, ship, let the eval set prioritize the rest.

---

## TODAY'S PLAN

Four tasks for Claude. Ordered by dependency — each one consumes the output of the previous. Last task is a full audit re-run.

### 1. Rebuild `toc.json` from PDF bookmarks and propagate downstream

- Replace the current TOC builder with `fitz.get_toc()` against `nvme_spec/NVMe_spec_full.pdf`, using `PAGE_OFFSET = 24` to map bookmark page numbers to PDF indices.
- Validate the new TOC against `NVMe_spec_TOC.pdf`: no §3 hierarchy drift, no phantom annex children, no title pollution like "§3.2.1.1 Offset E18h: PMRMSCU", and dup `5.3.1` resolved to two distinct `section_id`s.
- Re-run the downstream stages that derive from TOC so fixes propagate: `cards.json` titles + `section_id`s + parent/child trees, `prose.json` section assignments, and the `relationships.json` orphan endpoints (currently 3.9%).
- Done when: `toc.json` matches the PDF bookmarks, `cards.json` has no duplicate `section_id`s, relationship orphan rate drops below 1%.

### 2. Fix `tables.json` containment and regenerate `cards.tables`

- Add `parent_section` to every row in `tables.json` by joining on page range against the rebuilt `toc.json`. All 717 rows must carry it.
- Regenerate `cards.tables` from the now-attributed tables. Currently 203/717 figures are linked into cards (28.3%); target 717/717.
- While in this stage, split the Fig 198/199 merge if cheap (one table is currently subsuming the other). If expensive, flag it and skip — not a blocker today.
- Done when: `parent_section` is non-null on 100% of tables, every figure appears in exactly one card's `tables` array, and the §1.4 containment rule ("fields belong to structures, structures belong to sections") is derivable from `tables.json` alone.

### 3. Fill the 364 empty card summaries and fix `cards_state`

- Identify all cards where `summary == ""` (expect 364, concentrated at L3/L4/L6; L6 is 10/10 empty).
- Run Haiku over each with the card's prose + table content as input, generating both `summary` and `keywords` in one call. Budget: a few dollars on free/cheap tier.
- Reset `cards_state.processed` to reflect reality — currently claiming 100% while 35% are empty. Any resume-from-checkpoint logic should see truthful state afterward.
- Done when: 0 empty summaries, 0 empty keyword arrays, `cards_state` accurately reports completion.

### 4. Review — full audit re-run and scoreboard update

- Re-execute the spot-checks that produced the current audits in `02_toc_accuracy.md` through `06_cards_and_integrity.md` on the new artifacts. Same sample sizes (15–30 per artifact), same methodology, so deltas are comparable.
- Recompute whole-corpus integrity stats: card `section_id` uniqueness, figure-to-card linkage coverage, relationship orphan rate, empty-summary count, `parent_section` coverage on tables.
- Update the Scoreboard table in this file with new numbers, and add a **Delta** subsection showing what moved (artifact-by-artifact, before → after).
- Re-evaluate the §1.8 exit bar — specifically whether "every table field extracted correctly," "every cross-reference captured," and "no misattributed content" now clear the "trustworthy across the spec" threshold.
- Done when: all six per-artifact reports are updated with the post-fix numbers, the top-level scoreboard reflects the new state, and the verdict at the top of this file is rewritten if warranted.

**Out of scope today:** L2 prose intro recovery, LLM relationship reconciliation, definition expansion (§1.6/§1.7), register-acronym index expansion. Those are post-demo polish — let the eval set decide the order.
