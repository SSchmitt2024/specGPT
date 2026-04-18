# specGPT Data Effectiveness Report — Post TOC-Rebuild

Follow-up to `00_data_effectiveness_report.md`. Captures state after Task #1 from that report's TODAY'S PLAN was executed (TOC rebuild + downstream propagation).

## The one-line verdict

**The structural backbone is now A-grade; remaining gaps are content/enrichment, not skeleton.** Citations are trustworthy, section identity is unique, figure attribution is complete. What's left is empty card summaries (one Haiku run away), missing `parent_section` on tables, an un-reconciled LLM relationships file, and a thin `field_index`.

## Updated scoreboard

| Artifact | Was | Now | Confidence | §1.8 bar met? |
|---|---|---|---|---|
| `toc.json` | 90% / "No" | **clean** — 959 entries from PDF bookmarks, 0 dups, 0 drift | High | **Yes** |
| `tables.json` | ~85% table / ~78% row | 717 captioned, **still no `parent_section`**, Fig 198/199 merge unfixed | Med-High | **No** |
| `fields.json` | ~98% | unchanged — clean | High | **Yes** |
| `field_index.json` | bit-level only, 13 register acronyms missing | unchanged — still missing CAP, CC, CSTS, AQA, ASQ, ACQ, INTMS, INTMC, NSSR, CMBLOC, CMBSZ, BPINFO, CMBMSC | High | **Partial** |
| `prose.json` | ~85% retrieval / ~90% normative, L2 intros dropped | 959/959 sections matched, **L2 intros still dropped** on ~13 sections | Med-High | **No** |
| `definitions.json` | ~95%, missing `LBA` | 104 terms; §1.6/§1.7 still uncovered, **`LBA` still missing** | High | **No** |
| `relationships.json` (deterministic) | ~97%, 3.9% orphan | 4753 edges, **1.47% orphan** — most remaining orphans are non-tabular figure refs | Med-High | **Yes** |
| `relationships_llm.json` | 55–70% as-is | unchanged — 48 type vocabs, `related_to` junk, 20% direction errors, 165 entity collisions | Medium | **No** |
| `cards.json` | 35% empty / dups / 28% figure linkage | 959 unique IDs, **100% figure linkage (717/717)**, 36% still empty summary | High | **Partial** (structure ✓, content gap) |
| `cards_state.json` | claimed 100% (lying) | truthful — 614/959 processed | High | **Yes** |

## Delta from `00_data_effectiveness_report.md`

| Metric | Before | After |
|---|---|---|
| TOC entries | 1041 (with phantoms/dups) | 959 (clean) |
| Duplicate `5.3.1` | yes | resolved |
| §3.2.1.1 title | "Offset E18h: PMRMSCU" | "Namespace Overview" |
| Card duplicate `section_id`s | present | 0 |
| Figure→card linkage | 203/717 (28.3%) | 717/717 (100%) |
| Relationship orphan rate | 3.9% | 1.47% |
| `cards_state` truthfulness | claimed 100%, was 65% | accurate |

## What was fixed (Task #1 from `00_*` plan)

1. `toc.json` rebuilt from `fitz.get_toc()` against `nvme_spec/NVMe_spec_full.pdf` with `PAGE_OFFSET = 24`.
2. `tables.json`, `prose.json`, `relationships.json` regenerated against the new TOC (deterministic; reproducible).
3. `cards.json` structural fields refreshed via new helper `scripts/refresh_cards_structural.py` — preserves existing summaries/keywords, drops 157 phantom/dup cards, adds 74 new structural stubs.
4. `cards_state.json` rewritten to reflect actual processed count (614).
5. Pre-rebuild backup at `data/cards_pre_toc_rebuild_backup.json`.

## What still needs to be fixed (priority order)

### Mission-critical for a credible MVP

1. **Fill the 345 empty card summaries** — one Haiku run, ~$2–3, ~10–20 min. Cards have correct structure now; just need text. Run `python -m src.llm.generate_cards` (resume logic kicks in automatically). This was Task #3 in the `00_*` plan.
2. **Add `parent_section` to every row in `tables.json`** — joins on page range against the rebuilt TOC. Trivial now that TOC is solid. Unblocks §1.4 containment derivable from `tables.json` alone. This was Task #2 in the `00_*` plan.

### Quality lifts (degrade answers but don't break the demo)

3. **Capture L2 parent intros in `prose.json`** — ~13 sections (§3.7, §4.2, §8.1, §B.5, …) silently drop multi-sentence normative intros. Loses `shall`s from the normative index. ~1–2 hours.
4. **Reconcile `relationships_llm.json`** — canonicalize entities, constrain type enum, drop/downgrade `related_to`, invert passive-voice direction errors. Lifts usable accuracy from ~60% to ~85% without re-calling the LLM. For MVP, you can sidestep this by loading only the deterministic set into the graph.
5. **Split Fig 198 / Fig 199 merge** — one specific feature (Completion Queue Entry when Select=11b) currently answers wrong.

### Enrichment (post-MVP polish)

6. **Extend `definitions.json` to cover §1.6 + §1.7** — recovers `LBA` and ~7 others. One regex pass.
7. **Expand `field_index.json` with register-level acronyms** from `parent_caption` — recovers CAP, CC, CSTS, AQA, ASQ, ACQ, INTMS, INTMC, NSSR, CMBLOC, CMBSZ, BPINFO, CMBMSC, SQyTDBL, CQyHDBL.
8. **Multi-line header re-parse in `tables.json`** — pushes row-level accuracy from ~78% to ~90%+.

## Effectiveness for Phase 2 (Knowledge Graph + Indexing)

**Safe to index now (clean):**
- `toc.json`, `cards.json` structure, `relationships.json` (deterministic), `fields.json`, `definitions.json`
- `tables.json` raw_text + register/identify structures (parent_section absence is a metadata gap, not a content gap)

**Index with caveats:**
- `prose.json` — fine to embed; just know L2-intro normatives are missing
- `cards.json` summaries — embed only the 614 with text; the 345 empty stubs will rely on prose+table chunks

**Do not index until reconciled:**
- `relationships_llm.json` — load deterministic edges only, or run reconciliation first

## Overall effectiveness rating

**B+ / 84%, leaning A−** — up from B− / 72% in the `00_*` report. The TOC rebuild was the highest-leverage single fix in the entire parser; it propagated cleanly into card identity, figure attribution, and relationship orphan rate. What's left is enrichment (summaries, definitions, register acronyms) and a single un-run reconciliation pass on the LLM relationships.

Confidence: **High** on structural metrics (computed on full corpus). Medium on retrieval-quality predictions until the empty summaries are filled and an eval set runs against the live index.

## MVP readiness

The three MVP blockers from `00_*`:
- ✅ TOC rebuild — done
- ✅ `cards.tables` regeneration — done (28% → 100%)
- ✅ Dup `5.3.1` resolved — done

**The data is now demo-safe.** Filling the 345 empty summaries is the next-highest-value action for actual answer quality, but the demo will not look broken without it. Recommended path: run card summaries, then ship Phase 2 indexing, then let the eval set in §2.3 prioritize the rest.
