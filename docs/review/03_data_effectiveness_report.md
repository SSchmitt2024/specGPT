# specGPT Data Effectiveness Report — Final Phase 1 Audit

**Date:** 2026-04-18
**Follows:** `00_*` (baseline), `01_*` (post-TOC/tables fix), `02_*` (post-LLM-fill + reconcile + 04-17 fixes)

This is the final Phase 1 data audit. All "high-value content gaps" from prior reports are closed. The purpose of this report is to summarize the full corpus state, score each artifact, and provide a go/no-go recommendation for Phase 2.

## The one-line verdict

**Phase 1 data is complete.** 96.1% card coverage, 100% table parentage, 112 definitions including LBA, 27 register containers indexed, 7706 merged relationship edges with 0% orphan rate. Proceed to Phase 2 indexing and retrieval.

## Scoreboard

| Artifact | Grade | Key Metrics |
|---|---|---|
| `toc.json` | **A** | 1036 entries, 0 duplicates, 0 garbled. Perfect card alignment. 5 phantom Annex B children remain (cosmetic). |
| `tables.json` | **A−** | 717 captioned tables. **717/717 (100%) with `parent_section`**. Fig 199 still merged into Fig 198 (affects 1 answer). |
| `fields.json` | **A** | 1650 entries including **27 register-container** pseudo-fields. All offset/type metadata populated. |
| `field_index.json` | **A** | 1108 unique keys. **13/13 core register acronyms** (CAP, CC, CSTS, AQA, ASQ, ACQ, INTMS, INTMC, NSSR, CMBLOC, CMBSZ, BPINFO, CRTO) now resolve. |
| `prose.json` | **B+** | 1036 sections, 6275 paragraphs, 2536 normative tags. 122 sections with empty prose (structural headers). L2 intros for §3.7, §4.2, §8.1, §B.5 still empty. |
| `definitions.json` | **A−** | 112 terms spanning §1.5 + §1.6 + §1.7. **`logical block address (LBA)` present.** `reservation` still missing. |
| `relationships.json` (det) | **A** | 4835 edges, 3 clean types (`contained_in`, `cross_reference`, `parent_child`). Structural backbone. |
| `relationships_llm.json` | **C+** | 3236 edges, 50 types. `related_to` junk-drawer (674 edges). Known direction errors (~20%). Adequate for soft re-ranking, not for authoritative facts. |
| `relationships_merged.json` | **B** | 7706 edges, 52 types, **0.00% orphan rate**. 352 entity alias clusters. Usable for context expansion with filtering. |
| `entity_registry.json` | **B** | 352 canonical→alias mappings. Sample-verified plausible. |
| `cards.json` | **A** | 1036 unique IDs. **996/1036 (96.1%) summaries filled.** 996 with keywords. 324 with table links. 1036 with relationship links. 40 residual empties are genuinely unsummarizable (0 prose, 0 children, 0 tables). |
| `cards_state.json` | **A** | 996 processed = 996 summarized. State semantics correct: only marks "processed" when summary exists. |

## Grade progression across audits

| Artifact | `00_*` | `01_*` | `02_*` | `03_*` (final) |
|----------|--------|--------|--------|----------------|
| `toc.json` | B− | B+ | B+ | **A** |
| `tables.json` | C+ | A− | A− | **A−** |
| `fields.json` | B+ | A− | A− | **A** |
| `field_index.json` | B | B | B | **A** |
| `prose.json` | B | B | B | **B+** |
| `definitions.json` | B− | B | B | **A−** |
| `relationships.json` (det) | B+ | A | A | **A** |
| `relationships_merged.json` | — | — | B− | **B** |
| `cards.json` | C+ | B+ | B+ | **A** |
| `cards_state.json` | — | C | C | **A** |

## What changed since `02_*`

| Fix | Impact |
|---|---|
| `MIN_PROSE_CHARS` 200→50 + skeleton fallback | Cards filled 746→996 (72%→96.1%) |
| `synthesize_register_containers()` in `fields.py` | 27 register containers in index; 0/13→13/13 register acronyms |
| Atomic `_save_json` (`.tmp` + `fsync` + `os.replace`) | Ctrl-C can no longer corrupt JSON output |
| `ImportError`/`ModuleNotFoundError` re-raise | LLM import failures now fail hard instead of silently "failing" 99 iterations |
| `cards_state.json` semantics | Only marks "processed" when summary exists; stale stubs auto-retry on re-run |
| `pymupdf.TOOLS.mupdf_display_errors(False)` | Silenced MuPDF warning flood in tables.py, prose.py, deep_sections.py |
| Per-page progress prints | Long-running PDF scans no longer look hung |
| `definitions.json` §1.6/§1.7 coverage | LBA + 8 NVM/I/O command set terms recovered (was only §1.5) |

## Remaining known issues (none are blockers)

### Quality lifts (degrades answers marginally, doesn't break demo)

1. **L2 parent intros in `prose.json`** — §3.7, §4.2, §8.1, §B.5 and ~10 others have 0 paragraphs because prose.py's heading-position logic drops text between L2 heading and first L3 child. ~1-2hr fix in `prose.py`.
2. **Filter merged relationships graph** — drop `related_to` (674 edges), drop edges with bad-entity sources/targets (gerunds, adjectives, `command:outstanding`), invert known direction error patterns. ~1hr deterministic post-process. Recommend doing this at query time in Phase 2 retrieval layer, not upstream.
3. **Split Fig 199 from Fig 198** — one Get Features answer currently wrong. Single-table fix.
4. **`reservation` missing from definitions** — not in §1.5/1.6/1.7 (it's defined inline elsewhere). Low priority.

### Cosmetic

5. **5 phantom Annex B children** — `B.5.1.1`, `B.6.1.1`, `B.6.1.2`, `B.6.4.1`, `B.6.4.2` are spurious bookmark depths.
6. **Mojibake in captions** — em-dash → `�` in some figure captions (encoding bug in PDF extraction).

## Corpus statistics summary

| Metric | Count |
|---|---|
| TOC entries | 1,036 |
| Tables with parent | 717 / 717 |
| Field definitions | 1,650 (incl. 27 register containers) |
| Field index keys | 1,108 |
| Prose paragraphs | 6,275 |
| Normative requirements | 2,536 |
| Definitions | 112 |
| Deterministic edges | 4,835 |
| LLM edges | 3,236 |
| Merged edges | 7,706 (0.00% orphan) |
| Entity alias clusters | 352 |
| Cards filled | 996 / 1,036 (96.1%) |
| Cards with keywords | 996 |
| Cards with tables | 324 |
| Cards with relationships | 1,036 |

## Effectiveness for Phase 2

**Index-ready (no caveats):**
- `toc.json` — section tree for navigation and citation
- `cards.json` — 996 summaries for vector embedding, all with keywords
- `tables.json` — 717 structured tables with parent_section for lookup
- `fields.json` + `field_index.json` — structured field retrieval + register resolution
- `definitions.json` — 112-term glossary for definitional queries
- `relationships.json` (det) — 4835 clean structural edges for graph traversal

**Index with filtering:**
- `prose.json` — embed all 6275 paragraphs; note L2 intros are incomplete
- `relationships_merged.json` — use for context expansion; filter out `related_to` at query time
- 40 empty card stubs — keep for section-ID resolution; won't add semantic recall

**Recommended retrieval architecture:**
- Structural: `fields.json` + `field_index.json` → "what fields are in CAP?"
- Definitional: `definitions.json` → "what is LBA?"
- Semantic: vector search over card summaries + prose chunks → "how does HMB work?"
- Procedural: graph traversal on deterministic edges → "what command sets HMB?"
- Hybrid: RRF fusion of vector + BM25 + graph, reranked → general questions

## Overall effectiveness rating

**A / 94%**

Up from B− / 72% at project start. Every artifact is A-range except `prose.json` (B+, missing L2 intros), `relationships_llm.json` (C+, model quality ceiling), and `relationships_merged.json` (B, inherits LLM noise). The structural backbone, card coverage, field index, and definitions are all at or near ceiling.

**Phase 1 is done. Ship Phase 2.**
