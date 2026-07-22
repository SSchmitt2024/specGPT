# specGPT — Field Usage Reference

How each field in every data file is actually used in the product.

**Legend:**
- ✅ Core — actively used in retrieval or generation
- ✅ Used — stored and surfaced in results
- ⚠️ Stored only — written to DB but not read at query time
- ⚠️ Pipeline only — used during data processing, gone by runtime
- ⚠️ Partial — wired up but not fully exercised
- ❌ Not used — extracted but never read

---

## `toc.json`

| Field            | Status              | How                                                                                                     |
| ---------------- | ------------------- | ------------------------------------------------------------------------------------------------------- |
| `section_number` | ✅ Core             | Becomes `section_id` on every chunk. Returned in search results. Shown in citations the user sees.      |
| `title`          | ✅ Core             | Becomes `section_title` on every chunk. Returned in search results and citations.                       |
| `level`          | ⚠️ Pipeline only   | Used during parsing to build hierarchy and assign prose to sections. Not stored on chunks.              |
| `target_page`    | ✅ Used             | Becomes `pdf_pages` on chunks. Returned in search results for linking to the exact spec page.           |

---

## `tables.json`

| Field            | Status            | How                                                                                                                             |
| ---------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `figure_number`  | ✅ Core           | Key for structured lookup. Stored on every table chunk for filtering. Matched against entities extracted from queries.          |
| `caption`        | ✅ Used           | Becomes `section_title` on table chunks. Prepended to every split chunk so each is self-contained.                              |
| `printed_page`   | ✅ Used           | Becomes `pdf_pages` on chunks.                                                                                                  |
| `headers`        | ✅ Core           | Serialized into `text_raw` by `serialize_table()`. Prepended to every row-group chunk when splitting large tables.              |
| `rows`           | ✅ Core           | Serialized, embedded, stored as `text_raw` in Supabase `spec_chunks`. This is what the LLM reads.                              |
| `raw_text`       | ❌ Not used       | `table_serializer.py` rebuilds text from `headers` + `rows` directly — `raw_text` is never read.                               |
| `parent_section` | ✅ Used           | Becomes `section_id` and `card_id` on table chunks. Used to find the card summary to prepend.                                  |
| `table_json`     | ✅ Used           | Stored in `spec_tables.table_json` in Supabase. Used by structured lookup to extract exact rows when a field entity is matched. |

---

## `fields.json`

| Field          | Status            | How                                                                                                                              |
| -------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `name`         | ✅ Core           | Matched against entities extracted from the query. Primary key in `spec_fields` Supabase table. Returned in citations.           |
| `description`  | ✅ Core           | Stored in `spec_fields`. Sent to the LLM as part of context. Used as gold answer in eval.                                       |
| `offset`       | ✅ Used           | `_offset_range()` + `_field_matches_bit_ranges()` in `retriever.py` — filters fields when query mentions specific bit ranges.    |
| `offset_type`  | ✅ Used           | `_field_matches_bit_ranges()` — distinguishes bit-addressed vs byte-addressed fields to avoid false filtering.                   |
| `figure_number`| ✅ Core           | Links a field to its parent table. Used to pull `table_json` from Supabase for the structured lookup response.                   |
| `full_name`    | ⚠️ Stored only   | Stored in the `data` blob in Supabase. Not directly surfaced in query responses.                                                 |
| `parent_type`  | ⚠️ Stored only   | Stored in the `data` blob. Not used in retrieval logic.                                                                          |
| `values`       | ⚠️ Stored only   | Stored in the `data` blob. LLM can read it if it lands in context, but the pipeline doesn't explicitly surface value enums.     |
| `requirements` | ⚠️ Stored only   | Stored in the `data` blob. Not used in any retrieval logic.                                                                      |
| `cross_refs`   | ❌ Not used       | Stored in the `data` blob but graph expansion is not implemented — these links are never walked at query time.                   |

---

## `field_index.json`

| Field                    | Status          | How                                                                                                                                    |
| ------------------------ | --------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `{field_name: [records]}`| ⚠️ Redundant   | Pre-computed name→records lookup that predates the DB. Replaced by the `name` primary key on `spec_fields` in Supabase. Still used as local fallback when Supabase isn't configured. |

---

## `prose.json`

| Field                   | Status          | How                                                                                                                      |
| ----------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `section_number`        | ✅ Core         | Becomes `section_id` on prose chunks.                                                                                    |
| `title`                 | ✅ Core         | Becomes `section_title` on prose chunks.                                                                                 |
| `paragraphs[].text`     | ✅ Core         | Chunked by `chunker.py`, embedded, stored as `text_raw` in Supabase `spec_chunks`. The primary search corpus.           |
| `paragraphs[].pdf_page` | ✅ Used         | Becomes `pdf_pages` on chunks.                                                                                           |
| `normative[].strength`  | ⚠️ Partial      | Drives the `has_normative` boolean on chunks. That boolean is filterable in search but nothing currently filters on it.  |
| `normative[].text`      | ✅ Indirect     | Part of the paragraph text that gets chunked — flows into `text_raw` and gets embedded. Not separately indexed.          |
| `normative[].pdf_page`  | ❌ Not used     | Captured but not propagated to chunks.                                                                                   |

---

## `definitions.json`

| Field                 | Status           | How                                                                                                                                              |
| --------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `{term: definition}`  | ❌ Not connected | Intended to be prepended to chunks before embedding so the model understands NVMe jargon, but `chunker.py` only prepends the card summary. Easy win to wire in. |

---

## `relationships.json`

| Field        | Status           | How                                                                                                          |
| ------------ | ---------------- | ------------------------------------------------------------------------------------------------------------ |
| `source`     | ❌ Not connected | Extracted and stored but graph expansion is not implemented in `orchestrator.py`. Never walked at query time. |
| `target`     | ❌ Not connected | Same.                                                                                                        |
| `type`       | ❌ Not connected | Same.                                                                                                        |
| `evidence`   | ❌ Not connected | Same.                                                                                                        |
| `confidence` | ❌ Not connected | Same.                                                                                                        |
| `strength`   | ❌ Not connected | Same. (cross_reference edges only)                                                                           |

---

## `relationships_llm.json`

| Field        | Status           | How                                                                          |
| ------------ | ---------------- | ---------------------------------------------------------------------------- |
| `source`     | ❌ Not connected | Same situation as `relationships.json` — extracted, stored, never consumed.  |
| `target`     | ❌ Not connected | Same.                                                                        |
| `type`       | ❌ Not connected | Same.                                                                        |
| `evidence`   | ❌ Not connected | Same.                                                                        |
| `confidence` | ❌ Not connected | Same.                                                                        |

---

## `cards.json`

| Field             | Status            | How                                                                                                                              |
| ----------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `section_id`      | ✅ Core           | Used to look up the right card for each chunk in `chunker.py` and `table_serializer.py`.                                        |
| `summary`         | ✅ Core           | Prepended to the `text` field (not `text_raw`) of every chunk before embedding — the "definition-enriched" context layer.       |
| `title`           | ✅ Used           | Part of the card prefix prepended to chunks.                                                                                     |
| `spec_document`   | ✅ Used           | Stored on every chunk in Supabase.                                                                                               |
| `spec_version`    | ✅ Used           | Stored on every chunk. Filterable in search.                                                                                     |
| `keywords`        | ❌ Not used       | Stored on the card but never read by the chunker, embedder, or retriever. Would improve BM25 if added to the `text` field.      |
| `parent_section`  | ⚠️ Stored only   | On the card record but not propagated to chunks or used in retrieval.                                                            |
| `child_sections`  | ⚠️ Stored only   | Stored, never read at query time.                                                                                                |
| `tables`          | ❌ Not used       | List of figure numbers — chunker doesn't use this to pull in related tables.                                                     |
| `prose_blocks`    | ⚠️ Stored only   | Indices into prose.json. Not used at query time.                                                                                 |
| `relationships`   | ❌ Not used       | Graph expansion not implemented.                                                                                                 |
| `normative_count` | ⚠️ Stored only   | Stored on the card, never read.                                                                                                  |
