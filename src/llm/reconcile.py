"""
Phase 1.5 closing step — reconcile + merge relationships.

Inputs:
  data/relationships.json      — deterministic edges (regex + structural)
  data/relationships_llm.json  — LLM-extracted edges
  data/toc.json                — canonical section IDs
  data/fields.json             — canonical field abbreviations

Outputs:
  data/relationships_merged.json  — authoritative edge list
  data/entity_registry.json       — {canonical: [aliases]} for inspection

What this script does
---------------------
1. NORMALIZE entity names on every edge:
     - collapse whitespace
     - strip leading articles ("the ")
     - strip trailing type suffixes ("Set Features command" → "Set Features")
     - strip parenthetical trailers  ("FID 0Dh (HMB)" → "FID 0Dh")
     - fold common-case variants to lowercase for equality

2. SNAP to known entities where possible:
     - section:<num>   → validated against toc.json (dropped if unknown)
     - field:<abbrev>  → validated against fields.json (kept but flagged)

3. BUILD entity registry:
     - One canonical name per (type, normalized-key)
     - Canonical = most frequent original casing across all edges
     - All aliases logged for inspection

4. REWRITE all edges to canonical names.

5. DEDUP by (source, target, type). When deterministic + llm both present,
   keep deterministic (higher confidence) but record that LLM confirmed.

6. REBUILD cards.json relationships[] using the merged list so cards stay
   in sync with the authoritative edges.

Run:
  python -m src.llm.reconcile
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Paths

from src import spec_env

DETERMINISTIC_PATH = spec_env.data_path("relationships.json")
LLM_PATH = spec_env.data_path("relationships_llm.json")
TOC_PATH = spec_env.data_path("toc.json")
FIELDS_PATH = spec_env.data_path("fields.json")
CARDS_PATH = spec_env.data_path("cards.json")

MERGED_PATH = spec_env.data_path("relationships_merged.json")
REGISTRY_PATH = spec_env.data_path("entity_registry.json")


# ---------------------------------------------------------------------------
# Normalization

_WHITESPACE_RE = re.compile(r"\s+")
_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_TRAILING_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")

# Per-type suffixes the model tends to append redundantly.
_TYPE_SUFFIXES: dict[str, list[str]] = {
    "command": ["command", "admin command", "i/o command"],
    "feature": ["feature"],
    "structure": ["data structure", "structure"],
    "log_page": ["log page"],
    "field": ["field"],
    "register": ["register"],
    "specification": ["specification", "spec"],
    "protocol": ["protocol"],
}


def _normalize_name(name: str, entity_type: str) -> str:
    """Return a normalized-for-equality form of an entity name."""
    s = name.strip()
    s = _WHITESPACE_RE.sub(" ", s)
    # Remove trailing parenthetical annotations: "FID 0Dh (HMB)" → "FID 0Dh"
    s = _TRAILING_PAREN_RE.sub("", s).strip()
    # Drop a leading article
    s = _LEADING_ARTICLE_RE.sub("", s).strip()
    # Drop type suffix (case-insensitive)
    for suffix in _TYPE_SUFFIXES.get(entity_type, []):
        pattern = re.compile(rf"\s+{re.escape(suffix)}$", re.IGNORECASE)
        s = pattern.sub("", s).strip()
    # Collapse repeated punctuation
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    return s


def _normkey(name: str, entity_type: str) -> str:
    """Lowercased key used for equality. Canonical form is preserved separately."""
    return _normalize_name(name, entity_type).lower()


def _split_node(node_id: str) -> tuple[str, str]:
    """'command:Set Features' → ('command', 'Set Features')"""
    if ":" in node_id:
        t, n = node_id.split(":", 1)
        return t.strip(), n.strip()
    return "entity", node_id.strip()


def _join_node(entity_type: str, name: str) -> str:
    return f"{entity_type}:{name}"


# ---------------------------------------------------------------------------
# I/O

def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main pipeline

def build_canonical_registry(edges: list[dict]) -> dict[tuple[str, str], str]:
    """
    For each (type, normkey), pick the most-frequent ORIGINAL spelling as canonical.
    Returns {(type, normkey): canonical_name}.
    """
    spellings: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for e in edges:
        for endpoint in (e["source"], e["target"]):
            t, n = _split_node(endpoint)
            if not n:
                continue
            key = (t, _normkey(n, t))
            if not key[1]:
                continue
            spellings[key][_normalize_name(n, t)] += 1

    canonical: dict[tuple[str, str], str] = {}
    for key, counts in spellings.items():
        canonical[key] = counts.most_common(1)[0][0]
    return canonical


def canonicalize_edge(
    edge: dict,
    canonical: dict[tuple[str, str], str],
    known_sections: set[str],
) -> dict | None:
    """
    Rewrite source/target using canonical names. Drop edges that reference
    a section that doesn't exist in the TOC (LLM hallucination).
    """
    new_edge = dict(edge)
    for side in ("source", "target"):
        t, n = _split_node(edge[side])
        if not n:
            return None
        key = (t, _normkey(n, t))
        canon = canonical.get(key, _normalize_name(n, t))

        # Section sanity: must exist in the TOC.
        if t == "section":
            sec_id = canon.strip().rstrip(".")
            if sec_id not in known_sections:
                return None
            canon = sec_id

        new_edge[side] = _join_node(t, canon)
    return new_edge


def dedup_edges(edges: list[dict]) -> list[dict]:
    """
    Merge edges with the same (source, target, type).
    Confidence precedence: deterministic > llm.
    Concatenate distinct evidence strings (capped).
    """
    by_key: dict[tuple[str, str, str], dict] = {}
    for e in edges:
        key = (e["source"], e["target"], e["type"])
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = dict(e)
            by_key[key].setdefault("evidence_list", [e.get("evidence", "")])
            by_key[key]["confirmed_by"] = {e.get("confidence", "unknown")}
            continue
        existing["confirmed_by"].add(e.get("confidence", "unknown"))
        # Upgrade confidence if the new edge is stronger.
        
        if e.get("confidence") == "deterministic":
            existing["confidence"] = "deterministic"
            if "evidence" in e:
                existing["evidence"] = e["evidence"]
        ev = e.get("evidence", "")
        if ev and ev not in existing["evidence_list"]:
            existing["evidence_list"].append(ev)

    merged: list[dict] = []
    for e in by_key.values():
        ev_list = e.pop("evidence_list", [])
        if ev_list and not e.get("evidence"):
            e["evidence"] = ev_list[0]
        # Keep a compact multi-evidence field only if there is > 1.
        if len(ev_list) > 1:
            e["evidence_all"] = ev_list[:5]
        e["confirmed_by"] = sorted(e["confirmed_by"])
        merged.append(e)
    return merged


def rebuild_card_relationships(cards: list[dict], merged: list[dict]) -> None:
    """Replace each card's relationships[] with edges that reference its section."""
    by_section: dict[str, list[dict]] = defaultdict(list)
    for e in merged:
        for endpoint in (e["source"], e["target"]):
            t, n = _split_node(endpoint)
            if t == "section":
                by_section[n].append(e)

    for card in cards:
        sec = card["section_id"]
        card["relationships"] = by_section.get(sec, [])


# ---------------------------------------------------------------------------
# Main

def run() -> None:
    det = _load_json(DETERMINISTIC_PATH, [])
    llm = _load_json(LLM_PATH, [])
    toc = _load_json(TOC_PATH, [])
    cards = _load_json(CARDS_PATH, [])

    if not det:
        print(f"ERROR: {DETERMINISTIC_PATH} empty.", file=sys.stderr)
        sys.exit(1)

    known_sections = {e["section_number"] for e in toc}

    print(f"deterministic edges: {len(det)}")
    print(f"llm edges:           {len(llm)}")

    all_edges = det + llm

    # Pass 1: build canonical name registry.
    canonical = build_canonical_registry(all_edges)
    print(f"distinct entities:   {len(canonical)}")

    # Pass 2: rewrite + drop hallucinated sections.
    rewritten: list[dict] = []
    dropped = 0
    for e in all_edges:
        ne = canonicalize_edge(e, canonical, known_sections)
        if ne is None:
            dropped += 1
            continue
        rewritten.append(ne)
    print(f"dropped (unknown section refs): {dropped}")

    # Pass 3: dedup.
    merged = dedup_edges(rewritten)
    print(f"after dedup:         {len(merged)}")

    # Break down the final set.
    by_type = Counter(e["type"] for e in merged)
    by_conf = Counter(e["confidence"] for e in merged)
    print(f"by type:       {dict(by_type.most_common())}")
    print(f"by confidence: {dict(by_conf)}")
    both = sum(1 for e in merged if len(e.get("confirmed_by", [])) > 1)
    print(f"edges confirmed by both deterministic + llm: {both}")

    _save_json(MERGED_PATH, merged)
    print(f"wrote {MERGED_PATH}")

    # Entity registry for inspection
    registry: dict[str, list[str]] = {}
    aliases: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for e in all_edges:
        for endpoint in (e["source"], e["target"]):
            t, n = _split_node(endpoint)
            if not n:
                continue
            key = (t, _normkey(n, t))
            aliases[key][n.strip()] += 1

    for key, spellings in aliases.items():
        canon = canonical.get(key)
        if not canon:
            continue
        canonical_id = _join_node(key[0], canon)
        other = [s for s in spellings if s != canon]
        if other:
            registry[canonical_id] = sorted(other, key=lambda s: -spellings[s])
    _save_json(REGISTRY_PATH, registry)
    print(f"wrote {REGISTRY_PATH}  (entities with aliases: {len(registry)})")

    # Rebuild card relationships from merged edges so cards stay in sync.
    if cards:
        rebuild_card_relationships(cards, merged)
        _save_json(CARDS_PATH, cards)
        print(f"refreshed relationships[] in {len(cards)} cards -> {CARDS_PATH}")


if __name__ == "__main__":
    run()
