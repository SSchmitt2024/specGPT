"""Read-only audit of prose.json + definitions.json."""
import json, random, statistics, re
from pathlib import Path
from collections import Counter

DATA = Path("C:/Users/sawye/Desktop/Projects/specGPT/data")
prose = json.loads((DATA / "prose.json").read_text(encoding="utf-8"))
defs = json.loads((DATA / "definitions.json").read_text(encoding="utf-8"))

# ---------- 1. PROSE CORPUS STATS ----------
print("=" * 70)
print("1. PROSE CORPUS STATS")
print("=" * 70)
total_sections = len(prose)
sections_with_empty_paras = sum(1 for s in prose if not s.get("paragraphs"))
all_paras = [p for s in prose for p in s.get("paragraphs", [])]
total_paras = len(all_paras)

def para_text(p):
    if isinstance(p, str):
        return p
    if isinstance(p, dict):
        for k in ("text", "content", "body", "paragraph"):
            if k in p and isinstance(p[k], str):
                return p[k]
        return json.dumps(p)
    return str(p)

# probe
if all_paras:
    print("paragraph sample type:", type(all_paras[0]).__name__)
    if isinstance(all_paras[0], dict):
        print("paragraph keys:", list(all_paras[0].keys()))
        print("paragraph sample:", json.dumps(all_paras[0], indent=2)[:300])

short_paras = sum(1 for p in all_paras if len(para_text(p).strip()) < 20)

# normatives
all_norms = []
for s in prose:
    for n in s.get("normative", []) or []:
        all_norms.append(n)
total_norms = len(all_norms)

# distribution by strength
def detect_strength(n):
    if isinstance(n, dict):
        s = n.get("strength") or n.get("type") or n.get("keyword")
        if s:
            return str(s).lower()
        text = n.get("text", "")
    else:
        text = str(n)
    text_l = text.lower()
    # detect first occurrence
    for kw in ["shall not", "should not", "may not", "shall", "should", "may"]:
        if re.search(r"\b" + kw + r"\b", text_l):
            return kw
    return "unknown"

# probe normative shape
if all_norms:
    print("normative sample shape:", type(all_norms[0]).__name__)
    print("first normative:", json.dumps(all_norms[0], indent=2)[:400] if isinstance(all_norms[0], dict) else str(all_norms[0])[:400])

strength_counts = Counter(detect_strength(n) for n in all_norms)
sections_with_norms = sum(1 for s in prose if s.get("normative"))

print(f"total sections:               {total_sections}")
print(f"sections w/ empty paragraphs: {sections_with_empty_paras} ({sections_with_empty_paras/total_sections:.1%})")
print(f"total paragraphs:             {total_paras}")
print(f"paragraphs <20 chars (noise): {short_paras} ({short_paras/total_paras:.1%} of paras)")
print(f"total normatives:             {total_norms}")
print(f"sections w/ >=1 normative:    {sections_with_norms} ({sections_with_norms/total_sections:.1%})")
print(f"strength distribution:        {dict(strength_counts.most_common())}")

# ---------- 2. L2 INTRO RECOVERY ----------
print("\n" + "=" * 70)
print("2. L2 INTRO RECOVERY CHECK")
print("=" * 70)
targets = ["3.7", "4.2", "8.1", "B.5"]
by_num = {s["section_number"]: s for s in prose}
for tgt in targets:
    s = by_num.get(tgt)
    if not s:
        print(f"  [{tgt}] MISSING from prose.json")
        continue
    paras = s.get("paragraphs", [])
    norms = s.get("normative", [])
    has_norm_kw = any(re.search(r"\b(shall|should|may)\b", p.lower()) for p in paras)
    print(f"  [{tgt}] {s['title'][:60]!r}")
    print(f"        paragraphs={len(paras)}  normative_entries={len(norms)}  paras_contain_shall/should/may={has_norm_kw}")
    if paras:
        print(f"        first para preview: {paras[0][:120]!r}")

# ---------- 3. PROSE SPOT CHECKS ----------
print("\n" + "=" * 70)
print("3. PROSE SPOT CHECKS (random.sample seed=42, k=6)")
print("=" * 70)
random.seed(42)
sample = random.sample(prose, 6)
for s in sample:
    paras = s.get("paragraphs", [])
    norms = s.get("normative", [])
    first = paras[0][:80] if paras else "<EMPTY>"
    print(f"  [{s['section_number']}] L{s.get('level')} {s['title'][:50]!r}")
    print(f"        paras={len(paras)}  norms={len(norms)}")
    print(f"        first80: {first!r}")

# ---------- 4. DEFINITIONS CORPUS STATS ----------
print("\n" + "=" * 70)
print("4. DEFINITIONS CORPUS STATS")
print("=" * 70)
# defs is dict: term -> string definition (based on probe)
total_terms = len(defs)
def_lengths = [len(v) if isinstance(v, str) else len(json.dumps(v)) for v in defs.values()]
avg_len = statistics.mean(def_lengths)
median_len = statistics.median(def_lengths)
# source section detection — definitions may include "(refer to section X.Y)" or have a 'source' field
# since values are strings, count refs to 1.5 vs other sections
src_counter = Counter()
for v in defs.values():
    if not isinstance(v, str):
        v = json.dumps(v)
    # crude: heuristic — definitions from §1.5 typically don't say "refer to". We can't tell directly
    # check for explicit section attribution
    m = re.search(r"defined in section (\d+(?:\.\d+)*)", v, re.I)
    if m:
        src_counter[m.group(1)] += 1
    else:
        src_counter["__no_explicit_attribution__"] += 1
print(f"total terms:               {total_terms}")
print(f"avg def length (chars):    {avg_len:.1f}")
print(f"median def length (chars): {median_len:.1f}")
print(f"min/max def length:        {min(def_lengths)}/{max(def_lengths)}")
print(f"explicit-attribution top5: {src_counter.most_common(5)}")

# ---------- 5. TARGETED DEFINITION CHECKS ----------
print("\n" + "=" * 70)
print("5. TARGETED DEFINITION CHECKS")
print("=" * 70)
def find_term(needle):
    needle_l = needle.lower()
    # exact key match
    for k in defs.keys():
        if k.lower() == needle_l:
            return ("KEY_EXACT", k)
    # key contains
    for k in defs.keys():
        if needle_l in k.lower():
            return ("KEY_CONTAINS", k)
    # in body
    hits = [k for k, v in defs.items() if isinstance(v, str) and needle_l in v.lower()]
    if hits:
        return ("BODY", hits[:3])
    return ("MISSING", None)

for term in ["LBA", "logical block address", "Namespace", "Domain", "Controller", "Reservation", "NVM Subsystem"]:
    status, hit = find_term(term)
    print(f"  {term!r:35} -> {status:15} {hit}")

# ---------- 6. DEFINITIONS SPOT CHECKS ----------
print("\n" + "=" * 70)
print("6. DEFINITIONS SPOT CHECKS (random.sample seed=42, k=6)")
print("=" * 70)
random.seed(42)
keys = list(defs.keys())
sample_keys = random.sample(keys, 6)
for k in sample_keys:
    v = defs[k]
    s = v if isinstance(v, str) else json.dumps(v)
    print(f"  [{k!r}]")
    print(f"    {s[:100]!r}")
