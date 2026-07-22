#!/usr/bin/env bash
#
# run_phase2.sh
# -------------
# Interactive runner for the Phase 2 pipeline (retrieval + demo).
# Same UX as rerun_pipeline.sh — pick steps, backup outputs, run in order.
#
# Pipeline (output files in parentheses):
#   1. Chunk prose         scripts/chunker.py             (chunks_prose.json)
#   2. Serialize tables    src.pipeline.table_serializer  (chunks_tables.json)
#   3. Embed chunks        scripts/embedder.py            (chunks_embedded.json)
#   4. Apply schema        scripts/apply_schema.py        (remote DB — DDL)
#   5. Index to Supabase   scripts/indexer.py             (remote DB — chunks)
#   6. Load lookup data    scripts/load_lookup_data.py    (remote DB — fields/tables/enum_index)
#   7. Generate eval set   scripts/eval_gen.py            (eval_set.json)
#   8. Run eval            scripts/eval_run.py            (eval_results.json)
#
# Usage:
#   ./scripts/run_phase2.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Locate project root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Source .env if present
# ---------------------------------------------------------------------------
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

# ---------------------------------------------------------------------------
# Choose which specification to ingest: NVMe Base vs PCIe Transport.
# Exports SPEC_DATA_DIR (consumed by chunker/embedder/indexer/load_lookup_data)
# plus the spec metadata tags. With NVME_SPEC unset/"base", behavior is
# unchanged. See docs/PCIE_MULTI_SPEC_PLAN.md.
# ---------------------------------------------------------------------------
select_spec() {
  local choice="${NVME_SPEC:-}"   # pre-set (env or .env) skips the prompt
  if [[ -z "$choice" ]]; then
    echo ""
    echo "Which specification do you want to ingest?"
    echo "  1) base     — NVM Express Base Specification              <- data/"
    echo "  2) pcie     — NVM Express PCIe Transport Spec              <- data/pcie/"
    echo "  3) command  — NVM Express NVM Command Set Spec             <- data/command/"
    echo "  4) boot     — NVM Express Boot Specification                <- data/boot/"
    echo "  5) cps      — NVM Express Computational Programs Cmd Set  <- data/cps/"
    echo "  6) kv       — NVM Express Key Value Command Set            <- data/kv/"
    echo "  7) mi       — NVM Express Management Interface Spec        <- data/mi/"
    echo "  8) rdma     — NVM Express over RDMA Transport Spec         <- data/rdma/"
    echo "  9) tcp      — NVM Express over TCP Transport Spec          <- data/tcp/"
    echo " 10) slm      — NVM Express Subsystem Local Memory Cmd Set  <- data/slm/"
    echo " 11) zns      — NVM Express Zoned Namespace Command Set     <- data/zns/"
    read -rp "spec> [1] " spec_reply
    case "${spec_reply:-1}" in
      1|base|Base|BASE)             choice="base" ;;
      2|pcie|Pcie|PCIE|PCIe)        choice="pcie" ;;
      3|command|Command|COMMAND)    choice="command" ;;
      4|boot|Boot|BOOT)             choice="boot" ;;
      5|cps|Cps|CPS)                choice="cps" ;;
      6|kv|Kv|KV)                   choice="kv" ;;
      7|mi|Mi|MI)                   choice="mi" ;;
      8|rdma|Rdma|RDMA)             choice="rdma" ;;
      9|tcp|Tcp|TCP)                choice="tcp" ;;
      10|slm|Slm|SLM)               choice="slm" ;;
      11|zns|Zns|ZNS)               choice="zns" ;;
      *) echo "ERROR: unknown spec '$spec_reply' (pick 1-11)" >&2; exit 1 ;;
    esac
  fi

  case "$choice" in
    base)
      export NVME_SPEC="base"
      export SPEC_DATA_DIR="${SPEC_DATA_DIR:-data}"
      export SPEC_DOCUMENT="${SPEC_DOCUMENT:-NVM Express Base Specification}"
      export SPEC_VERSION="${SPEC_VERSION:-2.1}"
      export SPEC_FIRST_CONTENT="${SPEC_FIRST_CONTENT:-24}"
      ;;
    pcie)
      export NVME_SPEC="pcie"
      export SPEC_DATA_DIR="${SPEC_DATA_DIR:-data/pcie}"
      export SPEC_DOCUMENT="${SPEC_DOCUMENT:-NVM Express PCIe Transport Specification}"
      export SPEC_VERSION="${SPEC_VERSION:-1.3}"
      export SPEC_FIRST_CONTENT="${SPEC_FIRST_CONTENT:-5}"
      ;;
    command)
      export NVME_SPEC="command"
      export SPEC_DATA_DIR="${SPEC_DATA_DIR:-data/command}"
      export SPEC_DOCUMENT="${SPEC_DOCUMENT:-NVM Express NVM Command Set Specification}"
      export SPEC_VERSION="${SPEC_VERSION:-1.2}"
      export SPEC_FIRST_CONTENT="${SPEC_FIRST_CONTENT:-7}"
      ;;
    boot)
      export NVME_SPEC="boot"
      export SPEC_DATA_DIR="${SPEC_DATA_DIR:-data/boot}"
      export SPEC_DOCUMENT="${SPEC_DOCUMENT:-NVM Express Boot Specification}"
      export SPEC_VERSION="${SPEC_VERSION:-1.3}"
      export SPEC_FIRST_CONTENT="${SPEC_FIRST_CONTENT:-6}"
      ;;
    cps)
      export NVME_SPEC="cps"
      export SPEC_DATA_DIR="${SPEC_DATA_DIR:-data/cps}"
      export SPEC_DOCUMENT="${SPEC_DOCUMENT:-NVM Express Computational Programs Command Set Specification}"
      export SPEC_VERSION="${SPEC_VERSION:-1.2}"
      export SPEC_FIRST_CONTENT="${SPEC_FIRST_CONTENT:-6}"
      ;;
    kv)
      export NVME_SPEC="kv"
      export SPEC_DATA_DIR="${SPEC_DATA_DIR:-data/kv}"
      export SPEC_DOCUMENT="${SPEC_DOCUMENT:-NVM Express Key Value Command Set Specification}"
      export SPEC_VERSION="${SPEC_VERSION:-1.3}"
      export SPEC_FIRST_CONTENT="${SPEC_FIRST_CONTENT:-4}"
      ;;
    mi)
      export NVME_SPEC="mi"
      export SPEC_DATA_DIR="${SPEC_DATA_DIR:-data/mi}"
      export SPEC_DOCUMENT="${SPEC_DOCUMENT:-NVM Express Management Interface Specification}"
      export SPEC_VERSION="${SPEC_VERSION:-2.1}"
      export SPEC_FIRST_CONTENT="${SPEC_FIRST_CONTENT:-9}"
      ;;
    rdma)
      export NVME_SPEC="rdma"
      export SPEC_DATA_DIR="${SPEC_DATA_DIR:-data/rdma}"
      export SPEC_DOCUMENT="${SPEC_DOCUMENT:-NVM Express NVMe-over-RDMA Transport Specification}"
      export SPEC_VERSION="${SPEC_VERSION:-1.2}"
      export SPEC_FIRST_CONTENT="${SPEC_FIRST_CONTENT:-4}"
      ;;
    tcp)
      export NVME_SPEC="tcp"
      export SPEC_DATA_DIR="${SPEC_DATA_DIR:-data/tcp}"
      export SPEC_DOCUMENT="${SPEC_DOCUMENT:-NVM Express NVMe-over-TCP Transport Specification}"
      export SPEC_VERSION="${SPEC_VERSION:-1.2}"
      export SPEC_FIRST_CONTENT="${SPEC_FIRST_CONTENT:-5}"
      ;;
    slm)
      export NVME_SPEC="slm"
      export SPEC_DATA_DIR="${SPEC_DATA_DIR:-data/slm}"
      export SPEC_DOCUMENT="${SPEC_DOCUMENT:-NVM Express Subsystem Local Memory Command Set Specification}"
      export SPEC_VERSION="${SPEC_VERSION:-1.2}"
      export SPEC_FIRST_CONTENT="${SPEC_FIRST_CONTENT:-5}"
      ;;
    zns)
      export NVME_SPEC="zns"
      export SPEC_DATA_DIR="${SPEC_DATA_DIR:-data/zns}"
      export SPEC_DOCUMENT="${SPEC_DOCUMENT:-NVM Express Zoned Namespace Command Set Specification}"
      export SPEC_VERSION="${SPEC_VERSION:-1.4}"
      export SPEC_FIRST_CONTENT="${SPEC_FIRST_CONTENT:-5}"
      ;;
  esac

  echo ""
  echo "=== spec: $NVME_SPEC (data dir: $SPEC_DATA_DIR, $SPEC_DOCUMENT v$SPEC_VERSION) ==="
}

select_spec

# ---------------------------------------------------------------------------
# Find Python interpreter
# ---------------------------------------------------------------------------
export PYTHONUNBUFFERED=1

detect_python() {
  local -a candidates=()
  [[ -n "${VIRTUAL_ENV:-}" ]] && candidates+=("$VIRTUAL_ENV/bin/python" "$VIRTUAL_ENV/Scripts/python.exe")
  candidates+=(".venv/bin/python" ".venv/Scripts/python.exe")
  candidates+=("python3" "py -3" "python")

  for cand in "${candidates[@]}"; do
    # shellcheck disable=SC2206
    local -a parts=($cand)
    if "${parts[@]}" --version >/dev/null 2>&1; then
      PYTHON_BIN=("${parts[@]}")
      return 0
    fi
  done
  return 1
}

if ! detect_python; then
  echo "ERROR: no working Python interpreter found." >&2
  exit 1
fi
echo "using python: ${PYTHON_BIN[*]} ($("${PYTHON_BIN[@]}" --version 2>&1))"

# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------
STEP_NAME=(
  "Chunk prose"
  "Serialize tables"
  "Embed chunks"
  "Apply schema"
  "Index to Supabase"
  "Load lookup data"
  "Generate eval set"
  "Run eval"
)
# For script-based steps, leave MODULE empty and set SCRIPT path instead.
STEP_MODULE=(
  ""
  "src.pipeline.table_serializer"
  ""
  ""
  ""
  ""
  ""
  ""
)
STEP_SCRIPT=(
  "scripts/chunker.py"
  ""
  "scripts/embedder.py"
  "scripts/apply_schema.py"
  "scripts/indexer.py"
  "scripts/load_lookup_data.py"
  "scripts/eval_gen.py"
  "scripts/eval_run.py"
)
# Output/input paths are scoped to the active spec's data dir ($SPEC_DATA_DIR,
# exported by select_spec above; "data" for base, "data/pcie" for pcie).
DD="$SPEC_DATA_DIR"
STEP_OUTPUTS=(
  "$DD/chunks_prose.json"
  "$DD/chunks_tables.json"
  "$DD/chunks_embedded.json"
  ""
  ""
  ""
  "$DD/eval_set.json"
  "$DD/eval_results.json"
)
STEP_INPUTS=(
  "$DD/prose.json $DD/cards.json"
  "$DD/tables.json $DD/cards.json"
  "$DD/chunks_prose.json $DD/chunks_tables.json"
  ""
  "$DD/chunks_embedded.json"
  "$DD/fields.json $DD/field_index.json $DD/tables.json"
  "$DD/cards.json $DD/fields.json $DD/field_index.json"
  "$DD/eval_set.json"
)
STEP_API=(
  "no"
  "no"
  "voyage"
  "db"
  "supabase"
  "supabase"
  "gemini"
  "supabase gemini anthropic voyage"
)

NUM_STEPS=${#STEP_NAME[@]}

# ---------------------------------------------------------------------------
# Menu + selection parser
# ---------------------------------------------------------------------------
print_menu() {
  echo ""
  echo "specGPT Phase 2 pipeline — pick the step(s) to run:"
  echo ""
  for i in "${!STEP_NAME[@]}"; do
    local tag=""
    [[ "${STEP_API[$i]}" == "yes" ]] && tag=" [API]"
    printf "  %d) %s%s\n" "$((i+1))" "${STEP_NAME[$i]}" "$tag"
  done
  echo ""
  echo "Syntax: 3           single"
  echo "        1,3         list"
  echo "        1-3         range"
  echo "        all         everything (in order)"
  echo ""
}

SELECTED_STEPS=()
parse_selection() {
  local input="$1"
  local -a out=()

  if [[ "$input" == "all" ]]; then
    for i in "${!STEP_NAME[@]}"; do out+=("$i"); done
  else
    IFS=',' read -ra toks <<< "$input"
    for tok in "${toks[@]}"; do
      tok="${tok// /}"
      if [[ "$tok" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        local a="${BASH_REMATCH[1]}" b="${BASH_REMATCH[2]}"
        (( a <= b )) || { echo "ERROR: invalid range '$tok'" >&2; return 1; }
        for (( i=a; i<=b; i++ )); do out+=("$((i-1))"); done
      elif [[ "$tok" =~ ^[0-9]+$ ]]; then
        out+=("$((tok-1))")
      else
        echo "ERROR: bad token '$tok'" >&2
        return 1
      fi
    done
  fi

  for idx in "${out[@]}"; do
    if (( idx < 0 || idx >= NUM_STEPS )); then
      echo "ERROR: step $((idx+1)) out of range (1-$NUM_STEPS)" >&2
      return 1
    fi
  done

  IFS=$'\n' read -r -d '' -a SELECTED_STEPS < <(printf '%s\n' "${out[@]}" | sort -nu; printf '\0')
}

print_menu
read -rp "selection> " raw_selection
parse_selection "$raw_selection" || exit 1

if (( ${#SELECTED_STEPS[@]} == 0 )); then
  echo "no steps selected. exiting."
  exit 0
fi

echo ""
echo "will run (in order):"
for idx in "${SELECTED_STEPS[@]}"; do
  printf "  %d) %s\n" "$((idx+1))" "${STEP_NAME[$idx]}"
done

# ---------------------------------------------------------------------------
# API key check
# ---------------------------------------------------------------------------
_needs_api() {
  local tag="$1"
  for idx in "${SELECTED_STEPS[@]}"; do
    [[ " ${STEP_API[$idx]} " == *" $tag "* ]] && return 0
  done
  return 1
}

_prompt_env() {
  local var="$1" prompt="$2" secret="${3:-n}"
  if [[ -n "${!var:-}" ]]; then
    echo "    $var: present"
    return
  fi
  echo "    $var not set."
  if [[ "$secret" == "y" ]]; then
    read -rsp "    $prompt (input hidden): " entered; echo ""
  else
    read -rp "    $prompt: " entered
  fi
  if [[ -z "$entered" ]]; then
    echo "ERROR: empty value. aborting." >&2; exit 1
  fi
  export "$var"="$entered"
  read -rp "    save $var to .env? [y/N] " persist
  if [[ "$persist" == "y" || "$persist" == "Y" ]]; then
    touch ".env"
    grep -q "^${var}=" .env || echo "${var}=${entered}" >> .env
    echo "    written to .env"
  fi
}

any_api=0
for idx in "${SELECTED_STEPS[@]}"; do
  [[ "${STEP_API[$idx]}" != "no" ]] && any_api=1
done

if (( any_api == 1 )); then
  echo ""
  echo "=== API setup ==="
  _needs_api "voyage"    && _prompt_env VOYAGE_API_KEY    "paste Voyage AI key"       y
  _needs_api "db"        && _prompt_env DATABASE_URL       "paste Postgres DATABASE_URL (Supabase Project Settings → Database → URI)"
  _needs_api "supabase"  && _prompt_env SUPABASE_URL       "paste Supabase project URL"
  _needs_api "supabase"  && _prompt_env SUPABASE_KEY       "paste service_role key"    y
  _needs_api "gemini"    && _prompt_env GEMINI_API_KEY     "paste Gemini API key"      y
  _needs_api "anthropic" && _prompt_env ANTHROPIC_API_KEY  "paste Anthropic API key"   y
fi

# ---------------------------------------------------------------------------
# Backup + overwrite handling
# ---------------------------------------------------------------------------
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="Backups/phase2_${NVME_SPEC}_${STAMP}"
mkdir -p "$BACKUP_DIR"
echo ""
echo "=== backup dir: $BACKUP_DIR ==="

OVERWRITE_ALL=0

backup_file() {
  local f="$1"
  if [[ -f "$f" ]]; then
    mkdir -p "$BACKUP_DIR/$(dirname "$f")"
    cp -p "$f" "$BACKUP_DIR/$f"
    echo "    backed up: $f"
  fi
}

prompt_overwrite() {
  local step_label="$1"
  local files_desc="$2"
  if (( OVERWRITE_ALL == 1 )); then
    echo "    [overwrite-all] proceeding with $step_label"
    return 0
  fi
  local reply
  read -rp "    overwrite ${files_desc}? [y / N / Y=all / q=quit] " reply
  case "$reply" in
    y)     return 0 ;;
    Y)     OVERWRITE_ALL=1; return 0 ;;
    q|Q)   return 2 ;;
    *)     return 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# Execute steps
# ---------------------------------------------------------------------------
run_step() {
  local idx="$1"
  local name="${STEP_NAME[$idx]}"
  local module="${STEP_MODULE[$idx]}"
  local outputs="${STEP_OUTPUTS[$idx]}"
  local inputs="${STEP_INPUTS[$idx]}"
  local script="${STEP_SCRIPT[$idx]}"

  echo ""
  echo "=============================================================="
  echo "Step $((idx+1)): $name"
  if [[ -n "$script" ]]; then
    echo "  script:  $script"
  else
    echo "  module:  python -m $module"
  fi
  echo "  outputs: $outputs"
  echo "=============================================================="

  # Input warnings
  if [[ -n "$inputs" ]]; then
    local missing=()
    for f in $inputs; do
      [[ -f "$f" ]] || missing+=("$f")
    done
    if (( ${#missing[@]} > 0 )); then
      echo "    WARN: missing inputs:"
      for m in "${missing[@]}"; do echo "      - $m"; done
    fi
  fi

  # Backup existing outputs
  local any_existed=0
  for out in $outputs; do
    if [[ -f "$out" ]]; then
      any_existed=1
      backup_file "$out"
    fi
  done

  # Overwrite prompt
  if (( any_existed == 1 )); then
    prompt_overwrite "$name" "$outputs"
    local rc=$?
    case "$rc" in
      0) ;;
      1) echo "    skipped."; return 0 ;;
      2) echo "    aborting."; exit 0 ;;
    esac
  else
    echo "    (no existing outputs — first run)"
  fi

  # Run — script path takes precedence over -m module
  if [[ -n "$script" ]]; then
    echo "    running: ${PYTHON_BIN[*]} -u $script"
    if command -v stdbuf >/dev/null 2>&1; then
      stdbuf -oL -eL "${PYTHON_BIN[@]}" -u "$script"
    else
      "${PYTHON_BIN[@]}" -u "$script"
    fi
  else
    echo "    running: ${PYTHON_BIN[*]} -u -m $module"
    if command -v stdbuf >/dev/null 2>&1; then
      stdbuf -oL -eL "${PYTHON_BIN[@]}" -u -m "$module"
    else
      "${PYTHON_BIN[@]}" -u -m "$module"
    fi
  fi
  echo "    done: $name"
}

for idx in "${SELECTED_STEPS[@]}"; do
  run_step "$idx"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "Phase 2 pipeline complete."
echo "  backup dir: $BACKUP_DIR"
echo "  rollback:   cp -pr \"$BACKUP_DIR\"/data/* data/"
echo "=============================================================="
