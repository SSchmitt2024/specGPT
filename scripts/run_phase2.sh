#!/usr/bin/env bash
#
# run_phase2.sh
# -------------
# Interactive runner for the Phase 2 pipeline (retrieval + demo).
# Same UX as rerun_pipeline.sh — pick steps, backup outputs, run in order.
#
# Pipeline (output files in parentheses):
#   1. Chunk prose         src.pipeline.chunker           (chunks_prose.json)
#   2. Serialize tables    src.pipeline.table_serializer  (chunks_tables.json)
#   3. Embed chunks        src.pipeline.embedder          (chunks_embedded.json)
#
# Future steps (uncomment as built):
#   4. Index to Supabase   src.pipeline.indexer           (remote DB)
#
# Future steps (uncomment as built):
#   5. Eval set gen        src.pipeline.eval_gen          (eval_set.json)
#   6. Retriever           src.pipeline.retriever         (module — no file output)
#   7. Generator           src.pipeline.generator         (module — no file output)
#   8. Web app             src.pipeline.app               (server)
#   9. Eval run            src.pipeline.eval_run          (eval_results.json)
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
  "Index to Supabase"
)
STEP_MODULE=(
  "src.pipeline.chunker"
  "src.pipeline.table_serializer"
  "src.pipeline.embedder"
  "src.pipeline.indexer"
)
STEP_OUTPUTS=(
  "data/chunks_prose.json"
  "data/chunks_tables.json"
  "data/chunks_embedded.json"
  ""
)
STEP_INPUTS=(
  "data/prose.json data/cards.json"
  "data/tables.json data/cards.json"
  "data/chunks_prose.json data/chunks_tables.json"
  "data/chunks_embedded.json"
)
STEP_API=(
  "no"
  "no"
  "yes"
  "yes"
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

  mapfile -t SELECTED_STEPS < <(printf '%s\n' "${out[@]}" | sort -nu)
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
# API key check (Voyage AI for embedder)
# ---------------------------------------------------------------------------
needs_api=0
for idx in "${SELECTED_STEPS[@]}"; do
  [[ "${STEP_API[$idx]}" == "yes" ]] && needs_api=1
done

if (( needs_api == 1 )); then
  echo ""
  echo "=== API setup ==="

  # --- Voyage AI (embedder) ---
  needs_voyage=0
  for idx in "${SELECTED_STEPS[@]}"; do
    [[ "${STEP_MODULE[$idx]}" == "src.pipeline.embedder" ]] && needs_voyage=1
  done
  if (( needs_voyage == 1 )); then
    if [[ -z "${VOYAGE_API_KEY:-}" ]]; then
      echo "    VOYAGE_API_KEY not set."
      read -rsp "    paste key (input hidden): " entered_key
      echo ""
      if [[ -z "$entered_key" ]]; then
        echo "ERROR: empty key. aborting." >&2
        exit 1
      fi
      export VOYAGE_API_KEY="$entered_key"

      read -rp "    save VOYAGE_API_KEY to .env? [y/N] " persist
      if [[ "$persist" == "y" || "$persist" == "Y" ]]; then
        touch ".env"
        grep -q "^VOYAGE_API_KEY=" .env || echo "VOYAGE_API_KEY=${entered_key}" >> .env
        echo "    written to .env"
      fi
    else
      echo "    VOYAGE_API_KEY: present"
    fi
  fi

  # --- Supabase (indexer) ---
  needs_supabase=0
  for idx in "${SELECTED_STEPS[@]}"; do
    [[ "${STEP_MODULE[$idx]}" == "src.pipeline.indexer" ]] && needs_supabase=1
  done
  if (( needs_supabase == 1 )); then
    if [[ -z "${SUPABASE_URL:-}" ]]; then
      echo "    SUPABASE_URL not set."
      read -rp "    paste project URL: " entered_url
      if [[ -z "$entered_url" ]]; then
        echo "ERROR: empty URL. aborting." >&2
        exit 1
      fi
      export SUPABASE_URL="$entered_url"

      read -rp "    save SUPABASE_URL to .env? [y/N] " persist
      if [[ "$persist" == "y" || "$persist" == "Y" ]]; then
        touch ".env"
        grep -q "^SUPABASE_URL=" .env || echo "SUPABASE_URL=${entered_url}" >> .env
        echo "    written to .env"
      fi
    else
      echo "    SUPABASE_URL: present"
    fi

    if [[ -z "${SUPABASE_KEY:-}" ]]; then
      echo "    SUPABASE_KEY not set."
      read -rsp "    paste service_role key (input hidden): " entered_skey
      echo ""
      if [[ -z "$entered_skey" ]]; then
        echo "ERROR: empty key. aborting." >&2
        exit 1
      fi
      export SUPABASE_KEY="$entered_skey"

      read -rp "    save SUPABASE_KEY to .env? [y/N] " persist
      if [[ "$persist" == "y" || "$persist" == "Y" ]]; then
        touch ".env"
        grep -q "^SUPABASE_KEY=" .env || echo "SUPABASE_KEY=${entered_skey}" >> .env
        echo "    written to .env"
      fi
    else
      echo "    SUPABASE_KEY: present"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Backup + overwrite handling
# ---------------------------------------------------------------------------
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="Backups/phase2_${STAMP}"
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

  echo ""
  echo "=============================================================="
  echo "Step $((idx+1)): $name"
  echo "  module:  python -m $module"
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

  # Run
  echo "    running: ${PYTHON_BIN[*]} -u -m $module"
  if command -v stdbuf >/dev/null 2>&1; then
    stdbuf -oL -eL "${PYTHON_BIN[@]}" -u -m "$module"
  else
    "${PYTHON_BIN[@]}" -u -m "$module"
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
