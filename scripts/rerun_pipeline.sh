#!/usr/bin/env bash
#
# rerun_pipeline.sh
# -----------------
# Interactive re-run tool for the specGPT data pipeline. Pick one or more
# pipeline steps, the script backs up the files each step would overwrite
# (to Backups/pipeline_<timestamp>/), prompts before replacing, then runs
# the steps in pipeline order.
#
# Pipeline (output files in parentheses):
#   1. TOC rebuild               src.toc_rebuild               (toc.json)
#   2. Deep-section enrichment   src.deep_sections             (toc.json in place)
#   3. Tables                    src.tables                    (tables.json)
#   4. Prose + definitions       src.prose                     (prose.json, definitions.json)
#   5. Fields + field index      src.fields                    (fields.json, field_index.json)
#   6. Relationships (det.)      src.relationships             (relationships.json)
#   7. LLM relationships         src.llm.extract_relationships (relationships_llm.json, _state.json)
#   8. Reconcile                 src.llm.reconcile             (relationships_merged.json, entity_registry.json, cards.json refresh)
#   9. Cards (summaries)         src.llm.generate_cards        (cards.json, cards_state.json)
#
# Usage:
#   ./scripts/rerun_pipeline.sh
#
# The script is interactive — just run it and answer the prompts.

set -euo pipefail

# ---------------------------------------------------------------------------
# Step 0 — locate project root so the script works from anywhere.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Step 1 — if a .env file exists, source it so existing keys / LLM_PROVIDER
# are picked up automatically. `set -a` exports every variable assigned.
# ---------------------------------------------------------------------------
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

# ---------------------------------------------------------------------------
# Step 1.5 — find a working Python interpreter.
# On Windows / git-bash, `.venv/bin/python` is often a broken symlink and a
# bare `python` may not be on PATH. We probe a list of candidates and pick
# the first one that actually runs `--version` successfully.
#
# PYTHON_BIN is an array because launchers like `py -3` are two tokens.
# PYTHONUNBUFFERED=1 forces stdout/stderr to flush per line so you see
# LLM progress live instead of in a buffered dump at the end.
# ---------------------------------------------------------------------------
export PYTHONUNBUFFERED=1

detect_python() {
  local -a candidates=()
  # Prefer the currently-activated venv if one is active.
  [[ -n "${VIRTUAL_ENV:-}" ]] && candidates+=("$VIRTUAL_ENV/bin/python" "$VIRTUAL_ENV/Scripts/python.exe")
  # Then the in-repo venv, Linux-style then Windows-style.
  candidates+=(".venv/bin/python" ".venv/Scripts/python.exe")
  # Fall back to system interpreters in rough "most likely to work" order.
  candidates+=("python3" "py -3" "python")

  for cand in "${candidates[@]}"; do
    # Split multi-token candidates (e.g. "py -3") on whitespace.
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
  echo "  tried: \$VIRTUAL_ENV/bin/python, .venv/bin/python," >&2
  echo "         .venv/Scripts/python.exe, python3, py -3, python" >&2
  echo "  install Python or activate the venv and retry." >&2
  exit 1
fi
echo "using python: ${PYTHON_BIN[*]} ($("${PYTHON_BIN[@]}" --version 2>&1))"

# ---------------------------------------------------------------------------
# Step 2 — pipeline definition.
# Parallel arrays, indexed 0..N-1. Edit here to add or reorder steps.
# ---------------------------------------------------------------------------
#   STEP_NAME    : human label shown in the menu
#   STEP_MODULE  : python -m <this> is how each step is invoked
#   STEP_OUTPUTS : space-separated output files (backed up + overwrite-prompted)
#   STEP_INPUTS  : space-separated expected input files (warn if missing)
#   STEP_LLM     : "yes" if the step makes LLM calls (needs provider + key)
STEP_NAME=(
  "TOC rebuild"
  "Deep-section enrichment"
  "Tables"
  "Prose + definitions"
  "Fields + field index"
  "Relationships (deterministic)"
  "Relationships (LLM)"
  "Reconcile relationships"
  "Cards (summaries + keywords)"
)
STEP_MODULE=(
  "src.toc_rebuild"
  "src.deep_sections"
  "src.tables"
  "src.prose"
  "src.fields"
  "src.relationships"
  "src.llm.extract_relationships"
  "src.llm.reconcile"
  "src.llm.generate_cards"
)
STEP_OUTPUTS=(
  "data/toc.json"
  "data/toc.json"
  "data/tables.json"
  "data/prose.json data/definitions.json"
  "data/fields.json data/field_index.json"
  "data/relationships.json"
  "data/relationships_llm.json data/relationships_llm_state.json"
  "data/relationships_merged.json data/entity_registry.json data/cards.json"
  "data/cards.json data/cards_state.json"
)
STEP_INPUTS=(
  ""
  "data/toc.json"
  ""
  "data/toc.json"
  "data/tables.json"
  "data/toc.json data/tables.json"
  "data/prose.json data/toc.json data/fields.json"
  "data/relationships.json data/relationships_llm.json data/toc.json data/fields.json data/cards.json"
  "data/toc.json data/prose.json data/tables.json data/relationships.json"
)
STEP_LLM=(
  "no" "no" "no" "no" "no" "no" "yes" "yes" "yes"
)

NUM_STEPS=${#STEP_NAME[@]}

# ---------------------------------------------------------------------------
# Step 3 — print the menu and read the user's step selection.
# Accepts: individual numbers ("3"), comma lists ("1,3,5"), ranges ("1-4"),
# mixed ("1,3-5,9"), or the literal word "all".
# ---------------------------------------------------------------------------
print_menu() {
  echo ""
  echo "specGPT pipeline — pick the step(s) to re-run:"
  echo ""
  for i in "${!STEP_NAME[@]}"; do
    local tag=""
    [[ "${STEP_LLM[$i]}" == "yes" ]] && tag=" [LLM]"
    printf "  %d) %s%s\n" "$((i+1))" "${STEP_NAME[$i]}" "$tag"
  done
  echo ""
  echo "Syntax: 3           single"
  echo "        1,3,5       list"
  echo "        1-4         range"
  echo "        1,3-5,9     mixed"
  echo "        all         everything (in pipeline order)"
  echo ""
}

# Populates the global SELECTED_STEPS array (0-indexed, sorted, deduped).
SELECTED_STEPS=()
parse_selection() {
  local input="$1"
  local -a out=()

  if [[ "$input" == "all" ]]; then
    for i in "${!STEP_NAME[@]}"; do out+=("$i"); done
  else
    IFS=',' read -ra toks <<< "$input"
    for tok in "${toks[@]}"; do
      tok="${tok// /}"  # strip spaces
      if [[ "$tok" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        local a="${BASH_REMATCH[1]}" b="${BASH_REMATCH[2]}"
        (( a <= b )) || { echo "ERROR: invalid range '$tok'" >&2; return 1; }
        for (( i=a; i<=b; i++ )); do out+=("$((i-1))"); done
      elif [[ "$tok" =~ ^[0-9]+$ ]]; then
        out+=("$((tok-1))")
      else
        echo "ERROR: bad selection token '$tok'" >&2
        return 1
      fi
    done
  fi

  # validate bounds
  for idx in "${out[@]}"; do
    if (( idx < 0 || idx >= NUM_STEPS )); then
      echo "ERROR: step $((idx+1)) out of range (1-$NUM_STEPS)" >&2
      return 1
    fi
  done

  # sort + dedupe, assign to global
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
# Step 4 — if any selected step is an LLM step, make sure LLM_PROVIDER is set
# and that the matching API key is in the environment. If the key is missing,
# prompt for it (input is hidden), export it for this session, and optionally
# append it to .env so future runs pick it up.
# ---------------------------------------------------------------------------
needs_llm=0
for idx in "${SELECTED_STEPS[@]}"; do
  [[ "${STEP_LLM[$idx]}" == "yes" ]] && needs_llm=1
done

if (( needs_llm == 1 )); then
  echo ""
  echo "=== LLM setup ==="

  # Provider
  if [[ -z "${LLM_PROVIDER:-}" ]]; then
    echo "LLM_PROVIDER not set. pick a provider:"
    PS3="provider> "
    select p in gemini openai; do
      if [[ -n "$p" ]]; then
        export LLM_PROVIDER="$p"
        break
      fi
    done
  fi
  echo "    provider: $LLM_PROVIDER"

  # Map provider -> required key env var
  case "$LLM_PROVIDER" in
    gemini) KEY_VAR="GEMINI_API_KEY" ;;
    openai) KEY_VAR="OPENAI_API_KEY" ;;
    *) echo "ERROR: unknown LLM_PROVIDER '$LLM_PROVIDER' (expected gemini|openai)" >&2; exit 1 ;;
  esac

  # Key (prompt if missing; offer to persist to .env)
  if [[ -z "${!KEY_VAR:-}" ]]; then
    echo "    $KEY_VAR is not set."
    read -rsp "    paste key (input hidden): " entered_key
    echo ""
    if [[ -z "$entered_key" ]]; then
      echo "ERROR: empty key. aborting." >&2
      exit 1
    fi
    export "$KEY_VAR=$entered_key"

    read -rp "    save $KEY_VAR (and LLM_PROVIDER) to .env? [y/N] " persist
    if [[ "$persist" == "y" || "$persist" == "Y" ]]; then
      # Avoid clobbering existing lines with a crude grep-and-append.
      touch ".env"
      grep -q "^LLM_PROVIDER=" .env || echo "LLM_PROVIDER=$LLM_PROVIDER" >> .env
      grep -q "^${KEY_VAR}="   .env || echo "${KEY_VAR}=${entered_key}"   >> .env
      echo "    written to .env (remember: .env should be in .gitignore)"
    fi
  else
    echo "    $KEY_VAR: present"
  fi
fi

# ---------------------------------------------------------------------------
# Step 5 — create a timestamped backup folder for this run.
# Every file we touch that already exists is copied here first.
# ---------------------------------------------------------------------------
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="Backups/pipeline_${STAMP}"
mkdir -p "$BACKUP_DIR"
echo ""
echo "=== backup dir: $BACKUP_DIR ==="

{
  echo "timestamp: $STAMP"
  echo "provider:  ${LLM_PROVIDER:-n/a}"
  echo "git_commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
  echo "selected steps:"
  for idx in "${SELECTED_STEPS[@]}"; do
    echo "  $((idx+1))) ${STEP_NAME[$idx]}  -> ${STEP_MODULE[$idx]}"
  done
  echo ""
  echo "git status at start:"
  git status --short 2>/dev/null || true
} > "$BACKUP_DIR/manifest.txt"

# ---------------------------------------------------------------------------
# Step 6 — overwrite-confirmation helpers.
# `OVERWRITE_ALL=1` short-circuits all further prompts once the user has
# answered Y (yes-to-all). `n` skips this step; `q` aborts the whole run.
# ---------------------------------------------------------------------------
OVERWRITE_ALL=0

backup_file() {
  local f="$1"
  if [[ -f "$f" ]]; then
    # Preserve timestamps and permissions so diffs stay meaningful.
    mkdir -p "$BACKUP_DIR/$(dirname "$f")"
    cp -p "$f" "$BACKUP_DIR/$f"
    echo "    backed up: $f"
  fi
}

# returns 0 = proceed (overwrite), 1 = skip this step, 2 = abort
prompt_overwrite() {
  local step_label="$1"
  local files_desc="$2"
  if (( OVERWRITE_ALL == 1 )); then
    echo "    [overwrite-all] proceeding with $step_label"
    return 0
  fi
  local reply
  read -rp "    overwrite ${files_desc}? [y = yes / N = skip / Y = yes to all / q = quit] " reply
  case "$reply" in
    y)     return 0 ;;
    Y)     OVERWRITE_ALL=1; return 0 ;;
    q|Q)   return 2 ;;
    *)     return 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# Step 7 — execute each selected step in pipeline order.
# For each step:
#   a) warn on missing inputs (non-fatal — the user may know they exist elsewhere)
#   b) back up every existing output to Backups/pipeline_<ts>/
#   c) if any output exists, prompt to overwrite (unless Y=yes-to-all is set)
#   d) for LLM steps whose primary output already exists, optionally pass
#      --no-resume to force a full regenerate
#   e) invoke the Python module
# ---------------------------------------------------------------------------
run_step() {
  local idx="$1"
  local name="${STEP_NAME[$idx]}"
  local module="${STEP_MODULE[$idx]}"
  local outputs="${STEP_OUTPUTS[$idx]}"
  local inputs="${STEP_INPUTS[$idx]}"
  local is_llm="${STEP_LLM[$idx]}"

  echo ""
  echo "=============================================================="
  echo "Step $((idx+1)): $name"
  echo "  module:  python -m $module"
  echo "  outputs: $outputs"
  echo "=============================================================="

  # (a) input warnings — show but don't block
  if [[ -n "$inputs" ]]; then
    local missing=()
    for f in $inputs; do
      [[ -f "$f" ]] || missing+=("$f")
    done
    if (( ${#missing[@]} > 0 )); then
      echo "    WARN: expected input file(s) missing:"
      for m in "${missing[@]}"; do echo "      - $m"; done
      echo "    (continuing — the module may fail if it really needs these)"
    fi
  fi

  # (b) backup existing outputs
  local any_existed=0
  for out in $outputs; do
    if [[ -f "$out" ]]; then
      any_existed=1
      backup_file "$out"
    fi
  done

  # (c) overwrite prompt (only if anything actually existed)
  if (( any_existed == 1 )); then
    prompt_overwrite "$name" "$outputs"
    local rc=$?
    case "$rc" in
      0) ;;                  # proceed
      1) echo "    skipped."; return 0 ;;
      2) echo "    aborting run."; exit 0 ;;
    esac
  else
    echo "    (no existing outputs — first run for this step)"
  fi

  # (d) LLM-only full-regenerate sub-prompt
  local extra_args=()
  if [[ "$is_llm" == "yes" ]]; then
    local primary="${outputs%% *}"
    if [[ -f "$primary" ]] || [[ -f "$BACKUP_DIR/$primary" ]]; then
      local reply
      read -rp "    full regenerate from scratch (--no-resume)? [y/N] " reply
      if [[ "$reply" == "y" || "$reply" == "Y" ]]; then
        extra_args+=(--no-resume)
      fi
    fi
  fi

  # (e) run — `-u` = unbuffered, so print() lines stream live to the terminal.
  # `stdbuf -oL -eL` on the outside also line-buffers anything the child
  # shells out to (best-effort: not present on every system, so we guard it).
  echo "    running: ${PYTHON_BIN[*]} -u -m $module ${extra_args[*]:-}"
  if command -v stdbuf >/dev/null 2>&1; then
    stdbuf -oL -eL "${PYTHON_BIN[@]}" -u -m "$module" "${extra_args[@]}"
  else
    "${PYTHON_BIN[@]}" -u -m "$module" "${extra_args[@]}"
  fi
  echo "    done: $name"
}

for idx in "${SELECTED_STEPS[@]}"; do
  run_step "$idx"
done

# ---------------------------------------------------------------------------
# Step 8 — final summary and rollback hint.
# ---------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "pipeline complete."
echo "  backup dir: $BACKUP_DIR"
echo "  rollback (copy back everything that was replaced):"
echo "    cp -pr \"$BACKUP_DIR\"/data/* data/"
echo "=============================================================="
