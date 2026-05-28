"""
Load Phase 1 lookup data into Supabase.

Populates three tables used by retriever.py's structured lookup path:
  spec_fields       — one row per field/register (from fields.json)
  spec_field_index  — one row per field-name/location pair (from field_index.json)
  spec_tables       — one row per spec table (from tables.json)

Run the DDL in scripts/supabase_schema.sql first (once), then run this script
whenever the Phase 1 data changes.

Requires env vars (or .env file):
  SUPABASE_URL  — https://xxxxx.supabase.co
  SUPABASE_KEY  — service_role key

Run:
  python scripts/load_lookup_data.py
  python scripts/load_lookup_data.py --data-dir /path/to/data
  python scripts/load_lookup_data.py --tables fields          # one table only
  python scripts/load_lookup_data.py --tables fields tables   # subset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from supabase import create_client
except ImportError:
    print("Missing dependency: pip install supabase")
    sys.exit(1)


BATCH_SIZE = 200

ALL_TABLES = ("fields", "field_index", "tables")


# ---------------------------------------------------------------------------
# Env / client

def _load_env_var(name: str) -> str | None:
    val = os.environ.get(name)
    if val:
        return val
    env_path = Path(".env")
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _get_client():
    url = _load_env_var("SUPABASE_URL")
    key = _load_env_var("SUPABASE_KEY")
    if not url or not key:
        print("ERROR: Set SUPABASE_URL and SUPABASE_KEY (env vars or .env file)")
        sys.exit(1)
    return create_client(url, key)


# ---------------------------------------------------------------------------
# Row builders

def _fields_row(record: dict) -> dict:
    return {
        "name":          str(record.get("field_name") or "").upper().strip(),
        "description":   record.get("description"),
        "offset":        str(record.get("offset") or "") or None,
        "figure_number": str(record.get("parent_figure") or "") or None,
        "section_id":    record.get("section_id"),
        "data":          record,
    }


def _field_index_rows(field_index: dict) -> list[dict]:
    """Flatten field_index dict (name → [records]) into rows."""
    rows: list[dict] = []
    for field_name, entries in field_index.items():
        if not isinstance(entries, list):
            entries = [entries]
        for entry in entries:
            rows.append({
                "field_name":    field_name.upper().strip(),
                "section_id":    entry.get("section_id"),
                "figure_number": str(entry.get("figure_number") or "") or None,
                "data":          entry,
            })
    return rows


def _tables_row(record: dict) -> dict | None:
    fig = record.get("figure_number")
    if fig is None:
        return None
    return {
        "figure_number": str(fig),
        "title":         record.get("title"),
        "section_id":    record.get("section_id"),
        "raw_text":      record.get("raw_text"),
        "table_json":    record.get("table_json"),
        "data":          record,
    }


# ---------------------------------------------------------------------------
# Upload helpers

def _upsert_batched(client, table: str, rows: list[dict], conflict_col: str | None = None) -> int:
    total = len(rows)
    batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    uploaded = 0
    for i in range(0, total, BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch = rows[i : i + BATCH_SIZE]
        if conflict_col:
            client.table(table).upsert(batch, on_conflict=conflict_col).execute()
        else:
            client.table(table).upsert(batch).execute()
        uploaded += len(batch)
        print(f"  batch {batch_num}/{batches} — {uploaded}/{total}")
    return uploaded


def load_fields(client, data_dir: Path) -> None:
    path = data_dir / "fields.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    rows = [_fields_row(r) for r in records if r.get("field_name")]
    # Remove rows with empty name after normalization
    rows = [r for r in rows if r["name"]]
    # The same field name can appear in multiple registers; spec_fields uses
    # name as PRIMARY KEY, so keep the first occurrence. field_index.json is
    # the authoritative store for "same name, multiple locations".
    seen: set = set()
    deduped: list[dict] = []
    for r in rows:
        if r["name"] in seen:
            continue
        seen.add(r["name"])
        deduped.append(r)
    rows = deduped
    print(f"spec_fields: upserting {len(rows)} rows...")
    n = _upsert_batched(client, "spec_fields", rows, conflict_col="name")
    print(f"  done — {n} rows")


def load_field_index(client, data_dir: Path) -> None:
    path = data_dir / "field_index.json"
    field_index = json.loads(path.read_text(encoding="utf-8"))
    rows = _field_index_rows(field_index)
    print(f"spec_field_index: replacing {len(rows)} rows...")
    # Delete all then insert — no natural unique key across (name, section, figure)
    client.table("spec_field_index").delete().neq("id", 0).execute()
    n = _upsert_batched(client, "spec_field_index", rows)
    print(f"  done — {n} rows")


def load_tables(client, data_dir: Path) -> None:
    path = data_dir / "tables.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in (_tables_row(rec) for rec in records) if r is not None]
    print(f"spec_tables: upserting {len(rows)} rows...")
    n = _upsert_batched(client, "spec_tables", rows, conflict_col="figure_number")
    print(f"  done — {n} rows")


# ---------------------------------------------------------------------------
# Main

def run(data_dir: Path, tables: list[str]) -> None:
    client = _get_client()

    loaders = {
        "fields":      load_fields,
        "field_index": load_field_index,
        "tables":      load_tables,
    }

    for name in tables:
        path_map = {
            "fields":      data_dir / "fields.json",
            "field_index": data_dir / "field_index.json",
            "tables":      data_dir / "tables.json",
        }
        if not path_map[name].exists():
            print(f"ERROR: {path_map[name]} not found — skipping {name}")
            continue
        loaders[name](client, data_dir)

    print("\nAll done.")


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Load Phase 1 lookup data into Supabase.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("SPEC_DATA_DIR", "data")),
        help="Directory containing fields.json, field_index.json, tables.json "
             "(defaults to $SPEC_DATA_DIR or 'data')",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=list(ALL_TABLES),
        default=list(ALL_TABLES),
        help="Which tables to load (default: all)",
    )
    args = parser.parse_args(argv)
    run(args.data_dir, args.tables)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
