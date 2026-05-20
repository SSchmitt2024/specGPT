"""
Apply scripts/supabase_schema.sql to the Supabase Postgres database.

Uses a direct Postgres connection (psycopg2) — the Supabase Python client
only speaks REST and cannot run DDL. Safe to re-run: every statement in
the schema uses IF NOT EXISTS / OR REPLACE.

Required env var (or .env):
  DATABASE_URL — postgres connection string from Supabase dashboard
                 Project Settings → Database → Connection string (URI)
                 e.g. postgresql://postgres:[password]@db.xxxx.supabase.co:5432/postgres

Run:
  python scripts/apply_schema.py
  python scripts/apply_schema.py --dry-run     # print SQL, no DB changes
  python scripts/apply_schema.py --schema path/to/other.sql
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import psycopg2
except ImportError:
    print("Missing dependency: pip install psycopg2-binary")
    sys.exit(1)

DEFAULT_SCHEMA = Path(__file__).parent / "supabase_schema.sql"


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


def apply(schema_path: Path, dry_run: bool = False) -> None:
    sql = schema_path.read_text(encoding="utf-8")

    if dry_run:
        print(f"--- DRY RUN: would execute {schema_path} ---")
        print(sql)
        return

    url = _load_env_var("DATABASE_URL")
    if not url:
        print(
            "ERROR: DATABASE_URL not set.\n"
            "  Find it in Supabase: Project Settings → Database → Connection string (URI)"
        )
        sys.exit(1)

    print(f"Connecting to database...")
    conn = psycopg2.connect(url)
    conn.autocommit = True

    try:
        with conn.cursor() as cur:
            print(f"Applying {schema_path}...")
            cur.execute(sql)
        print("Schema applied successfully.")
    finally:
        conn.close()


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Apply Supabase schema SQL.")
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help=f"Path to SQL file (default: {DEFAULT_SCHEMA})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print SQL without executing",
    )
    args = parser.parse_args(argv)

    if not args.schema.exists():
        print(f"ERROR: schema file not found: {args.schema}")
        return 1

    apply(args.schema, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
