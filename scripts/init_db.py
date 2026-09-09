"""
scripts/init_db.py
───────────────────
Creates the SQLite schema (idempotent — safe to run multiple times).

Run:  python scripts/init_db.py

You don't NEED to run this manually — the SQLiteBackend creates the schema
automatically on first use.  This script is here for:
  - CI pipelines that need the DB pre-created
  - Verifying the schema without running a demo
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.storage import SQLiteBackend
from rich.console import Console

console = Console()


def main() -> None:
    backend = SQLiteBackend()   # __init__ calls _ensure_schema()
    console.print("[green]✅ Database schema verified / created.[/green]")
    console.print(f"[dim]Database file: {backend._db_path.resolve()}[/dim]")


if __name__ == "__main__":
    main()
