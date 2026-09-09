"""
scripts/view_db.py
───────────────────
Quick utility to print the last N saved calls from the local SQLite database.

Run:  python scripts/view_db.py
      python scripts/view_db.py 5       ← show last 5 calls (default: 10)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from agent.storage import SQLiteBackend

console = Console()


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    backend = SQLiteBackend()
    calls = backend.list_calls(limit=limit)

    if not calls:
        console.print("[yellow]No calls saved yet.[/yellow]")
        return

    console.print(f"\n[bold]Last {len(calls)} call(s) in agent_calls.db:[/bold]\n")

    for call in calls:
        table = Table(title=f"Call {call['call_id'][:8]}…", show_header=True, header_style="bold cyan")
        table.add_column("Field", style="cyan")
        table.add_column("Value")

        table.add_row("call_id",    call["call_id"])
        table.add_row("started_at", call.get("started_at", ""))
        table.add_row("ended_at",   call.get("ended_at", ""))
        table.add_row("turns",      str(call.get("turn_count", "")))
        table.add_row("missing",    ", ".join(call.get("missing_at_end", [])) or "none")

        for field, value in call.get("collected", {}).items():
            label = field.replace("_", " ").title()
            table.add_row(f"  {label}", str(value) if value else "(empty)")

        console.print(table)
        console.print()


if __name__ == "__main__":
    main()
