"""
demo/console_demo.py
────────────────────
Phase 1 demo — text-only console conversation.

This is the "prove the brain works" step.  No audio, no phone, no Twilio.
You type answers; the agent prints replies.

WHAT YOU NEED TO RUN THIS:
  1. A .env file with GROQ_API_KEY set (copy .env.example → .env, fill it in)
  2. Dependencies installed:  pip install -e ".[dev]"   (or pip install groq python-dotenv pydantic pydantic-settings rich aiosqlite)

HOW TO RUN:
  python demo/console_demo.py

HOW TO EXIT EARLY:
  Type 'quit' or press Ctrl+C at any prompt.

WHAT HAPPENS AT THE END:
  The collected answers are saved to agent_calls.db (SQLite file in the project root).
  Open it with any SQLite viewer, or run:  python scripts/view_db.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── Make sure the project root is on the Python path ─────────────────────────
# (needed when you run this file directly without installing the package)
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

from agent.conversation import ConversationManager
from agent.storage import get_storage

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/console_demo.log", mode="a", encoding="utf-8"),
    ],
)
# Create logs/ directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

console = Console()


def print_agent(text: str) -> None:
    """Print the agent's words in a styled panel."""
    for line in text.split("\n"):
        if line.strip():
            console.print(f"[bold cyan]🤖 Agent:[/bold cyan] {line}")
        elif line == "":
            console.print()


def print_divider() -> None:
    console.print(Rule(style="dim"))


def main() -> None:
    console.print()
    console.print(
        Panel.fit(
            "[bold white]Phone-Calling AI Voice Agent[/bold white]\n"
            "[dim]Phase 1 — Console Demo (text only, no audio/telephony)[/dim]",
            border_style="cyan",
        )
    )
    console.print(
        "[dim]Tip: type [bold]quit[/bold] at any prompt to exit early.[/dim]\n"
    )
    print_divider()

    manager = ConversationManager()
    storage = get_storage()

    # ── Start the conversation ────────────────────────────────────────────────
    first_turn = manager.start()
    print_agent(first_turn.text)
    print_divider()

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session interrupted.[/dim]")
            break

        if user_input.lower() in {"quit", "exit", "q"}:
            console.print("[dim]Exiting early.[/dim]")
            break

        if not user_input:
            console.print("[dim]  (no input detected — please say something)[/dim]")
            continue

        response = manager.process_reply(user_input)
        print_divider()
        print_agent(response.text)
        print_divider()

        if response.is_final:
            break

    # ── Save results ──────────────────────────────────────────────────────────
    result = manager.get_result()

    console.print()
    console.print(
        Panel.fit(
            "[bold green]✅ Session complete[/bold green]\n"
            f"[dim]Call ID:[/dim] [bold]{result['call_id']}[/bold]\n"
            f"[dim]Turns:[/dim]   {result['turn_count']}",
            border_style="green",
        )
    )

    console.print("\n[bold]Collected data:[/bold]")
    for field, value in result["collected"].items():
        label = field.replace("_", " ").title()
        value_str = str(value) if value is not None else "[dim](empty)[/dim]"
        console.print(f"  [cyan]{label}:[/cyan] {value_str}")

    if result["missing_at_end"]:
        console.print(
            f"\n[yellow]⚠ Fields that need human follow-up:[/yellow] "
            f"{', '.join(result['missing_at_end'])}"
        )

    # Save to database
    call_id = storage.save_call(result)
    console.print(
        f"\n[dim]Saved to database.[/dim] "
        f"[dim]Run [bold]python scripts/view_db.py[/bold] to inspect.[/dim]"
    )
    console.print()


if __name__ == "__main__":
    main()
