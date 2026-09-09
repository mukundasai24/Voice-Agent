"""
demo/mic_demo.py
─────────────────
Phase 2 demo — real speech in, real speech out.

You speak → Groq Whisper hears you → ConversationManager decides → TTS speaks back.

WHAT YOU NEED TO RUN THIS:
  - GROQ_API_KEY in .env (same key as Phase 1)
  - sounddevice installed:  pip install sounddevice numpy pyttsx3
  - A working microphone and speakers

HOW TO RUN:
  python demo/mic_demo.py

HOW IT WORKS:
  1. Agent speaks the greeting
  2. You see "Listening…" — speak your answer naturally
  3. When you go quiet for ~1.5 seconds, it stops recording
  4. Groq Whisper turns your speech into text (takes ~300ms)
  5. The ConversationManager decides what to say next
  6. The TTS speaks the response
  7. Repeat until all questions are answered

WHICH VOICE IS USED:
  Default: TTS_ENGINE=system  → Windows built-in voice (Zira/David)
  Better:  TTS_ENGINE=piper   → local Piper TTS (see SETUP_CHECKLIST.md for download)
  Best:    TTS_ENGINE=elevenlabs → ElevenLabs cloud (needs API key)

TROUBLESHOOTING:
  "No speech detected"      → speak louder, or lower SILENCE_THRESHOLD in interfaces/audio.py
  "PortAudioError"          → check your mic is set as the default recording device in Windows
  "ModuleNotFoundError"     → run: pip install sounddevice numpy pyttsx3
  Transcription is wrong    → try speaking more slowly and clearly; background noise matters a lot
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# ── Make sure project root is on the Python path ────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from agent.conversation import ConversationManager
from agent.storage import get_storage
from interfaces.audio import AudioRecorder
from interfaces.stt import GroqWhisperSTT
from interfaces.tts import get_tts

# ── Logging setup ─────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/mic_demo.log", mode="a", encoding="utf-8"),
    ],
)

console = Console()


def print_agent(text: str) -> None:
    for line in text.split("\n"):
        if line.strip():
            console.print(f"[bold cyan]🤖 Agent:[/bold cyan] {line}")
        elif line == "":
            console.print()


def print_user(text: str) -> None:
    console.print(f"[bold green]👤 You:[/bold green]   {text}")


def print_status(text: str) -> None:
    console.print(f"[dim]{text}[/dim]")


def print_divider() -> None:
    console.print(Rule(style="dim"))


def main() -> None:
    console.print()
    console.print(
        Panel.fit(
            "[bold white]Phone-Calling AI Voice Agent[/bold white]\n"
            "[dim]Phase 2 — Mic + Speaker Demo[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # ── Initialise components ─────────────────────────────────────────────────
    console.print("[dim]Initialising TTS…[/dim]", end=" ")
    try:
        tts = get_tts()
        console.print("[green]✓[/green]")
    except Exception as exc:
        console.print(f"[red]✗ {exc}[/red]")
        console.print(
            "[yellow]Tip: set TTS_ENGINE=system in .env for zero-setup voice[/yellow]"
        )
        return

    console.print("[dim]Initialising STT (Groq Whisper)…[/dim]", end=" ")
    try:
        stt = GroqWhisperSTT()
        console.print("[green]✓[/green]")
    except Exception as exc:
        console.print(f"[red]✗ {exc}[/red]")
        return

    recorder = AudioRecorder()
    manager = ConversationManager()
    storage = get_storage()

    console.print()
    console.print("[dim]Press [bold]Ctrl+C[/bold] at any time to end the session early.[/dim]")
    console.print()

    # ── Start conversation ─────────────────────────────────────────────────────
    first_turn = manager.start()
    print_divider()
    print_agent(first_turn.text)
    tts.speak(first_turn.text)
    print_divider()

    # ── Main conversation loop ─────────────────────────────────────────────────
    try:
        while True:
            # 1. Record from mic
            audio = None
            while audio is None:
                audio = recorder.record(status_callback=print_status)
                if audio is None:
                    # Nothing detected — play a soft prompt and try again
                    nudge = "I didn't catch that — could you say that again?"
                    print_agent(nudge)
                    tts.speak(nudge)

            # 2. Transcribe
            t0 = time.monotonic()
            transcript = stt.transcribe(audio, recorder.sample_rate)
            elapsed_ms = int((time.monotonic() - t0) * 1000)

            if not transcript:
                console.print("[dim]  (transcription empty — trying again)[/dim]")
                continue

            print_user(transcript)
            console.print(f"[dim]  (transcribed in {elapsed_ms}ms)[/dim]")

            # 3. ConversationManager decides the reply
            response = manager.process_reply(transcript)
            print_divider()
            print_agent(response.text)

            # 4. Speak the response
            tts.speak(response.text)
            print_divider()

            # 5. Check if done
            if response.is_final:
                break

    except KeyboardInterrupt:
        console.print("\n[dim]Session ended by user.[/dim]")
        tts.stop()

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
            f"\n[yellow]⚠  Fields for human follow-up:[/yellow] "
            f"{', '.join(result['missing_at_end'])}"
        )

    storage.save_call(result)
    console.print(
        "\n[dim]Saved to database. Run [bold]python scripts/view_db.py[/bold] to inspect.[/dim]\n"
    )


if __name__ == "__main__":
    main()
