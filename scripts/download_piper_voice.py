"""
scripts/download_piper_voice.py
────────────────────────────────
Downloads a Piper TTS voice model into the project root.

Run:  python scripts/download_piper_voice.py
      python scripts/download_piper_voice.py en_US-ryan-high   ← specific voice

Default voice: en_US-amy-medium  (natural American English female)

Browse all available voices at:
  https://huggingface.co/rhasspy/piper-voices/tree/main

Voice naming convention:
  en_US-amy-medium   → language_REGION-speaker-quality
  Quality options: x_low, low, medium, high

Indian English voices:
  en_IN-x_low        → small & fast, Indian English accent

After downloading, set in .env:
  TTS_ENGINE=piper
  PIPER_VOICE=en_US-amy-medium   (or whichever you downloaded)
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

VOICE_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

DEFAULT_VOICE = "en_US-amy-medium"


def download_voice(voice_name: str, dest_dir: Path) -> None:
    """Download .onnx + .onnx.json files for a Piper voice."""
    # voice_name like "en_US-amy-medium" → lang folder "en/en_US"
    parts = voice_name.split("-")
    lang_code = parts[0]           # e.g. "en_US"
    lang_family = lang_code.split("_")[0]  # e.g. "en"

    base_path = f"{lang_family}/{lang_code}/{voice_name}"
    files = [
        f"{voice_name}.onnx",
        f"{voice_name}.onnx.json",
    ]

    dest_dir.mkdir(parents=True, exist_ok=True)

    for filename in files:
        dest_file = dest_dir / filename
        if dest_file.exists():
            print(f"  ✓ Already exists: {filename}")
            continue

        url = f"{VOICE_BASE_URL}/{base_path}/{filename}"
        print(f"  ⬇ Downloading {filename} …")

        def _progress(count, block_size, total_size):
            if total_size > 0:
                pct = min(100, count * block_size * 100 // total_size)
                print(f"\r    {pct}%", end="", flush=True)

        try:
            urllib.request.urlretrieve(url, dest_file, reporthook=_progress)
            print(f"\r  ✓ {filename} ({dest_file.stat().st_size // 1024} KB)")
        except Exception as exc:
            print(f"\r  ✗ Failed to download {filename}: {exc}")
            print(f"    URL tried: {url}")
            print("    Try browsing: https://huggingface.co/rhasspy/piper-voices/tree/main")
            raise


def main() -> None:
    voice = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VOICE
    dest = Path(".")  # save to project root (same place PiperTTS looks)

    print(f"\nDownloading Piper voice: {voice}")
    print(f"Destination: {dest.resolve()}\n")

    try:
        download_voice(voice, dest)
    except Exception:
        sys.exit(1)

    print(f"\n✅ Done! Add to your .env:\n")
    print(f"   TTS_ENGINE=piper")
    print(f"   PIPER_VOICE={voice}\n")


if __name__ == "__main__":
    main()
