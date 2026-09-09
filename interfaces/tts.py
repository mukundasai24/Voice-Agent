"""
interfaces/tts.py
──────────────────
Text-to-Speech interface with three implementations:

  1. SystemTTS    — uses Windows built-in voices (pyttsx3), zero setup required.
                    Works immediately. Voice quality is basic but perfectly usable.

  2. PiperTTS     — local, offline, MIT-licensed, much better voice quality.
                    Requires: pip install piper-tts + downloading a voice model file.
                    See README for download instructions.

  3. ElevenLabsTTS — cloud TTS, best voice quality.
                    Requires: ELEVENLABS_API_KEY in .env, 10k chars/month free.

Which one is used is controlled by TTS_ENGINE in your .env file:
  TTS_ENGINE=system      ← default, works with no setup
  TTS_ENGINE=piper       ← better quality, needs voice download
  TTS_ENGINE=elevenlabs  ← best quality, needs API key

The factory function get_tts() at the bottom of this file reads TTS_ENGINE and
returns the right backend automatically.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from abc import ABC, abstractmethod

import numpy as np
import sounddevice as sd

from agent.config import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────────────────────────────────────

class TTSBackend(ABC):
    """Every TTS backend must implement speak()."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """
        Convert text to speech and play it through the speakers.
        Blocks until playback is complete.
        """
        ...

    def stop(self) -> None:
        """Interrupt playback (for barge-in). Override in implementations that support it."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. System TTS (pyttsx3 — Windows/Mac/Linux built-in voices)
# ─────────────────────────────────────────────────────────────────────────────

class SystemTTS(TTSBackend):
    """
    Uses the operating system's built-in text-to-speech.
    On Windows: SAPI5 voices (David, Zira, etc.)
    On Mac: say command
    On Linux: espeak

    Zero setup — works immediately after `pip install pyttsx3`.
    """

    def __init__(self) -> None:
        import pyttsx3  # type: ignore[import]
        self._engine = pyttsx3.init()
        # Slightly faster than the default speed (150 wpm is the default)
        self._engine.setProperty("rate", 165)
        self._engine.setProperty("volume", 0.95)

        # On Windows, prefer Zira (female) if available
        voices = self._engine.getProperty("voices")
        for v in voices:
            if "zira" in v.name.lower() or "female" in v.name.lower():
                self._engine.setProperty("voice", v.id)
                break

        logger.info("SystemTTS initialized (engine: %s)", self._engine)

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        logger.debug("SystemTTS speaking: %r", text[:60])
        self._engine.say(text)
        self._engine.runAndWait()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Piper TTS (local, offline, MIT-licensed)
# ─────────────────────────────────────────────────────────────────────────────

class PiperTTS(TTSBackend):
    """
    Local, offline TTS using Piper (https://github.com/rhasspy/piper).
    Much better voice quality than SystemTTS, still completely free forever.

    SETUP (do this once):
    ─────────────────────
    1. Install:  pip install piper-tts
    2. Download a voice model into the project root:
       python -c "from piper.download import ensure_voice_exists, get_voices, find_voice; import requests; ..."

    Easier — run the helper script we provide:
       python scripts/download_piper_voice.py

    The default voice is "en_US-amy-medium" (natural American English female).
    You can browse all voices at: https://huggingface.co/rhasspy/piper-voices

    HOW TO SWITCH VOICES:
    ─────────────────────
    Set PIPER_VOICE in .env:
       PIPER_VOICE=en_US-ryan-high        ← American English male, high quality
       PIPER_VOICE=en_GB-alan-medium      ← British English male
       PIPER_VOICE=en_IN-x_low            ← Indian English (small, fast model)
    """

    DEFAULT_VOICE = "en_US-amy-medium"

    def __init__(self, voice: str | None = None, models_dir: str = ".") -> None:
        """
        Args:
            voice:      Piper voice name (e.g. "en_US-amy-medium").
                        Falls back to PIPER_VOICE env var, then DEFAULT_VOICE.
            models_dir: Directory where .onnx voice model files are stored.
        """
        try:
            from piper import PiperVoice  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Piper TTS requires: pip install piper-tts\n"
                "Then download a voice: python scripts/download_piper_voice.py"
            ) from exc

        import os
        self._voice_name = voice or os.environ.get("PIPER_VOICE", self.DEFAULT_VOICE)
        model_path = f"{models_dir}/{self._voice_name}.onnx"

        if not __import__("pathlib").Path(model_path).exists():
            raise FileNotFoundError(
                f"Piper voice model not found: {model_path}\n"
                f"Run: python scripts/download_piper_voice.py\n"
                f"Or set TTS_ENGINE=system in .env to use the built-in voice instead."
            )

        self._voice = PiperVoice.load(model_path)
        self._sample_rate = self._voice.config.sample_rate
        self._stop_event = threading.Event()
        logger.info("PiperTTS loaded voice: %s (%d Hz)", self._voice_name, self._sample_rate)

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        self._stop_event.clear()
        logger.debug("PiperTTS speaking: %r", text[:60])

        audio_chunks: list[bytes] = []
        for audio_bytes in self._voice.synthesize_stream_raw(text):
            if self._stop_event.is_set():
                break
            audio_chunks.append(audio_bytes)

        if not audio_chunks or self._stop_event.is_set():
            return

        raw = b"".join(audio_chunks)
        audio_array = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        sd.play(audio_array, samplerate=self._sample_rate)
        # Poll so we can interrupt (barge-in)
        while sd.get_stream().active:
            if self._stop_event.is_set():
                sd.stop()
                break
            time.sleep(0.05)

    def stop(self) -> None:
        """Stop playback immediately (used for barge-in)."""
        self._stop_event.set()
        sd.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 3. ElevenLabs TTS (cloud, best quality, 10k chars/month free)
# ─────────────────────────────────────────────────────────────────────────────

class ElevenLabsTTS(TTSBackend):
    """
    Cloud TTS via ElevenLabs API — best voice quality of the three options.
    Free tier: 10,000 characters/month (≈10 minutes of audio).

    SETUP:
    ──────
    1. Sign up at: https://elevenlabs.io (free, no card)
    2. Get your API key from: https://elevenlabs.io/api
    3. Add to .env:
         ELEVENLABS_API_KEY=your_key_here
         ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM   ← "Rachel" voice (default)
         TTS_ENGINE=elevenlabs
    """

    def __init__(self) -> None:
        try:
            from elevenlabs.client import ElevenLabs  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                'ElevenLabs TTS requires: pip install "voice-agent[elevenlabs]"'
            ) from exc

        settings = get_settings()
        if not settings.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY not set in .env")

        self._client = ElevenLabs(api_key=settings.elevenlabs_api_key)
        self._voice_id = settings.elevenlabs_voice_id or "21m00Tcm4TlvDq8ikWAM"
        self._stop_event = threading.Event()
        logger.info("ElevenLabsTTS initialized, voice_id=%s", self._voice_id)

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        self._stop_event.clear()
        logger.debug("ElevenLabsTTS speaking: %r", text[:60])

        audio_stream = self._client.generate(
            text=text,
            voice=self._voice_id,
            model="eleven_turbo_v2",
            stream=True,
        )
        audio_chunks: list[bytes] = []
        for chunk in audio_stream:
            if self._stop_event.is_set():
                return
            if chunk:
                audio_chunks.append(chunk)

        if not audio_chunks:
            return

        raw = b"".join(audio_chunks)
        # ElevenLabs returns MP3 — convert via numpy (requires soundfile or pydub)
        try:
            import soundfile as sf  # type: ignore[import]
            audio_array, sr = sf.read(io.BytesIO(raw))
        except Exception:
            logger.error("Failed to decode ElevenLabs audio — install soundfile: pip install soundfile")
            return

        sd.play(audio_array.astype(np.float32), samplerate=sr)
        while sd.get_stream().active:
            if self._stop_event.is_set():
                sd.stop()
                break
            time.sleep(0.05)

    def stop(self) -> None:
        self._stop_event.set()
        sd.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Factory — returns the right TTS backend based on TTS_ENGINE in .env
# ─────────────────────────────────────────────────────────────────────────────

def get_tts() -> TTSBackend:
    """
    Return the TTS backend specified by TTS_ENGINE in .env.

    TTS_ENGINE=system      → SystemTTS  (default — works with no setup)
    TTS_ENGINE=piper       → PiperTTS   (better quality, needs voice download)
    TTS_ENGINE=elevenlabs  → ElevenLabsTTS (best quality, needs API key)
    """
    engine = get_settings().tts_engine.lower().strip()

    if engine == "piper":
        return PiperTTS()
    elif engine == "elevenlabs":
        return ElevenLabsTTS()
    else:
        if engine not in ("system", ""):
            logger.warning("Unknown TTS_ENGINE=%r — falling back to system TTS", engine)
        return SystemTTS()
