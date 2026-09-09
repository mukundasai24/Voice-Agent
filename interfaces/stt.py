"""
interfaces/stt.py
──────────────────
Speech-to-Text interface + Groq Whisper implementation.

The abstract base class (STTBackend) defines one method:
    transcribe(audio_array, sample_rate) → str

Swapping STT providers = write a new class, change one line in mic_demo.py.
"""

from __future__ import annotations

import io
import logging
import wave
from abc import ABC, abstractmethod

import numpy as np
from groq import Groq

from agent.config import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────────────────────────────────────

class STTBackend(ABC):
    """Every STT backend must implement transcribe()."""

    @abstractmethod
    def transcribe(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """
        Convert spoken audio into text.

        Args:
            audio_array: float32 numpy array, values in [-1, 1], mono
            sample_rate: samples per second (typically 16000)

        Returns:
            The transcribed text string (empty string if nothing detected).
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Groq Whisper implementation
# ─────────────────────────────────────────────────────────────────────────────

class GroqWhisperSTT(STTBackend):
    """
    Uses Groq's hosted Whisper-large-v3 for fast, accurate transcription.

    Free tier: ~2,000 requests/day, no credit card needed.
    Latency: typically 200–500ms even for 30s of audio — fast enough for
    natural conversation pacing.
    """

    def __init__(self, language: str = "en") -> None:
        """
        Args:
            language: ISO 639-1 language code hint for Whisper.
                      "en" for English, "hi" for Hindi, etc.
                      Leave as "en" for best accuracy on English calls.
        """
        settings = get_settings()
        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.stt_model
        self._language = language

    def transcribe(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """Send audio to Groq Whisper, return the transcript."""
        if audio_array is None or len(audio_array) == 0:
            return ""

        wav_bytes = self._numpy_to_wav(audio_array, sample_rate)

        logger.debug(
            "Sending %.2fs of audio to Groq Whisper (%s)",
            len(audio_array) / sample_rate,
            self._model,
        )

        try:
            # Groq's transcription endpoint — docs: console.groq.com/docs/speech-text
            transcription = self._client.audio.transcriptions.create(
                model=self._model,
                file=("audio.wav", wav_bytes, "audio/wav"),
                response_format="text",
                language=self._language,
            )
            text = str(transcription).strip()
            logger.info("Transcribed: %r", text)
            return text

        except Exception as exc:
            logger.error("Groq Whisper transcription failed: %s", exc)
            return ""

    @staticmethod
    def _numpy_to_wav(audio_array: np.ndarray, sample_rate: int) -> bytes:
        """
        Convert a float32 numpy array to a WAV byte string.

        Groq expects a standard WAV file — 16-bit PCM, mono, 16kHz.
        """
        # Convert float32 [-1, 1] → int16 [-32768, 32767]
        audio_int16 = np.clip(audio_array * 32767, -32768, 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)       # mono
            wf.setsampwidth(2)       # 16-bit = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        buf.seek(0)
        return buf.read()
