"""
interfaces/audio.py
────────────────────
Microphone capture with simple energy-based Voice Activity Detection (VAD).

How it works:
  1. Listens to the mic in short chunks (100 ms each)
  2. Measures the volume (RMS) of each chunk
  3. When you start speaking (volume goes above threshold) — starts recording
  4. When you go silent for 1.5 seconds — stops recording and returns the audio

This gives a natural conversational feel — you just speak normally, and the
agent knows when you've finished without you pressing any button.

TUNING (if mic detection is too sensitive or not sensitive enough):
  - SILENCE_THRESHOLD: raise it (e.g. 0.02) if background noise triggers recording;
                       lower it (e.g. 0.005) if your mic is quiet and speech isn't detected
  - SILENCE_TIMEOUT:   seconds of silence before we assume you've finished speaking
  - MIN_SPEECH_SECS:   minimum amount of speech before we process (avoids processing coughs)
"""

from __future__ import annotations

import logging
import time

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# ── VAD tuning constants ──────────────────────────────────────────────────────
SAMPLE_RATE: int = 16_000      # Hz — 16kHz mono is what Groq Whisper expects
CHUNK_SECS: float = 0.1        # each read chunk = 100ms
CHUNK_SIZE: int = int(SAMPLE_RATE * CHUNK_SECS)

SILENCE_THRESHOLD: float = 0.008   # RMS below this = silence
SILENCE_TIMEOUT: float = 1.5       # seconds of silence → stop listening
MIN_SPEECH_SECS: float = 0.4       # minimum speech length to bother transcribing
MAX_RECORD_SECS: float = 30.0      # hard cap — don't record forever


class AudioRecorder:
    """
    Records from the default microphone until the speaker goes quiet.

    Returns a numpy float32 array at 16kHz mono.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        silence_threshold: float = SILENCE_THRESHOLD,
        silence_timeout: float = SILENCE_TIMEOUT,
        min_speech_secs: float = MIN_SPEECH_SECS,
        max_record_secs: float = MAX_RECORD_SECS,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_timeout = silence_timeout
        self.min_speech_secs = min_speech_secs
        self.max_record_secs = max_record_secs

    def record(self, status_callback=None) -> np.ndarray | None:
        """
        Block until the caller speaks and then goes quiet.

        Args:
            status_callback: optional callable(str) called with status messages
                             like "Listening…" and "Got it, processing…"
                             (used by the demo UI to print status lines)

        Returns:
            numpy float32 array (16kHz mono) of the spoken audio, or
            None if nothing was captured (silence / too short).
        """
        def _status(msg: str) -> None:
            if status_callback:
                status_callback(msg)
            else:
                logger.debug(msg)

        chunks: list[np.ndarray] = []
        silent_chunk_count = 0
        max_silent_chunks = int(self.silence_timeout / CHUNK_SECS)
        min_speech_chunks = int(self.min_speech_secs / CHUNK_SECS)
        max_total_chunks = int(self.max_record_secs / CHUNK_SECS)

        speech_started = False
        _status("🎤 Listening… (speak now)")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SIZE,
            ) as stream:
                start_time = time.monotonic()

                while True:
                    chunk, _ = stream.read(CHUNK_SIZE)
                    chunk = chunk.flatten()
                    rms = float(np.sqrt(np.mean(chunk ** 2)))

                    if rms > self.silence_threshold:
                        if not speech_started:
                            _status("🔴 Recording…")
                        speech_started = True
                        silent_chunk_count = 0
                        chunks.append(chunk)
                    else:
                        if speech_started:
                            silent_chunk_count += 1
                            chunks.append(chunk)  # include trailing silence
                            if silent_chunk_count >= max_silent_chunks:
                                break   # enough silence → done
                        # else: still waiting for speech to start

                    # Hard cap
                    if len(chunks) >= max_total_chunks:
                        logger.warning("Hit max recording length (%ss)", self.max_record_secs)
                        break

        except sd.PortAudioError as exc:
            logger.error("Microphone error: %s", exc)
            raise

        if not chunks or len(chunks) < min_speech_chunks:
            _status("⚠️  No speech detected — try speaking a bit louder")
            return None

        _status("✅ Got it — processing…")
        audio = np.concatenate(chunks)
        logger.debug("Recorded %.2f seconds of audio", len(audio) / self.sample_rate)
        return audio
