"""
agent/config.py
───────────────
Single source of truth for all settings.

Reads from .env via python-dotenv.  Every other module imports from here
instead of reading os.environ directly — that way changing a key name only
needs one edit in one place.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings  # pip install pydantic-settings

# Load .env from the project root (works whether you run from root or a subdir)
load_dotenv()


class Settings(BaseSettings):
    """All configurable values, with sensible defaults where possible."""

    # ── Groq (required) ───────────────────────────────────────────────────────
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    llm_model: str = Field("llama-3.3-70b-versatile", alias="LLM_MODEL")
    stt_model: str = Field("whisper-large-v3", alias="STT_MODEL")

    # ── Twilio (required only for Phase 3) ────────────────────────────────────
    twilio_account_sid: str = Field("", alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = Field("", alias="TWILIO_AUTH_TOKEN")
    twilio_phone_number: str = Field("", alias="TWILIO_PHONE_NUMBER")

    # ── TTS ───────────────────────────────────────────────────────────────────
    tts_engine: str = Field("piper", alias="TTS_ENGINE")  # "piper" or "elevenlabs"
    elevenlabs_api_key: str = Field("", alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field("", alias="ELEVENLABS_VOICE_ID")

    # ── Storage ───────────────────────────────────────────────────────────────
    database_url: str = Field("sqlite:///agent_calls.db", alias="DATABASE_URL")

    # ── Debug ─────────────────────────────────────────────────────────────────
    debug_llm: bool = Field(False, alias="DEBUG_LLM")

    model_config = {"populate_by_name": True, "env_file": ".env"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (reads .env once at startup)."""
    return Settings()
