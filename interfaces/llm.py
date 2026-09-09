"""
interfaces/llm.py
──────────────────
LLM interface + Groq implementation.

The abstract base class (LLMBackend) defines one method:
    complete(messages) → str

This wraps the same Groq client already used by agent/extractor.py, but
as a clean interface so Phase 3 (Pipecat pipeline) can inject it separately.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from groq import Groq

from agent.config import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────────────────────────────────────

class LLMBackend(ABC):
    """Every LLM backend must implement complete()."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        """
        Run a chat completion.

        Args:
            messages:    list of {"role": "system"|"user"|"assistant", "content": "..."}
            temperature: 0.0 = deterministic, 1.0 = creative
            max_tokens:  max tokens in the response

        Returns:
            The assistant's reply as a plain string.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Groq LLM implementation
# ─────────────────────────────────────────────────────────────────────────────

class GroqLLM(LLMBackend):
    """
    Groq-hosted Llama (or whatever the best free-tier model is).
    Free tier: ~30 requests/minute, no credit card.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.llm_model

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()
