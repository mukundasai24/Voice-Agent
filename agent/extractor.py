"""
agent/extractor.py
──────────────────
The LLM-based extraction step.

For every caller turn, this sends:
  - the current field being asked about (+ its clarify_hint)
  - the caller's raw spoken/typed reply
  - recent conversation context

…to the Groq LLM and gets back a small JSON object:
  {
    "value": <the extracted answer, or null>,
    "confidence": <0.0–1.0>,
    "clarifying_question": <a targeted follow-up if confidence is low, or null>
  }

This is the heart of the "smart" part — it understands varied phrasings
("yeah sometime next week probably" → confidence 0.4, clarifying_question = "…")
instead of trying to keyword-match.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from groq import Groq

from agent.config import get_settings
from agent.questions import get_question

logger = logging.getLogger(__name__)

# ── Result type returned by the extractor ────────────────────────────────────

@dataclass
class ExtractionResult:
    value: Any            # the extracted answer (None if not extractable)
    confidence: float     # 0.0 = no idea,  1.0 = certain
    clarifying_question: str | None  # suggested follow-up (None if confidence ≥ threshold)
    raw_llm_response: str = ""       # kept for debugging / logging


# ── Confidence threshold ──────────────────────────────────────────────────────
# Below this, the agent will ask the clarifying question instead of accepting the answer.
CONFIDENCE_THRESHOLD = 0.75

# Max clarification attempts per field before we flag it and move on.
MAX_CLARIFY_ATTEMPTS = 2


class Extractor:
    """
    Wraps the Groq LLM call that turns a raw caller reply into a typed,
    validated value.

    To swap the LLM provider, replace the `_call_llm` method — everything
    else stays the same.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.llm_model
        self._debug = settings.debug_llm

    def extract(
        self,
        field_name: str,
        caller_reply: str,
        conversation_so_far: list[dict[str, str]],
    ) -> ExtractionResult:
        """
        Extract the value for `field_name` from the caller's `caller_reply`.

        Args:
            field_name:          The field we're currently asking about.
            caller_reply:        The raw text of what the caller said.
            conversation_so_far: A list of {"role": "assistant"|"user", "content": "..."}
                                 dicts representing the conversation history so far
                                 (used for context, e.g. the caller said "same as before").

        Returns:
            ExtractionResult with the extracted value, confidence, and optional follow-up.
        """
        question_def = get_question(field_name)
        if question_def is None:
            raise ValueError(f"No question definition found for field '{field_name}'")

        prompt = self._build_system_prompt(field_name, question_def)
        messages = self._build_messages(prompt, conversation_so_far, caller_reply)

        raw = self._call_llm(messages)
        return self._parse_response(raw)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_system_prompt(self, field_name: str, question_def: dict) -> str:
        return f"""You are an extraction assistant for a phone-based AI voice agent.

Your job: given the caller's reply, extract the value for the field "{field_name}".

Field description: {question_def['clarify_hint']}
Example of a valid value: {question_def['example']}

RULES:
1. Reply ONLY with a valid JSON object — no markdown, no explanation, nothing else.
2. The JSON must have exactly these three keys:
   - "value": the extracted value (string, number, or null if truly unextractable)
   - "confidence": a float between 0.0 (no idea) and 1.0 (certain)
   - "clarifying_question": if confidence < {CONFIDENCE_THRESHOLD}, write ONE short, 
     targeted follow-up question to ask the caller (focus on the ambiguous part only, 
     do NOT repeat the original question verbatim). If confidence >= {CONFIDENCE_THRESHOLD}, 
     set this to null.
3. Be generous: if the caller's intent is clear even if phrased oddly, extract it with 
   high confidence. Only flag low confidence for genuinely ambiguous replies.
4. Never invent information not present in the caller's reply.
5. Keep the clarifying question short — one sentence, conversational, friendly.

Example output (high confidence):
{{"value": "Priya Sharma", "confidence": 0.98, "clarifying_question": null}}

Example output (low confidence):
{{"value": null, "confidence": 0.3, "clarifying_question": "Did you mean this coming Monday, or a different week?"}}
"""

    def _build_messages(
        self,
        system_prompt: str,
        conversation_so_far: list[dict[str, str]],
        caller_reply: str,
    ) -> list[dict[str, str]]:
        """Assemble the full message list to send to the LLM."""
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

        # Add the recent conversation for context (last 6 turns max to keep tokens low)
        for msg in conversation_so_far[-6:]:
            messages.append(msg)

        # The caller's latest reply as the final user message
        messages.append({"role": "user", "content": caller_reply})
        return messages

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call the Groq API and return the raw text response."""
        if self._debug:
            logger.debug("LLM REQUEST:\n%s", json.dumps(messages, indent=2))

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.1,    # low temperature → more consistent JSON output
            max_tokens=256,
        )
        raw = response.choices[0].message.content or ""

        if self._debug:
            logger.debug("LLM RESPONSE: %s", raw)

        return raw

    def _parse_response(self, raw: str) -> ExtractionResult:
        """Parse the LLM's JSON response into an ExtractionResult."""
        raw = raw.strip()

        # Strip markdown code fences if the model added them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            data = json.loads(raw)
            return ExtractionResult(
                value=data.get("value"),
                confidence=float(data.get("confidence", 0.0)),
                clarifying_question=data.get("clarifying_question"),
                raw_llm_response=raw,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse LLM extraction response: %s — raw: %s", exc, raw)
            # Graceful fallback: low confidence, no extracted value
            return ExtractionResult(
                value=None,
                confidence=0.0,
                clarifying_question="Could you say that again? I didn't quite catch it.",
                raw_llm_response=raw,
            )
