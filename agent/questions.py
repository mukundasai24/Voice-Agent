"""
agent/questions.py
──────────────────
THIS IS THE FILE TO EDIT when you want to change what the agent asks.

It contains two things:
  1. QUESTIONS  — the ordered list of things the agent will ask
  2. CALL_SCHEMA — the JSON structure that will be saved to the database

Each question is a dict with these keys:
  field        (str)  — the key that will appear in the saved JSON
  required     (bool) — if True, the agent won't hang up until this is filled
  prompt       (str)  — what the agent says to ask the question
  clarify_hint (str)  — a hint to the LLM about what a valid answer looks like
                        (used to generate a good follow-up question when the
                        caller's answer is vague or incomplete)
  example      (str)  — an example of a valid answer (shown in clarify_hint
                        context to the LLM — never read aloud to the caller)

Default use-case: appointment booking.
Swap this list out entirely for a different use-case — nothing else needs to change.
"""

from __future__ import annotations

from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# QUESTION SCRIPT
# Change these to match whatever your agent is collecting.
# ─────────────────────────────────────────────────────────────────────────────

QUESTIONS: list[dict[str, Any]] = [
    {
        "field": "caller_name",
        "required": True,
        "prompt": "Could I start by getting your full name, please?",
        "clarify_hint": "The caller's first and last name.",
        "example": "Priya Sharma",
    },
    {
        "field": "preferred_date",
        "required": True,
        "prompt": "What date works best for your appointment?",
        "clarify_hint": (
            "A calendar date — could be a day name (Monday), a relative phrase "
            "(next Tuesday, tomorrow), or a specific date (15th October). "
            "Normalize to ISO format YYYY-MM-DD where possible."
        ),
        "example": "2026-10-15",
    },
    {
        "field": "preferred_time",
        "required": True,
        "prompt": "And what time would you prefer — morning, afternoon, or a specific time?",
        "clarify_hint": (
            "A time of day or rough period. Normalize to HH:MM 24h where possible, "
            "otherwise keep the descriptive phrase (e.g. 'morning', 'after 3 PM')."
        ),
        "example": "10:30",
    },
    {
        "field": "service_type",
        "required": True,
        "prompt": "What service are you looking to book? For example: a consultation, a follow-up, or a full assessment.",
        "clarify_hint": (
            "The type of appointment or service requested. Accept any reasonable "
            "description; don't force it into a fixed list."
        ),
        "example": "initial consultation",
    },
    {
        "field": "callback_number",
        "required": False,
        "prompt": (
            "Finally, is there a phone number we should use to confirm the booking? "
            "Or shall we use the number you're calling from?"
        ),
        "clarify_hint": (
            "A phone number for confirmation. Accept 'this number', 'the same one', "
            "or a spoken number string. Normalize to E.164 format if an Indian or "
            "international number is given."
        ),
        "example": "+919876543210",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# JSON SCHEMA (what the saved record looks like)
# This is mostly documentation — the extractor fills these fields by name.
# ─────────────────────────────────────────────────────────────────────────────

CALL_SCHEMA: dict[str, Any] = {
    "title": "AppointmentBooking",
    "description": "Structured data extracted from a single appointment-booking call.",
    "fields": {
        "caller_name":      {"type": "string",  "required": True},
        "preferred_date":   {"type": "string",  "required": True,  "format": "YYYY-MM-DD or natural language"},
        "preferred_time":   {"type": "string",  "required": True,  "format": "HH:MM or natural language"},
        "service_type":     {"type": "string",  "required": True},
        "callback_number":  {"type": "string",  "required": False, "format": "E.164 or 'use calling number'"},
    },
}


def required_fields() -> list[str]:
    """Return the list of field names that must be filled before the call ends."""
    return [q["field"] for q in QUESTIONS if q["required"]]


def optional_fields() -> list[str]:
    """Return the list of field names that are optional."""
    return [q["field"] for q in QUESTIONS if not q["required"]]


def get_question(field: str) -> dict[str, Any] | None:
    """Look up a question definition by its field name."""
    return next((q for q in QUESTIONS if q["field"] == field), None)
