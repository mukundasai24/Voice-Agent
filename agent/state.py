"""
agent/state.py
──────────────
ConversationState — a plain dataclass that tracks exactly where we are in the
conversation at any point in time.

Think of it like a checklist:
  - asked       = questions we've already spoken aloud
  - collected   = answers we've confirmed (field → extracted value)
  - missing     = required fields we still need
  - complete    = True once every required field is filled
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent.questions import QUESTIONS, required_fields


@dataclass
class TurnRecord:
    """One exchange in the conversation: what was asked, what was heard, what was extracted."""
    turn_number: int
    field_asked: str          # which question we were on
    agent_said: str           # the text the agent spoke
    user_said: str            # the raw transcript of what the caller said
    extracted_value: Any      # the value the LLM pulled out (None if extraction failed)
    confidence: float         # 0.0 – 1.0: how sure the LLM is about the extraction
    clarifying_question: str | None  # if confidence was low, the follow-up question generated


@dataclass
class ConversationState:
    """
    The full state of one call.

    This is the only object passed around between components —
    telephony, STT, LLM, TTS, and storage all read from or write to this.
    """

    call_id: str                          # unique ID for this call (UUID)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None

    # Which fields we've asked about (in order)
    asked: list[str] = field(default_factory=list)

    # Field name → extracted value for confirmed answers
    collected: dict[str, Any] = field(default_factory=dict)

    # History of every exchange (useful for logging + debugging)
    turns: list[TurnRecord] = field(default_factory=list)

    # Internal: how many clarifying attempts on the current field
    _clarify_count: int = field(default=0, repr=False)

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def missing(self) -> list[str]:
        """Required fields we haven't successfully collected yet."""
        return [f for f in required_fields() if f not in self.collected]

    @property
    def complete(self) -> bool:
        """True when every required field has a confirmed value."""
        return len(self.missing) == 0

    @property
    def current_field(self) -> str | None:
        """
        The field we're currently working on.
        Returns None if the conversation is complete.
        """
        if self.complete:
            return None
        # Walk the question list in order; return the first unanswered field
        for q in QUESTIONS:
            if q["field"] not in self.collected:
                return q["field"]
        return None

    @property
    def turn_number(self) -> int:
        return len(self.turns)

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def mark_asked(self, field_name: str) -> None:
        """Record that we've asked about this field."""
        if field_name not in self.asked:
            self.asked.append(field_name)

    def record_answer(self, field_name: str, value: Any) -> None:
        """Store a confirmed extracted value."""
        self.collected[field_name] = value
        self._clarify_count = 0  # reset clarification counter for the next field

    def increment_clarify(self) -> int:
        """Increment and return the number of clarification attempts for the current field."""
        self._clarify_count += 1
        return self._clarify_count

    def add_turn(self, turn: TurnRecord) -> None:
        self.turns.append(turn)

    def to_result_dict(self) -> dict[str, Any]:
        """
        Serialize the final collected answers as a plain dict (this is what
        gets saved to the database).
        """
        return {
            "call_id": self.call_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "collected": self.collected,
            "missing_at_end": self.missing,   # fields we gave up on (shouldn't happen ideally)
            "turn_count": self.turn_number,
        }
