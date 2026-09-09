"""
agent/conversation.py
──────────────────────
ConversationManager — the brain of the agent.

Drives the question-answer loop:
  1. Decides what to say next (next question, clarifying question, or goodbye)
  2. Takes the caller's reply, calls the Extractor
  3. If confidence is high → accept the answer, move to the next field
  4. If confidence is low → ask the clarifying question (up to MAX_CLARIFY_ATTEMPTS times)
  5. If stuck after max attempts → flag the field, move on
  6. When all required fields are filled → build a confirmation summary + goodbye

The ConversationManager does NOT know about audio, telephony, or databases.
It only knows about text in and text out — that's the clean separation that
lets Phase 1 (console), Phase 2 (mic/speaker), and Phase 3 (phone) all reuse
exactly the same object.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import NamedTuple

from agent.extractor import (
    CONFIDENCE_THRESHOLD,
    MAX_CLARIFY_ATTEMPTS,
    Extractor,
    ExtractionResult,
)
from agent.questions import QUESTIONS, get_question
from agent.state import ConversationState, TurnRecord

logger = logging.getLogger(__name__)


class AgentTurn(NamedTuple):
    """What the agent says in response to a caller turn."""
    text: str           # the text to speak / print
    is_final: bool      # True = conversation is over, hang up / end session


class ConversationManager:
    """
    Orchestrates the full Q&A conversation for one call.

    Usage (Phase 1 console demo — see demo/console_demo.py):
        manager = ConversationManager()
        greeting = manager.start()          # "Hello! I'd like to book…"
        print(greeting.text)
        while True:
            user_input = input("You: ")
            response = manager.process_reply(user_input)
            print("Agent:", response.text)
            if response.is_final:
                break
        result = manager.get_result()       # the final collected dict → goes to storage
    """

    # ── Conversation-level text (edit to change the agent's personality/wording) ──

    GREETING = (
        "Hello! I'm an automated booking assistant. "
        "I'll ask you a few quick questions to schedule your appointment. "
        "This should only take about a minute. "
        "This call may be recorded for quality purposes. "
        "Let's get started!"
    )

    CONFIRMATION_INTRO = "Great, I think I have everything. Let me read back what I've collected:"

    GOODBYE = (
        "Thank you! Your booking details have been saved. "
        "Someone will be in touch shortly to confirm. "
        "Have a wonderful day — goodbye!"
    )

    STUCK_FIELD_MESSAGE = (
        "No problem — I'll flag that one for a team member to follow up on. "
        "Let's move on."
    )

    def __init__(self) -> None:
        self._extractor = Extractor()
        self._state: ConversationState | None = None
        # Parallel list of {"role": ..., "content": ...} for LLM context
        self._history: list[dict[str, str]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, call_id: str | None = None) -> AgentTurn:
        """
        Start a new conversation. Call this once at the beginning of a call.
        Returns the greeting text + first question.
        """
        self._state = ConversationState(
            call_id=call_id or str(uuid.uuid4()),
            started_at=datetime.now(timezone.utc),
        )
        self._history = []

        first_question = self._next_question_text()
        greeting_text = f"{self.GREETING}\n\n{first_question}"

        self._append_history("assistant", greeting_text)
        logger.info("[%s] Conversation started.", self._state.call_id)
        return AgentTurn(text=greeting_text, is_final=False)

    def process_reply(self, caller_text: str) -> AgentTurn:
        """
        Process one caller reply.  Returns what the agent should say next.

        Args:
            caller_text: The raw spoken/typed reply from the caller.

        Returns:
            AgentTurn — the agent's response and whether the call is done.
        """
        assert self._state is not None, "Call start() before process_reply()"

        self._append_history("user", caller_text)
        current_field = self._state.current_field

        if current_field is None:
            # All done — shouldn't normally reach here, but be safe
            return self._build_final_turn()

        logger.info(
            "[%s] Turn %d — field=%s, caller said: %r",
            self._state.call_id, self._state.turn_number, current_field, caller_text,
        )

        # ── Ask the LLM to extract the answer ───────────────────────────────
        result: ExtractionResult = self._extractor.extract(
            field_name=current_field,
            caller_reply=caller_text,
            conversation_so_far=self._history[:-1],  # exclude the just-added user msg
        )

        logger.info(
            "[%s] Extraction — field=%s value=%r confidence=%.2f",
            self._state.call_id, current_field, result.value, result.confidence,
        )

        # ── Record the turn ──────────────────────────────────────────────────
        # We'll fill agent_said below; placeholder for now
        turn = TurnRecord(
            turn_number=self._state.turn_number,
            field_asked=current_field,
            agent_said="",       # filled below
            user_said=caller_text,
            extracted_value=result.value,
            confidence=result.confidence,
            clarifying_question=result.clarifying_question,
        )

        # ── Decide: accept, clarify, or give up? ────────────────────────────
        if result.confidence >= CONFIDENCE_THRESHOLD and result.value is not None:
            # ✅ Good answer — accept it, advance
            self._state.record_answer(current_field, result.value)
            agent_text = self._advance_or_finish()

        else:
            clarify_count = self._state.increment_clarify()

            if clarify_count <= MAX_CLARIFY_ATTEMPTS and result.clarifying_question:
                # 🔁 Low confidence — ask the targeted follow-up
                agent_text = result.clarifying_question
                logger.info(
                    "[%s] Clarifying attempt %d/%d for field=%s",
                    self._state.call_id, clarify_count, MAX_CLARIFY_ATTEMPTS, current_field,
                )
            else:
                # ❌ Stuck — flag it, move on
                logger.warning(
                    "[%s] Giving up on field=%s after %d attempts.",
                    self._state.call_id, current_field, clarify_count,
                )
                # Store None for optional fields; flag required ones
                question_def = get_question(current_field)
                if question_def and not question_def["required"]:
                    self._state.record_answer(current_field, None)
                else:
                    # For required fields: mark as attempted but unfilled
                    # (missing_at_end in the result will capture this)
                    self._state.asked.append(current_field)
                    self._state.collected[current_field] = "__NEEDS_FOLLOWUP__"

                agent_text = f"{self.STUCK_FIELD_MESSAGE} {self._advance_or_finish()}"

        turn.agent_said = agent_text
        self._state.add_turn(turn)
        self._append_history("assistant", agent_text)

        is_final = self._state.complete and "goodbye" in agent_text.lower()
        if is_final:
            self._state.ended_at = datetime.now(timezone.utc)

        return AgentTurn(text=agent_text, is_final=is_final)

    def get_result(self) -> dict:
        """Return the final extracted data as a plain dict (for storage)."""
        assert self._state is not None, "Call start() before get_result()"
        if self._state.ended_at is None:
            self._state.ended_at = datetime.now(timezone.utc)
        return self._state.to_result_dict()

    def get_state(self) -> ConversationState:
        """Return the raw ConversationState (for logging / debugging)."""
        assert self._state is not None
        return self._state

    # ── Private helpers ───────────────────────────────────────────────────────

    def _next_question_text(self) -> str:
        """Get the prompt text for the current unanswered field."""
        field = self._state.current_field  # type: ignore[union-attr]
        if field is None:
            return ""
        q = get_question(field)
        assert q is not None
        self._state.mark_asked(field)  # type: ignore[union-attr]
        return q["prompt"]

    def _advance_or_finish(self) -> str:
        """
        After accepting an answer: if there are more questions, ask the next one.
        If everything is collected, build the confirmation + goodbye.
        """
        if self._state.complete:  # type: ignore[union-attr]
            return self._build_confirmation_goodbye()
        else:
            return self._next_question_text()

    def _build_confirmation_goodbye(self) -> str:
        """Build the 'let me read back what I have' summary + goodbye."""
        collected = self._state.collected  # type: ignore[union-attr]

        lines = [self.CONFIRMATION_INTRO]
        for q in QUESTIONS:
            value = collected.get(q["field"])
            if value and value != "__NEEDS_FOLLOWUP__":
                label = q["field"].replace("_", " ").title()
                lines.append(f"  • {label}: {value}")
            elif value == "__NEEDS_FOLLOWUP__":
                label = q["field"].replace("_", " ").title()
                lines.append(f"  • {label}: to be confirmed by our team")

        lines.append("")
        lines.append(self.GOODBYE)
        return "\n".join(lines)

    def _build_final_turn(self) -> AgentTurn:
        """Emergency exit if we somehow arrive at process_reply with no field left."""
        goodbye = self._build_confirmation_goodbye()
        self._append_history("assistant", goodbye)
        self._state.ended_at = datetime.now(timezone.utc)  # type: ignore[union-attr]
        return AgentTurn(text=goodbye, is_final=True)

    def _append_history(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})
