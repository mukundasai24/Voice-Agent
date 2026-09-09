"""
tests/test_conversation.py
───────────────────────────
Integration tests for ConversationManager.

All LLM calls are mocked — these tests verify the conversation FLOW logic
(sequencing, clarification, completion detection, graceful degradation)
without any network access.

Run:  pytest tests/test_conversation.py -v
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation import ConversationManager, AgentTurn
from agent.extractor import ExtractionResult
from agent.questions import required_fields


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_manager() -> ConversationManager:
    """Create a ConversationManager with a dummy Groq key."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test_dummy"}):
        return ConversationManager()


def mock_extract(manager: ConversationManager, result: ExtractionResult) -> None:
    """Patch the extractor so it always returns `result`."""
    manager._extractor.extract = MagicMock(return_value=result)


def high_confidence(value: str) -> ExtractionResult:
    return ExtractionResult(value=value, confidence=0.99, clarifying_question=None)


def low_confidence(followup: str = "Could you clarify?") -> ExtractionResult:
    return ExtractionResult(value=None, confidence=0.2, clarifying_question=followup)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: start()
# ─────────────────────────────────────────────────────────────────────────────

def test_start_returns_agent_turn():
    manager = make_manager()
    turn = manager.start()
    assert isinstance(turn, AgentTurn)
    assert not turn.is_final


def test_start_text_contains_greeting():
    manager = make_manager()
    turn = manager.start()
    assert "Hello" in turn.text or "hello" in turn.text.lower()


def test_start_initialises_state():
    manager = make_manager()
    manager.start()
    state = manager.get_state()
    assert state is not None
    assert not state.complete


# ─────────────────────────────────────────────────────────────────────────────
# Tests: happy path — all answers accepted first time
# ─────────────────────────────────────────────────────────────────────────────

def test_happy_path_completes_after_all_required_fields():
    manager = make_manager()
    manager.start()

    required = required_fields()
    for i, field in enumerate(required):
        mock_extract(manager, high_confidence(f"value_{i}"))
        response = manager.process_reply(f"my answer for {field}")
        if i < len(required) - 1:
            assert not response.is_final
        else:
            # Last required field — should now be complete
            assert response.is_final or manager.get_state().complete


def test_happy_path_saves_all_field_values():
    manager = make_manager()
    manager.start()
    required = required_fields()

    for i, field in enumerate(required):
        mock_extract(manager, high_confidence(f"answer_{field}"))
        manager.process_reply("some reply")

    result = manager.get_result()
    for field in required:
        assert field in result["collected"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests: clarification flow
# ─────────────────────────────────────────────────────────────────────────────

def test_low_confidence_triggers_clarifying_question():
    manager = make_manager()
    manager.start()

    # First reply: low confidence
    mock_extract(manager, low_confidence("What day exactly did you mean?"))
    response = manager.process_reply("sometime next week")

    assert "What day exactly" in response.text
    assert not response.is_final


def test_after_clarification_accepted_answer_advances():
    manager = make_manager()
    manager.start()

    # First attempt: low confidence
    mock_extract(manager, low_confidence("What day exactly?"))
    manager.process_reply("sometime next week")

    # Second attempt: high confidence
    mock_extract(manager, high_confidence("2026-10-15"))
    response = manager.process_reply("Tuesday the 15th of October")

    # Should have accepted and moved to next field (or finished)
    state = manager.get_state()
    first_field = required_fields()[0]
    assert first_field in state.collected


def test_max_clarify_attempts_moves_on():
    """After MAX_CLARIFY_ATTEMPTS failures, the agent should give up and continue."""
    from agent.extractor import MAX_CLARIFY_ATTEMPTS

    manager = make_manager()
    manager.start()

    # Fail MAX_CLARIFY_ATTEMPTS + 1 times
    for _ in range(MAX_CLARIFY_ATTEMPTS + 1):
        mock_extract(manager, low_confidence("Please clarify"))
        response = manager.process_reply("I have no idea")

    # Agent should have moved on (not stuck in an infinite loop)
    state = manager.get_state()
    first_field = required_fields()[0]
    # The field is either marked as NEEDS_FOLLOWUP or the conversation has moved forward
    advanced = (
        first_field in state.collected
        or state.current_field != first_field
    )
    assert advanced, "Agent should move past a stuck field after max clarify attempts"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: result dict
# ─────────────────────────────────────────────────────────────────────────────

def test_get_result_has_call_id():
    manager = make_manager()
    manager.start()
    result = manager.get_result()
    assert "call_id" in result
    assert len(result["call_id"]) > 0


def test_get_result_before_completion_has_missing_fields():
    manager = make_manager()
    manager.start()
    # Answer only the first field
    mock_extract(manager, high_confidence("Priya Sharma"))
    manager.process_reply("Priya Sharma")

    result = manager.get_result()
    assert len(result["missing_at_end"]) > 0
