"""
tests/test_state.py
────────────────────
Unit tests for ConversationState — the pure-data state machine.

These tests do NOT call any external API (no Groq, no Twilio).
They run instantly and should always pass as long as the state logic is correct.

Run:  pytest tests/test_state.py -v
"""

from __future__ import annotations

import pytest

from agent.state import ConversationState, TurnRecord
from agent.questions import required_fields, optional_fields, QUESTIONS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_state(call_id: str = "test-001") -> ConversationState:
    return ConversationState(call_id=call_id)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: initial state
# ─────────────────────────────────────────────────────────────────────────────

def test_initial_state_is_incomplete():
    state = make_state()
    assert not state.complete


def test_initial_missing_equals_all_required():
    state = make_state()
    assert set(state.missing) == set(required_fields())


def test_initial_collected_is_empty():
    state = make_state()
    assert state.collected == {}


def test_initial_current_field_is_first_required():
    state = make_state()
    first_required = next(q["field"] for q in QUESTIONS if q["required"])
    assert state.current_field == first_required


def test_initial_turn_number_is_zero():
    state = make_state()
    assert state.turn_number == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: recording answers
# ─────────────────────────────────────────────────────────────────────────────

def test_recording_answer_removes_from_missing():
    state = make_state()
    field = required_fields()[0]
    state.record_answer(field, "some value")
    assert field not in state.missing


def test_recording_answer_adds_to_collected():
    state = make_state()
    field = required_fields()[0]
    state.record_answer(field, "Priya Sharma")
    assert state.collected[field] == "Priya Sharma"


def test_recording_all_required_sets_complete():
    state = make_state()
    for field in required_fields():
        assert not state.complete
        state.record_answer(field, f"value_for_{field}")
    assert state.complete


def test_complete_state_has_no_current_field():
    state = make_state()
    for field in required_fields():
        state.record_answer(field, f"value_for_{field}")
    assert state.current_field is None


# ─────────────────────────────────────────────────────────────────────────────
# Tests: clarification counter
# ─────────────────────────────────────────────────────────────────────────────

def test_clarify_counter_increments():
    state = make_state()
    assert state.increment_clarify() == 1
    assert state.increment_clarify() == 2
    assert state.increment_clarify() == 3


def test_clarify_counter_resets_after_successful_answer():
    state = make_state()
    state.increment_clarify()
    state.increment_clarify()
    field = required_fields()[0]
    state.record_answer(field, "Priya Sharma")
    # Counter should be reset — next increment starts from 1
    assert state.increment_clarify() == 1


# ─────────────────────────────────────────────────────────────────────────────
# Tests: mark_asked
# ─────────────────────────────────────────────────────────────────────────────

def test_mark_asked_is_idempotent():
    state = make_state()
    field = required_fields()[0]
    state.mark_asked(field)
    state.mark_asked(field)
    assert state.asked.count(field) == 1


def test_mark_asked_accumulates_across_fields():
    state = make_state()
    fields = required_fields()[:2]
    for f in fields:
        state.mark_asked(f)
    assert all(f in state.asked for f in fields)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: turn records
# ─────────────────────────────────────────────────────────────────────────────

def test_add_turn_increments_turn_number():
    state = make_state()
    turn = TurnRecord(
        turn_number=0,
        field_asked="caller_name",
        agent_said="What is your name?",
        user_said="Priya",
        extracted_value="Priya",
        confidence=0.99,
        clarifying_question=None,
    )
    state.add_turn(turn)
    assert state.turn_number == 1


# ─────────────────────────────────────────────────────────────────────────────
# Tests: serialization
# ─────────────────────────────────────────────────────────────────────────────

def test_to_result_dict_contains_expected_keys():
    state = make_state()
    result = state.to_result_dict()
    assert "call_id" in result
    assert "collected" in result
    assert "missing_at_end" in result
    assert "turn_count" in result


def test_to_result_dict_reflects_collected():
    state = make_state()
    state.record_answer("caller_name", "Priya Sharma")
    result = state.to_result_dict()
    assert result["collected"]["caller_name"] == "Priya Sharma"
