"""
tests/test_extractor.py
────────────────────────
Unit tests for the Extractor — covers parsing and edge-case handling.

Most tests mock the Groq API call so they run instantly without network access.
A small number of "live" tests (marked with @pytest.mark.live) hit the real API
and are skipped automatically if GROQ_API_KEY is not set.

Run all (fast, no API):   pytest tests/test_extractor.py -v -m "not live"
Run live tests too:       pytest tests/test_extractor.py -v  (requires GROQ_API_KEY)
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from agent.extractor import Extractor, ExtractionResult, CONFIDENCE_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_extractor() -> Extractor:
    """Create an Extractor with a dummy API key (for mocked tests)."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test_dummy_key_for_unit_tests"}):
        return Extractor()


def mock_llm_response(extractor: Extractor, json_payload: dict) -> None:
    """Patch _call_llm to return a specific JSON string."""
    extractor._call_llm = MagicMock(return_value=json.dumps(json_payload))


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _parse_response
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_clean_high_confidence_response():
    extractor = make_extractor()
    result = extractor._parse_response(
        '{"value": "Priya Sharma", "confidence": 0.98, "clarifying_question": null}'
    )
    assert result.value == "Priya Sharma"
    assert result.confidence == pytest.approx(0.98)
    assert result.clarifying_question is None


def test_parse_low_confidence_with_followup():
    extractor = make_extractor()
    result = extractor._parse_response(
        '{"value": null, "confidence": 0.3, "clarifying_question": "Did you mean this Monday?"}'
    )
    assert result.value is None
    assert result.confidence < CONFIDENCE_THRESHOLD
    assert result.clarifying_question is not None
    assert "Monday" in result.clarifying_question


def test_parse_strips_markdown_code_fences():
    extractor = make_extractor()
    raw = '```json\n{"value": "10:30", "confidence": 0.95, "clarifying_question": null}\n```'
    result = extractor._parse_response(raw)
    assert result.value == "10:30"
    assert result.confidence == pytest.approx(0.95)


def test_parse_broken_json_returns_safe_fallback():
    extractor = make_extractor()
    result = extractor._parse_response("this is not valid json at all")
    assert result.value is None
    assert result.confidence == 0.0
    assert result.clarifying_question is not None  # safe fallback message


def test_parse_missing_keys_returns_safe_fallback():
    extractor = make_extractor()
    result = extractor._parse_response('{"something_else": "unexpected"}')
    # Should not crash; confidence defaults to 0.0
    assert result.confidence == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: extract() with mocked LLM
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_returns_extraction_result_type():
    extractor = make_extractor()
    mock_llm_response(extractor, {
        "value": "Priya Sharma",
        "confidence": 0.99,
        "clarifying_question": None,
    })
    result = extractor.extract(
        field_name="caller_name",
        caller_reply="My name is Priya Sharma",
        conversation_so_far=[],
    )
    assert isinstance(result, ExtractionResult)
    assert result.value == "Priya Sharma"


def test_extract_high_confidence_has_no_clarifying_question():
    extractor = make_extractor()
    mock_llm_response(extractor, {
        "value": "initial consultation",
        "confidence": 0.97,
        "clarifying_question": None,
    })
    result = extractor.extract("service_type", "I want an initial consultation", [])
    assert result.clarifying_question is None


def test_extract_low_confidence_has_clarifying_question():
    extractor = make_extractor()
    mock_llm_response(extractor, {
        "value": None,
        "confidence": 0.2,
        "clarifying_question": "Could you be more specific about the date?",
    })
    result = extractor.extract("preferred_date", "sometime soon", [])
    assert result.confidence < CONFIDENCE_THRESHOLD
    assert result.clarifying_question is not None


def test_extract_with_conversation_context():
    """History is passed to the LLM — verify no crash and correct return type."""
    extractor = make_extractor()
    mock_llm_response(extractor, {
        "value": "use calling number",
        "confidence": 0.9,
        "clarifying_question": None,
    })
    history = [
        {"role": "assistant", "content": "What number should we use?"},
        {"role": "user", "content": "Same as this one"},
    ]
    result = extractor.extract("callback_number", "Same as this one", history)
    assert result.value == "use calling number"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: unknown field raises ValueError
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_unknown_field_raises():
    extractor = make_extractor()
    with pytest.raises(ValueError, match="No question definition found"):
        extractor.extract("nonexistent_field", "some reply", [])


# ─────────────────────────────────────────────────────────────────────────────
# Live tests (require GROQ_API_KEY — skipped in CI if not set)
# ─────────────────────────────────────────────────────────────────────────────

live = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live API test",
)


@live
def test_live_extract_clear_name():
    """Real API call — verifies the LLM returns valid JSON for a simple name."""
    extractor = Extractor()
    result = extractor.extract(
        field_name="caller_name",
        caller_reply="Hi, my name is Rahul Verma",
        conversation_so_far=[],
    )
    assert result.value is not None
    assert "Rahul" in str(result.value) or "Verma" in str(result.value)
    assert result.confidence >= CONFIDENCE_THRESHOLD


@live
def test_live_extract_ambiguous_date_generates_followup():
    """Real API call — a vague date should produce low confidence + follow-up."""
    extractor = Extractor()
    result = extractor.extract(
        field_name="preferred_date",
        caller_reply="sometime soon, not sure exactly",
        conversation_so_far=[],
    )
    # Either low confidence OR a clarifying question should be present
    is_ambiguous = result.confidence < CONFIDENCE_THRESHOLD or result.clarifying_question is not None
    assert is_ambiguous, (
        f"Expected ambiguous result for vague date, got confidence={result.confidence}"
    )
