"""Tests for src/glee/claude_glee_agent.py's schema-echoing bug fix.

Root cause (confirmed via cross-referencing a real run's trajectories.jsonl
against the GLEE server's move() rejection reasons): Claude sometimes
returns {"type": ..., "fields": {...}} -- echoing the *shape* of
valid_actions -- instead of a flat instance of just the field values. That
run recorded 40 server-side rejections across 12 duplicate-round groups and
at least 1 confirmed game loss (game 8b44bf34, 5/5 submissions rejected).

These tests use the ACTUAL malformed JSON captured from that run (not
synthetic examples) to confirm parse_response() now unwraps it, and confirm
already-flat actions pass through unchanged.

Sets a dummy ANTHROPIC_API_KEY before importing the module under test,
since Anthropic() is constructed at module import time and no real API
calls are made by anything exercised here.
"""

from __future__ import annotations

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-dummy-key-not-used")

import json  # noqa: E402
import logging  # noqa: E402

import pytest  # noqa: E402

from src.glee.claude_glee_agent import (  # noqa: E402
    _build_example_action,
    _example_value_for_field,
    parse_response,
)

# ---------------------------------------------------------------------------
# TestExampleValueForField
# ---------------------------------------------------------------------------


class TestExampleValueForField:
    def test_number_type_returns_zero(self) -> None:
        assert _example_value_for_field("number (your proposed amount for Alice)") == 0

    def test_string_type_returns_example(self) -> None:
        assert _example_value_for_field("string (optional message)") == "example"

    def test_enum_description_returns_first_quoted_option(self) -> None:
        desc = "'AcceptOffer', 'RejectOffer', or 'WalkAway'"
        assert _example_value_for_field(desc) == "AcceptOffer"

    def test_yes_no_enum_returns_first_option(self) -> None:
        desc = "'yes' (recommend) or 'no' (don't recommend)"
        assert _example_value_for_field(desc) == "yes"

    def test_unrecognised_description_falls_back_to_example(self) -> None:
        assert _example_value_for_field("some unusual free-form description") == "example"

    def test_case_insensitive_number(self) -> None:
        assert _example_value_for_field("NUMBER (a value)") == 0


# ---------------------------------------------------------------------------
# TestBuildExampleAction
# ---------------------------------------------------------------------------


class TestBuildExampleAction:
    def test_builds_flat_dict_from_fields_schema(self) -> None:
        valid_actions = {
            "type": "offer",
            "fields": {
                "alice_gain": "number (your proposed amount for Alice)",
                "bob_gain": "number (your proposed amount for Bob)",
                "message": "string (optional message to opponent)",
            },
        }
        example = _build_example_action(valid_actions)
        assert example == {"alice_gain": 0, "bob_gain": 0, "message": "example"}

    def test_no_wrapper_keys_in_output(self) -> None:
        valid_actions = {"type": "offer", "fields": {"product_price": "number (a price)"}}
        example = _build_example_action(valid_actions)
        assert "type" not in example
        assert "fields" not in example

    def test_missing_fields_key_returns_empty_dict(self) -> None:
        assert _build_example_action({"type": "decision"}) == {}

    def test_non_dict_fields_returns_empty_dict(self) -> None:
        assert _build_example_action({"type": "decision", "fields": "not a dict"}) == {}

    def test_decision_schema_example(self) -> None:
        valid_actions = {
            "type": "decision",
            "fields": {
                "decision": "'AcceptOffer', 'RejectOffer', or 'WalkAway'",
                "product_price": "number (required if RejectOffer - your counteroffer)",
                "message": "string (optional)",
            },
        }
        example = _build_example_action(valid_actions)
        assert example == {
            "decision": "AcceptOffer",
            "product_price": 0,
            "message": "example",
        }


# ---------------------------------------------------------------------------
# TestParseResponseUnwrapsRealCapturedMalformedOutputs
# ---------------------------------------------------------------------------


def _reasoning_action_text(reasoning: str, action: dict) -> str:
    return f"REASONING: {reasoning}\nACTION: {json.dumps(action)}"


class TestParseResponseUnwrapsRealCapturedMalformedOutputs:
    """Each case here is the literal action dict captured in
    tests/experiments/glee/trajectories.jsonl from the run that produced
    40 rejections, reconstructed into the REASONING:/ACTION: text shape
    parse_response actually receives."""

    def test_unwraps_bargaining_offer_from_game_74607f90(self) -> None:
        # Real capture: game 74607f90, rejected -- "Offer must include
        # 'alice_gain' and 'bob_gain'" -- because they were nested under "fields".
        malformed = {
            "type": "offer",
            "fields": {"alice_gain": 550000, "bob_gain": 450000},
        }
        text = _reasoning_action_text("Opening with a favorable split.", malformed)
        _, action = parse_response(text)
        assert action == {"alice_gain": 550000, "bob_gain": 450000}
        assert "fields" not in action
        assert "type" not in action

    def test_unwraps_seller_message_from_game_8b44bf34(self) -> None:
        # Real capture: game 8b44bf34, rejected 5/5 times -- "Must include a
        # non-empty 'message'" -- game was force-closed as a result.
        malformed = {
            "type": "seller_message",
            "fields": {
                "message": (
                    "I want to be straightforward with you: this product is "
                    "genuinely high quality."
                )
            },
        }
        text = _reasoning_action_text("Being honest to build reputation.", malformed)
        _, action = parse_response(text)
        assert action == {
            "message": (
                "I want to be straightforward with you: this product is "
                "genuinely high quality."
            )
        }

    def test_unwraps_negotiation_decision_from_game_ec55fb25(self) -> None:
        # Real capture: game ec55fb25, rejected -- "Decision must be one of:
        # AcceptOffer, RejectOffer, WalkAway".
        malformed = {
            "type": "decision",
            "fields": {"decision": "RejectOffer", "product_price": 9400.0},
        }
        text = _reasoning_action_text("Countering below their ask.", malformed)
        _, action = parse_response(text)
        assert action == {"decision": "RejectOffer", "product_price": 9400.0}

    def test_unwraps_negotiation_offer_from_game_c8712579(self) -> None:
        # Real capture: game c8712579, rejected -- "Offer must include
        # 'product_price'".
        malformed = {"type": "offer", "fields": {"product_price": 15000}}
        text = _reasoning_action_text("Anchoring the opening price.", malformed)
        _, action = parse_response(text)
        assert action == {"product_price": 15000}

    def test_unwrap_logs_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        malformed = {"type": "offer", "fields": {"alice_gain": 500, "bob_gain": 500}}
        text = _reasoning_action_text("x", malformed)
        with caplog.at_level(logging.WARNING, logger="claude_glee_agent"):
            parse_response(text)
        assert any("unwrapping schema-echoed action" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# TestParseResponseFlatActionsPassThroughUnchanged
# ---------------------------------------------------------------------------


class TestParseResponseFlatActionsPassThroughUnchanged:
    """Real, correctly-formed (accepted) actions captured in the same run --
    must not be altered, double-unwrapped, or otherwise corrupted."""

    def test_flat_bargaining_offer_unchanged(self) -> None:
        flat = {"alice_gain": 500000.0, "bob_gain": 500000.0}
        text = _reasoning_action_text("Fair 50/50 split.", flat)
        _, action = parse_response(text)
        assert action == flat

    def test_flat_decision_unchanged(self) -> None:
        flat = {"decision": "accept"}
        text = _reasoning_action_text("Good enough to take.", flat)
        _, action = parse_response(text)
        assert action == flat

    def test_flat_negotiation_offer_unchanged(self) -> None:
        # Real capture: accepted with valid=True in the same run.
        flat = {"product_price": 9800}
        text = _reasoning_action_text("Reasonable counter.", flat)
        _, action = parse_response(text)
        assert action == flat

    def test_flat_persuasion_decision_unchanged(self) -> None:
        flat = {"decision": "yes"}
        text = _reasoning_action_text("Signal has been reliable.", flat)
        _, action = parse_response(text)
        assert action == flat

    def test_flat_action_does_not_trigger_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        flat = {"alice_gain": 500, "bob_gain": 500}
        text = _reasoning_action_text("x", flat)
        with caplog.at_level(logging.WARNING, logger="claude_glee_agent"):
            parse_response(text)
        assert not any("unwrapping schema-echoed action" in r.message for r in caplog.records)

    def test_fields_key_present_but_not_a_dict_is_not_unwrapped(self) -> None:
        # Edge case: "fields" as a non-dict value is not the schema-echo
        # pattern -- must be left alone, not misinterpreted.
        odd = {"decision": "accept", "fields": "not a nested schema"}
        text = _reasoning_action_text("x", odd)
        _, action = parse_response(text)
        assert action == odd
