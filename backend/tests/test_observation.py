"""Parsing the vision model's text (format defined by COOKING_PROMPT in LLM/inference.py)."""

import pytest

from app.llm.base import parse_observation
from app.llm.mock_provider import format_response
from app.schemas import CookingStatus as S

CANONICAL = """Food Identified: Fried egg

Cooking Status: APPEARS_COOKED

Visual Evidence: The white is opaque and the edges are lightly browned.

Recommendation: It looks done. Remove it from the heat."""


def test_parse_canonical():
    p = parse_observation(CANONICAL)
    assert p.food == "Fried egg" and p.status == S.APPEARS_COOKED and p.parse_ok
    assert p.evidence.startswith("The white is opaque")
    assert p.recommendation.startswith("It looks done")


def test_parse_markdown_and_brackets():
    raw = "**Food Identified:** [Pancake]\n**Cooking Status:** [APPEARS_UNDERCOOKED]\n**Visual Evidence:** Pale."
    p = parse_observation(raw)
    assert p.food == "Pancake" and p.status == S.APPEARS_UNDERCOOKED and p.parse_ok


@pytest.mark.parametrize(
    "line,status",
    [
        ("POSSIBLY_OVERCOOKED_OR_BURNT", S.POSSIBLY_OVERCOOKED_OR_BURNT),
        ("Possibly overcooked or burnt", S.POSSIBLY_OVERCOOKED_OR_BURNT),
        ("POSSIBLY_OVERCOOKED/BURNT", S.POSSIBLY_OVERCOOKED_OR_BURNT),
        ("APPEARS_UNDERCOOKED", S.APPEARS_UNDERCOOKED),  # not misread as APPEARS_COOKED
        ("Undercooked", S.APPEARS_UNDERCOOKED),
        ("CANNOT_DETERMINE", S.CANNOT_DETERMINE),
        ("cannot be determined", S.CANNOT_DETERMINE),
        ("Appears cooked", S.APPEARS_COOKED),
    ],
)
def test_status_variants(line, status):
    p = parse_observation(f"Food Identified: Toast\nCooking Status: {line}\n")
    assert p.status == status and p.parse_ok


def test_unknown_food_is_none():
    p = parse_observation("Food Identified: Unknown\nCooking Status: CANNOT_DETERMINE")
    assert p.food is None and p.status == S.CANNOT_DETERMINE


def test_free_form_output_is_cannot_determine():
    p = parse_observation("Sure! This looks like a lovely breakfast, it is not burnt at all.")
    # "burnt" in free text must not trigger an overcook status.
    assert p.status == S.CANNOT_DETERMINE and not p.parse_ok


def test_status_token_outside_format_is_accepted_but_flagged():
    p = parse_observation("I think this is APPEARS_UNDERCOOKED chicken.")
    assert p.status == S.APPEARS_UNDERCOOKED and not p.parse_ok


def test_safety_claims_stay_in_recommendation_only():
    raw = CANONICAL.replace("It looks done. Remove it from the heat.", "The egg is safe to eat.")
    p = parse_observation(raw)
    assert "safe to eat" in p.recommendation and "safe" not in p.evidence


def test_mock_output_roundtrips_through_parser():
    for status in S:
        p = parse_observation(format_response("Rice", status))
        assert p.status == status and p.food == "Rice" and p.parse_ok
