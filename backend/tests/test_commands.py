"""Transcript-only commands: deterministic, no model, no frame needed."""

import pytest

from app.commands import parse_command, parse_seconds


@pytest.mark.parametrize(
    "text,seconds",
    [
        ("two minutes", 120),
        ("30 seconds", 30),
        ("1 hour and 5 minutes", 3900),
        ("one and a half minutes", 90),
        ("half a minute", 30),
        ("nothing here", None),
    ],
)
def test_parse_seconds(text, seconds):
    assert parse_seconds(text) == seconds


@pytest.mark.parametrize(
    "text,intent,seconds",
    [
        ("add two minutes", "add_time", 120),
        ("take off 30 seconds", "subtract_time", 30),
        ("set a timer for 10 minutes", "set_timer", 600),
        ("5 minute timer", "set_timer", 300),
        ("ten minutes", "set_timer", 600),
        ("cancel", "cancel", None),
        ("cancel the timer", "cancel", None),
        ("stop watching", "end_session", None),
        ("I'm done", "end_session", None),
        ("pause", "pause", None),
        ("resume the timer", "resume", None),
        ("yes", "confirm", None),
        ("no thanks", "reject", None),
        ("how does it look?", "status", None),
        ("is it done?", "status", None),
        ("how much time is left", "status", None),
        ("what is this?", "start_monitoring", None),
        ("watch my pan", "start_monitoring", None),
        ("banana phone", "unknown", None),
    ],
)
def test_intents(text, intent, seconds):
    cmd = parse_command(text)
    assert cmd.intent == intent
    assert cmd.seconds == seconds


@pytest.mark.parametrize("text,food", [("how long for pasta?", "pasta"), ("how long do I cook chicken", "chicken")])
def test_cook_time_question(text, food):
    cmd = parse_command(text)
    assert cmd.intent == "cook_time_question" and cmd.food == food
