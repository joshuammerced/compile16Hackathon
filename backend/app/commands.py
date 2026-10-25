"""Deterministic parser for spoken/typed user commands.

User commands are kept apart from visual observations: what the user *asked for* comes from
here, what the food *looks like* comes from the vision model, and the controller combines them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Intent = Literal[
    "start_monitoring",  # "what is this?", "watch my pan"
    "end_session",  # "stop watching", "I'm done"
    "cancel",  # "cancel the timer"
    "set_timer",  # "set a timer for 10 minutes"
    "add_time",
    "subtract_time",
    "pause",
    "resume",
    "confirm",  # "yes", "start it"
    "reject",  # "no", "skip"
    "status",  # "how does it look?", "how much time is left?"
    "cook_time_question",  # "how long for pasta?"
    "unknown",
]


@dataclass(frozen=True)
class Command:
    intent: Intent
    seconds: int | None = None
    food: str | None = None
    text: str = ""


NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45, "fifty": 50, "sixty": 60,
}  # fmt: skip

_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?|[a-z]+(?:-[a-z]+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)\b")


def parse_seconds(text: str) -> int | None:
    """Sum every duration in the text: 'one hour and 5 minutes' -> 3900, 'half a minute' -> 30."""
    text = text.lower()
    if re.search(r"\bhalf (a|an) minute\b", text):
        return 30

    def and_a_half(m: re.Match) -> str:  # "one and a half minutes" -> "1.5 minutes"
        word = m.group(1)
        n = float(word) if word[0].isdigit() else NUMBER_WORDS.get(word)
        return f"{n + 0.5} {m.group(2)}" if n is not None else m.group(0)

    text = re.sub(r"(\d+|[a-z]+) and a half (hours?|minutes?)", and_a_half, text)
    total = 0
    for word, unit in _DURATION_RE.findall(text):
        n = float(word) if word[0].isdigit() else NUMBER_WORDS.get(word)
        if n is None:
            continue
        total += int(n * (3600 if unit.startswith("h") else 60 if unit.startswith("m") else 1))
    return total or None


def _has(text: str, *patterns: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def parse_command(transcript: str) -> Command:
    text = " ".join(transcript.lower().replace("’", "'").split()).strip(" .!")
    secs = parse_seconds(text)

    def cmd(intent: Intent, **kw) -> Command:
        return Command(intent=intent, text=text, **kw)

    if not text:
        return cmd("unknown")
    if _has(text, r"\bstop (watching|monitoring|looking)\b", r"\b(i'?m|i am|we'?re|all) (done|finished)\b",
            r"\bend (the )?session\b", r"\bturn (it )?off\b", r"\bgoodbye\b"):  # fmt: skip
        return cmd("end_session")
    if _has(text, r"\b(cancel|stop|clear|kill)\b.*\btimer\b", r"^(cancel|stop|never ?mind)$", r"\bcancel\b"):
        return cmd("cancel")
    if _has(text, r"\bpause\b", r"\bhold on\b"):
        return cmd("pause")
    if _has(text, r"\b(resume|unpause|continue)\b"):
        return cmd("resume")
    if secs and _has(text, r"\b(add|plus|more|extra|another|longer)\b"):
        return cmd("add_time", seconds=secs)
    if secs and _has(text, r"\b(remove|minus|less|take off|subtract|shorter|take away)\b"):
        return cmd("subtract_time", seconds=secs)
    if secs and _has(text, r"\btimer\b", r"\bset\b", r"\bcount ?down\b", r"\bremind me\b", r"^(for )?\S+ \S+$"):
        return cmd("set_timer", seconds=secs)
    m = re.search(r"\bhow long (?:do i|should i|does|to|for|will)?\s*(?:cook|boil|fry|bake|take|it take)?\s*(?:for |a |an |the |my )*(.+)$", text)
    if m and not _has(text, r"\b(left|remaining)\b"):
        food = re.sub(r"\b(take|cook|for|to cook|need)\b", "", m.group(1)).strip(" ?")
        if food and food not in ("it", "this", "that"):
            return cmd("cook_time_question", food=food)
    if _has(text, r"\bhow (does|do|is) (it|this|that|the \w+) (look|doing)\b", r"\bis it (done|ready|cooked|burning)\b",
            r"\b(time|how much|how long).*(left|remaining)\b", r"\bstatus\b", r"\bhow'?s it (look|go)ing\b"):  # fmt: skip
        return cmd("status")
    if _has(text, r"^(yes|yeah|yep|sure|ok(ay)?|do it|go|start( it)?|start the timer|sounds good)( please)?$"):
        return cmd("confirm")
    if _has(text, r"^(no|nope|skip|no thanks|don'?t)( it| thanks)?$", r"\bno timer\b"):
        return cmd("reject")
    if _has(text, r"\bwhat (is|'s) (this|that|cooking)\b", r"\b(watch|monitor|look at|check)\b", r"\bstart\b",
            r"\bi'?m (cooking|making)\b"):  # fmt: skip
        return cmd("start_monitoring")
    return cmd("unknown")
