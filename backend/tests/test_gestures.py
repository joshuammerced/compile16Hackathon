from app.vision.gestures import GestureDebouncer


def feed(deb, seq, start=0.0, dt=0.1):
    """Feed (gesture, score) pairs at fixed intervals; return fired events."""
    events = []
    t = start
    for gesture, score in seq:
        ev = deb.update(gesture, score, t)
        if ev:
            events.append(ev)
        t += dt
    return events


def test_fires_after_hold_frames():
    deb = GestureDebouncer(hold_frames=5, min_score=0.6, cooldown_s=3)
    events = feed(deb, [("Thumb_Up", 0.9)] * 4)
    assert events == []
    events = feed(deb, [("Thumb_Up", 0.9)], start=0.4)
    assert len(events) == 1 and events[0].gesture == "Thumb_Up"


def test_interruption_resets_hold():
    deb = GestureDebouncer(hold_frames=5, min_score=0.6, cooldown_s=0)
    seq = [("Thumb_Up", 0.9)] * 4 + [(None, 0.0)] + [("Thumb_Up", 0.9)] * 4
    assert feed(deb, seq) == []


def test_low_score_does_not_count():
    deb = GestureDebouncer(hold_frames=3, min_score=0.6, cooldown_s=0)
    seq = [("Open_Palm", 0.9), ("Open_Palm", 0.5), ("Open_Palm", 0.9), ("Open_Palm", 0.9)]
    assert feed(deb, seq) == []


def test_switching_gesture_restarts_count():
    deb = GestureDebouncer(hold_frames=3, min_score=0.6, cooldown_s=0)
    seq = [("Thumb_Up", 0.9)] * 2 + [("Thumb_Down", 0.9)] * 3
    events = feed(deb, seq)
    assert [e.gesture for e in events] == ["Thumb_Down"]


def test_untracked_gestures_ignored():
    deb = GestureDebouncer(hold_frames=2, min_score=0.6, cooldown_s=0)
    assert feed(deb, [("Victory", 0.99)] * 10) == []


def test_cooldown_blocks_repeat_then_allows():
    deb = GestureDebouncer(hold_frames=2, min_score=0.6, cooldown_s=3)
    # Hold for 4 s at 10 fps: fires at t=0.1, then nothing until t>=3.1.
    events = feed(deb, [("Thumb_Up", 0.9)] * 40)
    times = [round(e.timestamp, 1) for e in events]
    assert times[0] == 0.1
    assert all(b - a >= 3 for a, b in zip(times, times[1:]))
    assert len(events) == 2
