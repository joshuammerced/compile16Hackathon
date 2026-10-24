import pytest

from app.schemas import Phase
from app.state import CookingSession, StateStore, Timer


def test_timer_counts_down_with_monotonic_clock(clock):
    t = Timer(300, "Pasta", clock)
    assert t.remaining() == 300
    clock.advance(100)
    assert t.remaining() == 200
    assert t.ends_at == 1000 + 300


def test_timer_pause_resume(clock):
    t = Timer(300, "Pasta", clock)
    clock.advance(50)
    assert t.pause()
    assert not t.pause()  # already paused
    clock.advance(1000)
    assert t.remaining() == 250
    assert t.ends_at is None
    assert t.resume()
    clock.advance(50)
    assert t.remaining() == 200
    assert t.ends_at == 1000 + 1000 + 300


def test_timer_adjust_and_expire(clock):
    t = Timer(120, "Egg", clock)
    t.adjust(60)
    assert t.remaining() == 180
    t.adjust(-90)
    assert t.remaining() == 90
    clock.advance(90)
    assert t.expired
    t.adjust(-1000)  # never negative
    assert t.remaining() == 0


def test_timer_rejects_non_positive():
    with pytest.raises(ValueError):
        Timer(0)


def test_session_lifecycle(clock):
    s = CookingSession()
    s.start_timer(60, "Egg", clock)
    assert s.phase == Phase.COOKING
    assert s.remaining_s() == 60
    clock.advance(59.5)
    assert s.remaining_s() == 1  # rounds up
    assert not s.timer_due()
    clock.advance(0.5)
    assert s.timer_due()
    s.finish()
    assert s.phase == Phase.DONE and s.alert
    assert s.acknowledge()
    assert s.phase == Phase.IDLE and not s.alert


def test_session_cancel_and_adjust_only_while_cooking(clock):
    s = CookingSession()
    assert not s.adjust(60)
    s.start_timer(60, "Egg", clock)
    assert s.adjust(60)
    assert s.total_s() == 120
    s.cancel_timer()
    assert s.phase == Phase.IDLE
    assert s.timer is None


def test_store_version_bumps_on_every_mutation(clock):
    store = StateStore(clock)
    v0 = store.version
    with store.mutate() as s:
        s.start_timer(60, "Egg", clock)
    with store.mutate() as s:
        s.pause()
    assert store.version == v0 + 2
    snap = store.snapshot()
    assert snap.state == Phase.COOKING and snap.paused and snap.total_s == 60
