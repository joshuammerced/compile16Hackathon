import pytest

from tests.conftest import FakeClock

from app.schemas import AlertType, Phase
from app.state import CookingSession, SessionPersistence, StateStore, Timer


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
    assert s.phase == Phase.COOKING and s.run_id is not None
    assert s.remaining_ms() == 60_000
    clock.advance(59.5)
    assert s.device_view(0).remaining_s == 1  # rounds up
    assert not s.timer_due()
    clock.advance(0.5)
    assert s.timer_due()
    s.finish()
    assert s.phase == Phase.DONE
    s.end_session("done")
    assert s.phase == Phase.IDLE and s.run_id is None and s.timer is None


def test_session_stop_timer_keeps_watching_and_adjust_only_while_cooking(clock):
    s = CookingSession()
    assert not s.adjust(60)
    s.start_timer(60, "Egg", clock)
    assert s.adjust(60)
    assert s.total_s() == 120
    s.stop_timer("cancel")
    assert s.phase == Phase.MONITORING and s.active
    assert s.timer is None


def test_generation_changes_on_every_session_boundary(clock):
    s = CookingSession()
    g0 = s.generation
    s.start_session("a")
    g1 = s.generation
    s.end_session("b")
    assert g0 < g1 < s.generation


def test_store_version_bumps_on_every_visible_change(clock):
    store = StateStore(clock)
    v0 = store.version
    with store.mutate() as s:
        s.start_timer(60, "Egg", clock)
    with store.mutate() as s:
        s.pause()
    assert store.version == v0 + 2
    snap = store.snapshot()
    assert snap.state == Phase.COOKING and snap.paused and snap.total_s == 60 and snap.timer == "paused"


def test_store_version_ignores_invisible_mutations_and_countdown(clock):
    store = StateStore(clock)
    with store.mutate() as s:
        s.start_timer(300, "Pasta", clock)
    v = store.version
    with store.mutate() as s:
        s.analyze_attempts += 1  # not visible to the device
    clock.advance(10)
    snap = store.snapshot()
    assert store.version == v
    # ...but the countdown in the snapshot is live without any version change.
    assert snap.remaining_ms == 290_000 and snap.timer == "running"


def test_alert_ids_ack_idempotent_and_priority():
    s = CookingSession()
    done = s.add_alert(AlertType.TIMER_DONE, "TIMER DONE", "done")
    burn = s.add_alert(AlertType.OVERCOOK_WARNING, "CHECK PAN!", "burning")
    assert burn.id == done.id + 1
    assert s.top_alert() is burn  # overcook outranks timer completion
    assert s.ack(burn.id) is burn
    assert s.ack(burn.id) is True  # already acked: idempotent
    assert s.ack(999) is None  # unknown
    assert s.top_alert() is done


def test_persistence_roundtrip_keeps_timer_running_through_downtime(tmp_path, clock):
    wall = FakeClock(5000.0)
    path = tmp_path / "session.json"
    store = StateStore(clock, SessionPersistence(path, wall=wall))
    with store.mutate() as s:
        s.start_timer(600, "Rice", clock)
        s.add_alert(AlertType.OVERCOOK_WARNING, "CHECK PAN!", "burning")
    clock.advance(100)
    with store.mutate() as s:
        s.display_text = "Rice"  # any mutation re-saves the snapshot
    saved_version = store.version
    run_id = store.snapshot().session_id

    # The backend is down for 50 s, then restarts with a fresh monotonic clock.
    wall.advance(50)
    clock2 = FakeClock(7.0)
    restored = StateStore(clock2, SessionPersistence(path, wall=wall))
    snap = restored.snapshot()
    assert snap.session_id == run_id and snap.state == Phase.COOKING
    assert abs(snap.remaining_ms - 450_000) < 50  # 600 - 100 - 50
    assert snap.alert and snap.alert_type == AlertType.OVERCOOK_WARNING
    assert restored.version > saved_version  # devices see a change after a restart
    clock2.advance(450)
    with restored.read() as s:
        assert s.timer_due()


def test_persistence_paused_timer_does_not_count_downtime(tmp_path, clock):
    wall = FakeClock()
    path = tmp_path / "s.json"
    store = StateStore(clock, SessionPersistence(path, wall=wall))
    with store.mutate() as s:
        s.start_timer(120, "Egg", clock)
        clock.advance(20)
        s.pause()
    wall.advance(1000)
    snap = StateStore(FakeClock(1.0), SessionPersistence(path, wall=wall)).snapshot()
    assert snap.paused and abs(snap.remaining_ms - 100_000) < 50


def test_corrupt_snapshot_starts_fresh(tmp_path, clock):
    path = tmp_path / "s.json"
    path.write_text("{not json")
    assert StateStore(clock, SessionPersistence(path)).snapshot().state == Phase.IDLE
