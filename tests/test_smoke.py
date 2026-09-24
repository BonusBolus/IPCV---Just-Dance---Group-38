"""Integration smoke tests: the whole pipeline runs without a camera. Run with `pytest`."""
import numpy as np

from core.config import Config
from core.image_utils import overlay_rgba, warp_rgba_affine
from core.types import EventType, FrameData, GameEvent, Grade
from gameplay.choreography import Choreography
from gameplay.game_state import GameState
from main import App
from scene.state_machine import Phase


def _frame(i: int) -> FrameData:
    return FrameData(image=np.zeros((360, 640, 3), np.uint8), timestamp=i / 30, index=i)


def _app() -> App:
    cfg = Config()
    cfg.game.enable_audio = False
    cfg.game.lobby_confirm_s = 0.1
    cfg.game.countdown_s = 0.1
    return App(cfg, use_mock=True, load_models=False)


def test_pipeline_runs_through_all_phases():
    app = _app()
    app.handle_key(ord(" "))  # START -> LOBBY
    t = 0.0
    for i in range(20):
        t += 1 / 30
        app.process_frame(_frame(i), now=t)
    assert len(app.tracker.active_players()) == 2
    app.phases.ready = {1, 2}  # mock dancers don't raise their arms on command
    seen = set()
    for i in range(20, 60):
        t += 1 / 30
        canvas = app.process_frame(_frame(i), now=t)
        seen.add(app.phases.phase)
        assert canvas.shape[1] == app.cfg.display.width
    assert Phase.PLAYING in seen
    app.phases.go(Phase.RESULTS, t)
    app.process_frame(_frame(99), now=t)
    assert not app.guard.failures, app.guard.last_error


def test_choreography_roundtrip(tmp_path):
    c = Choreography.synthetic(duration=20.0)
    c.save(tmp_path / "c.json")
    d = Choreography.load(tmp_path / "c.json")
    assert len(d.times) == len(c.times) and len(d.moves) == len(c.moves) and d.bpm == c.bpm
    ref = d.reference_at(5.0)
    assert ref is not None and ref.keypoints.shape == (17, 2)
    assert d.reference_at(-1.0) is None
    times, kps, _ = d.window(4.9, 5.1)
    assert len(times) == len(kps) > 0 and np.all((times >= 4.9) & (times <= 5.1))


def test_game_state_winner_and_combo():
    gs = GameState()
    gs.reset([1, 2])
    gs.apply([GameEvent(EventType.GRADE, 1.0, pid=1, grade=Grade.PERFECT),
              GameEvent(EventType.GRADE, 1.0, pid=2, grade=Grade.OK),
              GameEvent(EventType.GRADE, 2.0, pid=1, grade=Grade.GOOD)])
    assert gs.winner() == 1
    assert gs.stats[1].combo == 2 and gs.stats[2].combo == 0
    gs.reset([1, 2])
    assert gs.winner() is None  # tie


def test_stickers_clip_at_border():
    img = np.zeros((100, 100, 3), np.uint8)
    sticker = np.full((40, 40, 4), 255, np.uint8)
    overlay_rgba(img, sticker, center=(0, 0), scale=1.0, angle_deg=30)
    overlay_rgba(img, sticker, center=(500, 500))  # fully outside: no error
    warp_rgba_affine(img, sticker, [[0, 0], [40, 0], [0, 40]], [[90, 90], [130, 95], [88, 130]])
    assert img[0, 0].sum() > 0 and img[95, 95].sum() > 0
