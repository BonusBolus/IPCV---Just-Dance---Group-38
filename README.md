# IPCV Just Dance, Group 38

A webcam-based two-player Just Dance game for the IPCV project (University of Twente).
Two players dance along with a model dancer. Their poses are compared to the reference
choreography and scored, and the player with the highest score wins.

## Team and task owners

| Task | Package | Owner |
|---|---|---|
| 1 Face tracking & augmented effects | `face/` | TBD |
| 2 Body pose estimation & motion tracking | `pose/` | TBD |
| 3 Multiplayer detection & identity tracking | `identity/` | TBD |
| 4 Player interaction & game control | `gameplay/` | TBD |
| 5 Scene processing & real-time integration (lead) | `scene/`, `core/`, `main.py` | TBD |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Run

```bash
python main.py                  # webcam 0
python main.py --camera 1       # another webcam
python main.py --mock           # synthetic dancers, no second person or CV modules needed
python main.py --video recordings/crossing_01.mp4    # replay a recording instead of the webcam
python main.py --profile-csv logs/run.csv            # log per-stage timings for the report
python main.py --help
```

| Key | Action |
|---|---|
| SPACE | start / play again |
| N | force next phase (debug) |
| R | back to start screen |
| M | toggle mock players |
| C | mock scenario: dance / cross (players swap places, for the identity tests) |
| D | toggle debug overlay (FPS, per-stage ms, module failures, skeletons) |
| F | fullscreen |
| Q / ESC | quit |

Game flow: `START -> LOBBY (waits for 2 players) -> COUNTDOWN -> PLAYING -> RESULTS`.

## Architecture

```
Camera ─► PoseEstimator (T2) ─┐  anonymous detections (list[PoseObs], list[FaceObs])
      └─► FaceTracker  (T1) ──┤
                              ▼
                    PlayerTracker (T3)          who is who -> dict[pid, Player]
                              ▼
     PoseSmoother (T2) · FaceSmoother (T1) · SpatialEstimator (T3)   per player
                              ▼
     PhaseController (T5) · Scorer · MoveDetector · InteractionDetector (T4) -> GameState
                              ▼
                        Renderer (T5)           final frame
```

- **Contracts:** all modules exchange the dataclasses in [core/types.py](core/types.py).
  Coordinates are normalized to [0, 1], keypoints use the COCO-17 layout. Change these only
  after discussing it with the group.
- **Smoothing runs after identity:** a temporal filter needs to know which detection in
  frame *t* belongs to which in *t-1*, so T1/T2 filters are called per player id.
- **Failure isolation:** every module call goes through `ModuleGuard`. An exception gives a
  fallback value plus a red line in the debug overlay, not a crash.
- **Develop in isolation:** `--mock` feeds synthetic dancers in the exact T1/T2 output format.
  `--video` replays recordings for reproducible tests.
- **Stubs:** methods marked `TODO(Tn)` return safe defaults (e.g. no detections), so the game
  already runs end-to-end. Pure helper functions that nothing calls yet raise
  `NotImplementedError`.

## File structure

```
main.py                 App: wires all modules together, main loop, keyboard
core/                   shared infrastructure (T5)
  types.py              data contracts between tasks
  config.py             all settings
  camera.py             threaded webcam / video capture (always the newest frame)
  profiler.py           per-stage timing, FPS, CSV logging
  guard.py              module failure isolation
  filters.py            EMA baseline + One Euro filter (TODO T1/T2)
  image_utils.py        drawing helpers: text, bars, RGBA stickers, skeletons
  mock_source.py        synthetic dancers for development
face/                   Task 1: face_tracker, face_filter, face_effects
pose/                   Task 2: pose_estimator, keypoint_filter, features
identity/               Task 3: player_tracker, appearance, spatial
gameplay/               Task 4: choreography, scorer, move_detector, interactions,
                                game_state, action_effects
scene/                  Task 5: state_machine, audio, background, renderer
tools/                  record_session.py, extract_reference.py
eval/                   one evaluation script per task (report numbers)
tests/                  smoke tests (pytest)
assets/                 song, choreography, model weights, stickers (see assets/README.md)
```

## Offline tools

```bash
python -m tools.record_session --out recordings/crossing_01.mp4 --seconds 30
python -m tools.extract_reference --video assets/model_dance.mp4   # needs Task 2
pytest
```

## AI and external tools statement

TODO: models, libraries and coding agents used, what each did, how we adapted and tested
them, and their known limitations (required by the assignment, section 5).
