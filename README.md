# IPCV Just Dance, Group 38

A webcam-based two-player Just Dance game for the IPCV project (University of Twente).
Two players dance along with a model dancer. Their poses are compared to the reference
choreography and scored, and the player with the highest score wins. The game is controlled
entirely by body movement: raise both arms to start and to get ready.

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
.venv\Scripts\activate                 # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m tools.download_models        # MediaPipe model weights -> assets/models/ (~13 MB)
```
The song (`assets/song.wav`) and the choreography (`assets/choreography.json`) are generated
automatically on the first start.

## Run

```bash
python main.py                  # webcam 0
python main.py --camera 1       # another webcam
python main.py --mock           # synthetic dancers (no second person needed)
python main.py --video recordings/crossing_01.mp4    # replay a recording instead of the webcam
python main.py --profile-csv logs/run.csv            # log per-stage timings for the report
python main.py --help
```

| Control | Action |
|---|---|
| both arms up (hold ~1 s) | start / "I'm ready" in the lobby / play again |
| SPACE | start / play again (keyboard fallback) |
| N | force next phase (debug) |
| R | back to start screen |
| M | toggle mock players |
| C | mock scenario: dance / cross (players swap places) |
| D | toggle debug overlay (FPS, per-stage ms, positions, module failures, skeletons) |
| F | fullscreen |
| Q or ESC | quit |

**How to play.** Stand about 2 m from the camera, next to each other, with at least your upper
body in view. In the lobby, each player raises both arms to get ready. Then copy the golden
dancer in the middle. Every move is graded PERFECT / GOOD / OK / MISS. Bonus points come from:
- **gold moves:** the move is recognised explicitly while it is shown
- **duet moves:** both players get GOOD or better
- **HIGH FIVE:** your hands meet, measured in metres
- **SWAP PLACES:** you walk past each other

Game flow: `START -> LOBBY -> COUNTDOWN -> PLAYING (64 s song) -> RESULTS`.

## Architecture

```
Camera thread (newest frame only)
  └─► PoseEstimator (T2) ─► FaceTracker (T1, head crops)     anonymous detections
        └─► PlayerTracker (T3)                                who is who -> dict[pid, Player]
              └─► PoseSmoother (T2) · FaceSmoother (T1) · SpatialEstimator (T3)
                    └─► MoveDetector (T4) ─► PhaseController (T5)   gestures & game flow
                    └─► Scorer · InteractionDetector (T4) ─► GameState
                          └─► Renderer (T5): background replacement, scene FX,
                              effects of T1/T4, model dancer, HUD
```

- **Contracts:** all modules exchange the dataclasses in [core/types.py](core/types.py).
  Coordinates are normalized to [0, 1] (use `PoseObs.aspect` for angles), and keypoints use
  the COCO-17 layout.
- **Smoothing runs after identity:** a temporal filter needs to know which detection in
  frame *t* belongs to which in *t-1*, so the T1/T2 filters run per player id.
- **Failure isolation:** every module call goes through `ModuleGuard`. An exception gives a
  fallback value plus a red line in the debug overlay, not a crash.
- **Tunable parameters:** all in [core/config.py](core/config.py), grouped per task.

## What each task does

**T1: faces** ([face/](face/)). MediaPipe FaceLandmarker runs on a head crop per player,
built from the pose's head keypoints and upscaled. This makes the ~40 px faces of players at
2.5 m detectable, and each crop has a known owner. Head roll is computed from the eye line,
yaw/pitch from the facial transformation matrix. A One Euro filter smooths the face box.
After a missed detection the face is held for 0.35 s while the effect fades out. The effects
show the game state:
- sunglasses, affine-warped onto the eye corners and nose; the lens style shows the combo
- star eyes on a PERFECT
- a crown on the leader
- name tag plus a live "groove meter" above the head

**T2: pose** ([pose/](pose/)). MediaPipe PoseLandmarker (full model, VIDEO mode,
`num_poses=2`, segmentation masks). Its 33 landmarks are mapped to COCO-17, and visibility is
used as confidence. Unreliable keypoints are held for 0.3 s at below-threshold confidence,
then dropped, so gameplay never acts on them. A One Euro filter smooths per player. Features
(normalized pose, limb directions, joint angles, velocities) are scale- and aspect-invariant.

**T3: identity** ([identity/](identity/)). Each player has a constant-velocity Kalman track.
Detections are matched with the Hungarian algorithm on a cost of:
- predicted position
- torso colour histogram (HS, Bhattacharyya distance)
- body size
- pose shape (limb positions vs. the last frame)

New players are numbered left to right. A lost player keeps their pid and score, and gets it
back on re-entry through appearance matching; spectators cannot steal an identity. Faces are
assigned to bodies with a second Hungarian match. Depth is estimated with a pinhole model from
shoulder width and torso length, giving positions in metres.

**T4: gameplay** ([gameplay/](gameplay/)).
- **Scoring:** the angle error of 8 limb directions against the reference in a ±0.35 s window
  (players lag the model). A segment's score is the 75th percentile of its frame similarities.
- **Move classifier:** rule-based ARMS_UP, T_POSE, CLAP, SQUAT, POINT_LEFT, POINT_RIGHT, with
  0.2 s confirmation and a 1 s cooldown.
- **Interactions:** duet, high five (wrists < 0.3 m apart in metres), swap (left-right order
  flips).
- **Visual feedback:** wrist trails, popups, rings.

**T5: scene & integration** ([scene/](scene/), [main.py](main.py)).
- **Background replacement:** the pose model's person masks are merged, temporally smoothed and
  feathered, then blended onto a disco stage (`cv2.blendLinear`).
- **Scene reactions:** spotlights on the beat, a glow in the leader's colour, a flash on
  PERFECT, a red pulse on MISS, confetti on bonuses.
- **Game flow and sound:** the gesture-driven phase machine and a procedural 120 BPM song whose
  playback position is the game clock.

## Measured results (this laptop, CPU only)

| Measurement | Result | Command |
|---|---|---|
| pose inference (2 people, 640 px, masks) | ~25 ms/frame | `python -m eval.eval_system --video ...` |
| face tracker (2 head crops) | ~10 ms/frame | same |
| identity + smoothing + gameplay | ~2.5 ms/frame | same |
| rendering (1280 px) | ~5–7 ms/frame (compose: 34 ms → 3 ms with `cv2.blendLinear`) | same |
| end-to-end processing | ~21–22 fps with 2 players; webcam delivers 15 fps in this room | debug overlay |
| keypoint jitter raw → One Euro | 12.5 → 3.4 px/frame; error vs. truth 5.1 → 3.6 px, no added lag (EMA: 1 frame) | `python -m eval.eval_pose --mock` |
| identity switches | 0 in 12 crossings, also with 15% (tested up to 30%) missed detections | `python -m eval.eval_identity --mock` |
| re-entry after leaving | 4/4 correct pid | same |
| move classifier | 99.9% on perturbed templates (±15°, noise, scale, aspect) | `python -m eval.eval_gameplay` |
| scoring validity | follower 100% PERFECT; ±35° sloppy mostly GOOD/OK; random 64% MISS; standing still 60% MISS | same |
| false move triggers while free-styling | ≤ 0.5 / min | same |

Real-footage versions: `python -m eval.eval_pose`, `eval_face`, `eval_identity [--distance 2.5]`
run on the webcam or on `--video recordings/...`. Record test clips with
`python -m tools.record_session --out recordings/crossing_01.mp4`.

## Known limitations

- **Faces in profile** (more than ~60° yaw) are not detected. The tag stays, but the glasses
  are hidden.
- **Similar clothing** makes the appearance cue useless. Crossings then rely on motion and pose
  shape only, and re-entry may pick the wrong lost player when both are gone at once.
- **Full occlusion**: MediaPipe returns one merged or missing person, and the hidden player is
  LOST until they reappear. Their move segments during that time count as MISS.
- **Depth accuracy** depends on the configured camera FOV (65°) and body proportions; expect
  roughly 10–20% error. The high-five threshold is set generously for this reason.
- **Timing:** poses are held for 3 beats, so a player up to ~0.7 s late still scores well.
- **Webcam frame rate** drops to ~15 fps in dim light (auto exposure). Use good lighting.
- The synthetic choreography only contains frontal poses. A real model-dancer video can be
  turned into a choreography with `tools/extract_reference.py`, but its moves must then be
  annotated by hand.

## File structure

```
main.py                 App: wires all modules together, main loop, keyboard
core/                   shared infrastructure (T5): types, config, camera thread, profiler,
                        failure guard, filters (EMA, One Euro, Kalman), drawing helpers, mock dancers
face/                   T1: face_tracker, face_filter, face_effects, stickers
pose/                   T2: pose_estimator, keypoint_filter, features
identity/               T3: player_tracker, appearance, spatial
gameplay/               T4: choreography, scorer, move_detector, interactions, game_state, action_effects
scene/                  T5: state_machine, audio, music, background, renderer
tools/                  download_models, make_default_assets, record_session, extract_reference
eval/                   one evaluation script per task + eval_system
tests/                  pytest: smoke test of the full pipeline + module tests
assets/                 models (downloaded), song + choreography (generated), see assets/README.md
```

## AI and external tools statement

TODO (complete per task owner, required by the assignment, section 5). Starting point:
- **Pretrained models:** MediaPipe PoseLandmarker (full) and FaceLandmarker (Google, Apache 2.0),
  used unmodified via the Tasks API.
- **Libraries:** OpenCV (image processing, drawing, affine warps), NumPy, SciPy
  (`linear_sum_assignment`), pygame (audio).
- **Coding agent:** Claude Code (Anthropic) generated the initial project structure and
  implementations. The group reviewed, tested and tuned them: One Euro parameters via
  `eval_pose --mock`, scoring thresholds via `eval_gameplay`, tracker cost terms via
  `eval_identity --mock`. Each owner must be able to explain their module and its limitations
  (see above).
