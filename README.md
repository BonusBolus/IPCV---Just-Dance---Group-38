# IPCV - Just Dance - Group 38

A Just Dance game for **2 players**, controlled only with a webcam (IPCV 2026-1A, University of Twente).
The players copy the moves on the screen and get points for every move. The game:

- finds the body of every player (YOLO pose) and grades their pose against the move on screen;
- keeps track of who is who by the colour of their shirt;
- adds head effects: big heads, the player name on the forehead and a crown for the player who is ahead;
- replaces the room with a Just Dance stage (person segmentation); people walking past disappear;
- has duets (two players together), gold moves (double points), animated moves (a movement to follow, with a chain
  bonus) and a bonus star to grab.

![The game: the timeline at the bottom, the NOW card on the right](docs/figures/screenshot_timeline_next.png)

Thijs van der Burg (Task 1, face tracking), Len Versteeg (Task 2, pose estimation; Task 4), Joost ten Bolscher
(Task 3, identity tracking), Wessel Witteveen (system design; Task 4, interaction), Bram Hoernig (Task 5, scene).

## In the submission ZIP

| File | What |
|---|---|
| `Group38_Report.pdf` | The technical report. The **AI statement** is its appendix ("AI Usage Declaration and External Tools") |
| `Group38_Poster.pdf` | The poster |
| `yolo26n-pose.pt`, `yolo26n-seg.pt` | The two YOLO models (they are not in git) |
| everything else | The code, songs, tests and documentation, as described below |

## Install

Tested with **Python 3.11** on Ubuntu 26.04, on a laptop without a GPU (the models run on the CPU with OpenVINO).
Windows should work the same way, but we did not test it.

You need:
- a **webcam**. A 1080p webcam gives the sharpest big heads; a 720p webcam also works;
- **ffmpeg**, only for `--record` (Ubuntu: `sudo apt install ffmpeg`);
- about 2 GB of disk space for the Python packages.

In the project folder:

```
python3.11 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

The second line installs the CPU version of PyTorch (needed by Ultralytics); without it, pip takes the CUDA
version, which is about 4 GB bigger. All other versions are in `requirements.txt`. The last line makes `python` use
this environment; do it again in every new terminal. On Windows: `.venv\Scripts\pip` and `.venv\Scripts\activate`.

Models:
- the face model is in the project: `face_tracking/face_detection_yunet_2023mar.onnx` (YuNet, OpenCV Zoo, 2023 March
  version);
- the YOLO models `yolo26n-pose.pt` (pose) and `yolo26n-seg.pt` (person segmentation for the stage) are in the
  submission ZIP. When they are missing, Ultralytics 8.4.189 downloads them the first time you start the game
  (from the Ultralytics GitHub releases, about 7 MB each).

The first start converts both YOLO models to OpenVINO (the `*_openvino_model/` folders, a few seconds).

## Commands

Run every command in the project folder, with the environment active.

### The game: `python main.py [options]`

| Option | What it does |
|---|---|
| (none) | Play with webcam 0, full screen |
| `--camera N` | Use webcam number N (default 0) |
| `--video FILE` | Use a video file instead of the webcam |
| `--no-mirror` | Do not mirror the image. Use it with `--video` for a `webcam.mp4` of `--record` (already mirrored) |
| `--windowed` | A window instead of full screen |
| `--debug` | Debug drawings: skeletons, player boxes, the target pose, FPS and the state |
| `--measure` | Time every stage of every frame; the results go to `measurements/<date_time>/` ([docs/measurements.md](docs/measurements.md)) |
| `--duration S` | Stop by itself after S seconds |
| `--record` | Record the webcam and the game as MP4 videos in `recordings/<date_time>/`; stops after 60 s ([docs/demo_video.md](docs/demo_video.md)) |

Options can be combined. Examples:

```
python main.py --windowed --debug
python main.py --measure --duration 120
python main.py --video recordings/<date_time>/webcam.mp4 --no-mirror --windowed --measure
```

Keys (the game itself is played only with gestures):

| Key | What it does |
|---|---|
| `q` or `ESC` | Close the game (a safety exit) |
| `n` | Only with `--debug`: jump to the next state |

### Tests and evaluation

| Command | What it does |
|---|---|
| `python -m unittest discover -s tests` | Run all unit tests (about 10 s, no webcam needed) |
| `python tests/eval_identity.py [options]` | The evaluation of the identity tracking (Task 3) with two players: accuracy, ID switches, recovery and speed. Options: `--camera N`, `--video FILE`, `--condition NAME`, `--subjects "A,B"`, `--distance M`, `--no-faces`, `--record FILE`, `--save-video`, `--overlap X`. The keys and the test protocol are in [docs/identity_tracking.md](docs/identity_tracking.md); the results are in `tests/results/` |

### Building

| Command | What it does |
|---|---|
| `tools/make_zip.sh [report.pdf] [poster.pdf]` | Make the submission ZIP: the committed code, the two YOLO models and the PDFs |
| `python -m PyInstaller just_dance.spec --noconfirm` | Make an executable, see below |

## How to play

1. **Song selection:** raise your **left hand** for the previous song and your **right hand** for the next
   song (hold it for 0.3 s, then lower it again). You hear a preview of the song. Raise **both hands**
   for 0.5 s to choose it.
2. **Registration:** stand in front of the camera, both players, so the camera sees you from head to knees. Each
   player gets a colour and the label "Player 1" / "Player 2" on the forehead. Wear different coloured shirts, so
   the game does not mix you up.
3. **Raise both hands** (wrists above your nose) and hold them for 0.5 s. The bar above your head fills up.
   When both players are ready, the countdown starts.
4. **Playing:** the coming moves slide to the hit line at the bottom; be in the pose when it reaches the line,
   and copy the pose on the **NOW** card. Between moves the card already shows the **NEXT** move. Each move gives
   points once: Perfect, Good, OK or Miss, shown next to your score (your score is on your side of the bar).
   - **Gold moves** count double.
   - **Duets** (two figures in the player colours): both players do their own half of the pose, together. Both
     get +25%.
   - **Animated moves** (magenta): the figure on the card moves, follow it step by step. Every correct step in a
     row makes your chain longer and gives extra points ("Chain x3").
   - **Bonus star:** a gold star floats across the screen at hand height. The first player who touches it with a
     hand gets **+100** points.

   The player who is ahead wears a gold crown. If you walk out of the image, the game warns you and the song keeps
   going.
5. **Results:** the scores and the winner. Raise both hands (or wait 20 s) to go back to the song selection.

### Songs

| Song | Tempo (bpm) |
|---|---|
| Rasputin (selected first, used for the demo video) | 127 |
| Call Me Maybe | 120 |
| Gangnam Style | 132 |
| Lollipop | 143 |
| Song 1: a test song with generated music, used by the tests | 120 |

## Settings for the demo

All settings are in `config.py`. Some are set for the 60-second demo video, so do not be surprised:

| Setting | Demo value | What it does | For a normal game |
|---|---|---|---|
| `SONG_MAX_TIME` | 35.0 | Every song stops after 35 s (the music fades out) | `None`: full songs |
| `STAR_TIMES` | 6 s and 21 s | The bonus stars come at fixed times in the song | `None`, with `STAR_INTERVAL = 20.0` and `STAR_JITTER = 5.0` |
| `DEBUG_IN_RESULTS` | True | The results screen shows the debug drawings | False |
| `HOLD_TIME`, `COUNTDOWN_TIME` | 0.5 s, 2 s | Short waits | 1.0 s and 3.0 s if you like |

## Make an executable

You can pack the game into one folder with a program, so others can play without installing Python.

1. Make a **separate** Python environment with the **CPU version** of PyTorch. The normal PyTorch is the CUDA
   version, which adds about 4 GB of NVIDIA libraries that the game does not use (it runs YOLO with OpenVINO).
   ```
   python3.11 -m venv build-env
   build-env/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   build-env/bin/pip install -r requirements.txt pyinstaller
   ```
   On Windows use `build-env\Scripts\pip` and `build-env\Scripts\python`.
2. Start the game once (`python main.py`), so the YOLO models and their OpenVINO versions are in the project folder.
3. Build (about 2 minutes):
   ```
   build-env/bin/python -m PyInstaller just_dance.spec --noconfirm
   ```
4. The game is `dist/JustDance/JustDance` (Windows: `dist\JustDance\JustDance.exe`). It takes the same options
   as `main.py`. Share the **whole** `dist/JustDance` folder (about 1.5 GB).

PyInstaller makes a program for the system you build on: build on Windows for Windows players. The recipe
(`just_dance.spec`) packs the YOLO models, their OpenVINO versions, the face model and all songs (`.json` + `.ogg`).

## Project structure

| Folder / file | What it does |
|---|---|
| `main.py` | Reads the command line options and starts the game |
| `config.py` | All settings: paths, hold times, countdown, head enlargement, star, demo settings, ... |
| `camera.py` | Webcam / video input in its own thread (mirrored, always the newest image) and the FPS counter |
| `game_logic/` | `game.py` runs the loop; `states/` has one file per game state; `song.py` the music and moves; `pose_grading.py` the score of a pose; `motion.py` the motion check; `bonus_star.py` the star; `measurements.py` and `recorder.py` for `--measure` and `--record` |
| `pose_tracking/` | YOLO pose estimation: the keypoints of every person (Task 2) |
| `identity_tracking/` | Which person is Player 1 and which is Player 2: shirt colour + position (Task 3) |
| `face_tracking/` | Finds the faces; big heads, the crown and the player label (Task 1) |
| `background/` | The stage instead of the room: person segmentation and the stage itself (Task 5) |
| `scene/` | Draws the screens: title, scores, timeline, move cards, song selection, registration, results, star (Task 5) |
| `choreography/` | `poses.json` (all poses), one `.json` per song (the moves on the beats) and the music (`.ogg`). Every song `.json` here shows up in the song selection |
| `tests/` | Unit tests, and `eval_identity.py` with its results in `tests/results/` |
| `docs/` | Documentation per part of the game (below); `figures/` has the system overview and a screenshot |
| `tools/make_zip.sh` | Makes the submission ZIP |
| `requirements.txt` | The Python packages and their versions |
| `just_dance.spec` | The PyInstaller recipe for the executable |

Made while playing (not in git): `measurements/` (`--measure`), `recordings/` (`--record`) and the
`*_openvino_model/` folders.

## Documentation

| Document | About |
|---|---|
| [docs/game_structure.md](docs/game_structure.md) | The game states, the files, the song format and the scoring |
| [docs/face_tracking.md](docs/face_tracking.md) | Task 1: face detection, big heads, the crown and the player label |
| [docs/animated_moves.md](docs/animated_moves.md) | Task 2 and 4: animated moves, the chain bonus and the motion check |
| [docs/identity_tracking.md](docs/identity_tracking.md) | Task 3: how players are recognised by their shirt colour, and its evaluation |
| [docs/background_and_star.md](docs/background_and_star.md) | Task 4 and 5: the stage behind the players and the bonus star |
| [docs/scene.md](docs/scene.md) | Task 5: the `scene/` folder, which file draws which part of the screen |
| [docs/measurements.md](docs/measurements.md) | Measuring the frame rate and the time of every stage (`--measure`) |
| [docs/demo_video.md](docs/demo_video.md) | Recording the demo video (`--record`) and what the 35 s of Rasputin show |
