# IPCV - Just Dance - Group 38

Computer vision assignment: a Just Dance game for **1 to 5 players** (normally 2), controlled only with the camera.
The game finds all players (YOLO pose), keeps track of who is who (shirt colour), grades their pose
against the move on screen, and adds fun head effects (big heads, the player name on the forehead, and a crown for the player who is ahead).

## Install

```
pip install -r requirements.txt
```

Needs OpenCV 4.5.4 or newer (for the YuNet face detector). The model files are in the repository:
`yolo26n-pose.pt` (pose) and `face_tracking/face_detection_yunet_2023mar.onnx` (faces).

## Run

```
python main.py                  # webcam 0
python main.py --camera 1       # another webcam
python main.py --video file.mp4 # a video file instead of the webcam
python main.py --debug          # debug drawings (boxes, FPS, state); n jumps to the next state
python main.py --windowed       # a window instead of full screen (q or ESC closes the game)
```

`q` or `ESC` closes the game. That is only a safety exit: the game itself is played with gestures.

## How to play

1. **Song selection:** raise your **left hand** for the previous song and your **right hand** for the next
   song (hold it for half a second, then lower it again). You hear a preview of the song. Raise **both hands**
   for 1 s to choose it.
2. **Registration:** stand in front of the camera, 1 to 5 players. Each player gets a colour and the label
   "Player 1", "Player 2", ... on the forehead. Wear different coloured shirts, so the game does not mix you up.
3. **Raise both hands** (wrists above your nose) and hold them for 1 s. The bar above your head fills up.
   When everyone who stepped in is ready, the countdown starts. Someone who walked away for 3 s is not waited for.
4. **Playing:** the coming moves slide to the hit line at the bottom; be in the pose when it reaches the line,
   and copy the pose on the **NOW** card. Between moves the card already shows the **NEXT** move. Each move gives
   points once: Perfect, Good, OK or Miss, shown next to your score. With 2 players the score bar has your score on
   your side; with 1 or 3-5 players there is a row of score boxes at the top, in the order you stand.
   Gold moves count double. Duets: neighbours do them together (from left to right 1st+2nd, 3rd+4th); a 5th player
   does the left part alone, without the duet bonus. The player who is ahead wears a gold crown. If you walk out of the image, the game
   warns you and the song keeps going.
5. **Results:** a ranking (1st, 2nd, ...) and the winner. Raise both hands (or wait 20 s) to go back to the song selection.

The number of players is a setting in `config.py`: `MAX_PLAYERS` (5) and `MIN_PLAYERS` (1).
See `docs/five_players.md` for what to keep in mind with many players (room, camera, shirt colours).

## Make an executable

You can pack the game into one folder with a program, so others can play without installing Python.

1. Make a **separate** Python environment with the **CPU version** of PyTorch. The normal PyTorch is the CUDA
   version, which adds about 4 GB of NVIDIA libraries that the game does not use (it runs YOLO with OpenVINO).
   ```
   python -m venv build-env
   build-env/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   build-env/bin/pip install -r requirements.txt pyinstaller
   ```
   On Windows use `build-env\Scripts\pip` and `build-env\Scripts\python`.
2. Build (about 2 minutes):
   ```
   build-env/bin/python -m PyInstaller just_dance.spec --noconfirm
   ```
3. The game is `dist/JustDance/JustDance` (Windows: `dist\JustDance\JustDance.exe`). It takes the same options
   as `main.py` (`--debug`, `--windowed`, `--camera 1`, ...). Share the **whole** `dist/JustDance` folder (about 1.5 GB).

PyInstaller makes a program for the system you build on: build on Windows for Windows players. The recipe
(`just_dance.spec`) packs the YOLO model, its OpenVINO version, the face model and all songs (`.json` + `.ogg`).

## Project structure

| Folder / file | What it does |
|---|---|
| `main.py` | Reads the command line arguments and starts the game |
| `config.py` | All settings: paths, hold time, countdown time, head enlargement, ... |
| `camera.py` | Webcam / video input (mirrored, always the newest image) and the FPS counter |
| `game_logic/` | `game.py` runs the loop; `states/` has one file per game state; `song.py` the music and moves; `pose_grading.py` the score of a pose |
| `pose_tracking/` | YOLO pose estimation: the keypoints of every person |
| `identity_tracking/` | Which person is which player (shirt colour + position) |
| `face_tracking/` | Finds the faces; big heads, the crown and the player label |
| `scene/` | Draws the screens: title, scores, move cards, song selection, registration and results |
| `choreography/` | `poses.json` (all poses), one `.json` per song (the moves on the beats) and the music (`.ogg`). Every song `.json` here shows up in the song selection |

## Documentation

| Document | About |
|---|---|
| [docs/game_structure.md](docs/game_structure.md) | The game states, the files, the song format, scoring |
| [docs/scene_changes.md](docs/scene_changes.md) | The `scene/` folder: which file draws which part of the screen |
| [docs/face_tracking.md](docs/face_tracking.md) | Face detection, big heads, the crown and the player label |
| [docs/identity_tracking.md](docs/identity_tracking.md) | How players are recognised by their shirt colour |
