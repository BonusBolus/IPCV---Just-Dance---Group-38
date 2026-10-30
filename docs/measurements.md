# Measurements for the report

The game can time every stage of every frame. Use it for the timing table of the report (Section 4) and the
frame-rate, latency and resource numbers that the report template asks for.

## How to measure

```
python main.py --measure                    # play normally; close with q or ESC
python main.py --measure --duration 120     # stops by itself after 120 s
python main.py --measure --video test.mp4   # the same recording every time (repeatable)
```

When the game closes, it prints a summary and writes a folder `measurements/<date_time>/` (in the project folder,
or next to the program when you use the executable: `dist/JustDance/measurements/`):

| File | What is in it |
|---|---|
| `summary.md` | the laptop and versions, FPS (mean, median, 5th percentile, 1% low), freezes, start-up time, CPU, memory, CPU temperature, and per stage the mean, median, 95th percentile and maximum time |
| `table.tex` | the rows of the timing table of the report, ready to paste |
| `frames.csv` | one row per shown frame: the time of every stage, the state, the number of people, faces and visible players |
| `resources.csv` | once per second: CPU use, memory, CPU temperature |

The statistics use the frames of the **Playing** state (normal gameplay). The first 25 frames (1 s) are left out:
OpenVINO prepares the models then, and that time is already in the start-up time.

## What is measured

| Stage | What | Report row |
|---|---|---|
| `capture` | decode the camera image, mirror it, make the 1280x720 copy (camera thread) | Capture |
| `t1_faces` | face detection + smoothing (face thread, at the same time as T2) | Face tracking (T1) |
| `t1_effects` | big heads, crown, player labels | Face tracking (T1) |
| `t2_pose` | YOLO pose estimation | Pose estimation (T2) |
| `t3_identity` | the tracker, and giving the faces to the players | Identity tracking (T3) |
| `t4_logic` | the state's `update()`: grading, gestures, star, state changes | Interaction logic (T4) |
| `segmentation` | person segmentation (segmentation thread, at the same time as T2) | Scene rendering (T5) |
| `t5_render` | the stage, the edge glow and the screen of the state | Scene rendering (T5) |
| `display` | giving the image to the window | Scene rendering (T5) |
| `frame_time` | the time between two shown frames (1000 / FPS) | end-to-end frame rate |
| `latency` | from the moment the camera image arrives to the moment it is shown | end-to-end latency |

Good to know when you write the report:

- `t1_faces` and `segmentation` run **at the same time** as `t2_pose`, so the stages do not add up to the frame time.
  `wait_threads` shows how long the game had to wait for those two threads.
- The game is limited to 25 FPS (`MAX_FPS`), so the frame rate cannot be higher than 25.
- `latency` starts when the image arrives in the program. The camera itself and the screen add some more time.
  To measure that too (glass to glass): film a stopwatch on the screen and the real stopwatch together with a phone.

## Suggested test plan (from the report template)

| Test | How | Gives |
|---|---|---|
| Normal gameplay | 2 players, one full song, 3 times: `--measure` | the timing table, FPS (mean and low), latency, CPU, memory |
| Lighting | the same, in a bright and in a dim room | influence of lighting on FPS, faces per frame, players visible |
| Start-up time | start the game 3 times: `--measure --duration 10` | "game ready" and "first image" times |
| Long run | 10 minutes, with `tools/crash_log.py` in a second terminal | freezes, memory over time, temperature, crashes |
| Fast motion | 2 players move fast | `players_visible` and `faces` per frame: how often tracking is lost |

Write down the conditions of every run: room, light, distance to the camera, number of people.
Keep heavy runs short on a hot laptop: the laptop crashed before under long full load.

## Replaying a recording (repeatable, with real players)

A `webcam.mp4` of `--record` can be played into the game instead of the webcam:

```
python main.py --video recordings/<take>/webcam.mp4 --no-mirror --windowed --measure
```

(`--no-mirror`: the recording is mirrored already.) The game does all the work again, with the same two players, so
the **times, FPS and detection rates** are valid. The **ratings and chains are not**: the song starts a fraction of a
second earlier or later than in the live take, and with 0.6 s moves that is enough to be out of step. Also, a
"play again" in the results works in a replay (it is off while recording), so a second song can start.
The numbers of 29 Oct 2026 are in `docs/report_measurements.tex`.

## Live test for Task 4 (recognition rate and false triggers)

Accuracy needs people who do an action **on purpose**; the event log (`events.csv`, and the table in `summary.md`)
counts what the game recognised. Record and measure in one go:

```
python main.py --measure --record --duration 300
```

| Test | How (2 players) | Write down |
|---|---|---|
| Browse | 10 x left hand up, 10 x right hand up (lower it in between) | recognised: count `browse_left` / `browse_right` |
| Choose / ready | 10 x both hands up in the song selection and registration, also both players at the same time | `choose_song`, `ready` per player |
| False triggers | 60 s standing and talking in the song selection, and normal dancing in the song | every event nobody meant |
| Star | play 3 songs, try to grab every star | `star_grab` / `star_appear`, `after` = how fast |
| Animated moves | follow the figure; once on purpose stand still | `chain_step`: `correct` and `flow` |

Recognition rate = recognised / intended; false triggers = events nobody meant (watch the video to check).
