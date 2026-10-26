# Recording the demo video (peer review, at most 60 s)

```
python main.py --record
```

(or `dist/JustDance/JustDance --record`). Play as normal; the game stops by itself after 60 s (or close it
with q or ESC). Then the game makes
three videos in `recordings/<date_time>/` (next to the program for the executable), all H.264 MP4:

| File | What |
|---|---|
| `webcam.mp4` | the webcam image (mirrored, like on the screen) |
| `game.mp4` | the game, with the music of the song |
| `side_by_side.mp4` | webcam and game next to each other (1920 x 540), with the music: ready to upload |

The video plays at real speed and the music is in sync. Recording does not make the game slower
(measured: still 25 FPS). The song-selection previews are not in the sound. It needs `ffmpeg` (installed on
Ubuntu with `sudo apt install ffmpeg`); without it the raw videos (`.avi`) stay in the folder.

## The 60 seconds of the video

The song stops after 30 s of song (`SONG_MAX_TIME`), or earlier when the video would get longer than 60 s:
then it stops 10 s before the 60 s (`RECORD_RESULTS_TIME`), however long the start took. The music fades out in
the last second. Then come the **results with the debug drawings** (skeletons, player boxes, FPS:
`DEBUG_IN_RESULTS`), and `--record` stops by itself **10 s after the results appear** (at the latest at 60 s, or
give `--duration`). So the last 10 s of the video are always the results, and the video is never longer than 60 s.
While recording, the results cannot be skipped (and the "Raise both hands" hint is not shown).

| Part of the video | About |
|---|---|
| song selection (Rasputin is selected first: raise both hands 0.5 s) | a few seconds |
| registration (both players raise both hands 0.5 s) | a few seconds |
| countdown | 2 s |
| the song | up to 30 s (less when the start took more than 20 s) |
| results with the debug drawings | the last 10 s |

So the start (song selection, registration, countdown) may take **at most 20 s** to see the whole song.

## What the 30 seconds of Rasputin show

| Time in the song | What happens | Task |
|---|---|---|
| 2.5 s | a duet | T4 |
| 3.8 s | animated move **Conductor**: follow the moving figure, chain bonus | T2 (motion), T4 |
| 6 s | **bonus star 1** (5 s): the first hand that touches it gets +100 | T4 |
| 6.3 s | a gold move | T4 |
| 11.3 s | animated move **Kazachok** | T2, T4 |
| 18.9 s | animated move **Wave** | T2, T4 |
| 21 s | **bonus star 2** (5 s) | T4 |
| 26.5 s | animated move **Disco** | T2, T4 |
| 30 s | the song stops: the results (scores, winner) with the debug drawings | T2, T3, T4 |

In between: more normal moves, 3 gold moves and 4 duets in total. The stars come at fixed times
(`STAR_TIMES`), between the animated moves. All the time: big heads, crown and player labels (T1), identity
colours and the score bar (T3), the stage instead of the room (T5).

For the full songs again (not for the demo): `SONG_MAX_TIME = None`, `DEBUG_IN_RESULTS = False`,
`STAR_TIMES = None` (with `STAR_INTERVAL = 20.0`, `STAR_JITTER = 5.0`), and if you like `HOLD_TIME = 1.0` and
`COUNTDOWN_TIME = 3.0`.
