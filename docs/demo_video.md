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

## What the 40 seconds of Rasputin show

`--record` stops by itself after **60 s** (or give `--duration`). Every song stops after **40 s**
(`SONG_MAX_TIME` in `config.py`, the music fades out in the last second), and the steps before the song are
short (hold the hands up 0.5 s, a 2 s countdown), so the whole demo fits in 60 s:

| Part of the video | About |
|---|---|
| song selection (Rasputin is selected first: raise both hands 0.5 s) | 3-5 s |
| registration (both players raise both hands 0.5 s) | 3-5 s |
| countdown | 2 s |
| the song | 40 s |
| results | the rest |

Rasputin is made so that its first 40 s show everything:

| Time in the song | What happens | Task |
|---|---|---|
| 3 - 8 s | the **debug drawings** turn on by themselves: skeletons, boxes, FPS (`DEBUG_SHOW_AT`), during a duet, a gold move and the start of the Conductor | T2, T3 |
| 7.6 s | animated move **Conductor**: follow the moving figure, chain bonus | T2 (motion), T4 |
| 12 s | **bonus star 1**: the first hand that touches it gets +100 | T4 |
| 18.9 s | animated move **Kazachok** | T2, T4 |
| 25 - 29 s | **bonus star 2** (12 s after star 1 is gone) | T4 |
| 34.0 s | animated move **Wave** | T2, T4 |
| 37.8 s | animated move **Disco** | T2, T4 |
| 40 s | the song stops: the results (scores, winner) | T4, T5 |

In between: normal moves, 4 gold moves and 5 duets. All the time: big heads, crown and player labels (T1),
identity colours and the score bar (T3), the stage instead of the room (T5).

For the full songs again (not for the demo): `SONG_MAX_TIME = None`, `DEBUG_SHOW_AT = None`, and if you like
`HOLD_TIME = 1.0`, `COUNTDOWN_TIME = 3.0` and a random star (`STAR_INTERVAL = 20.0`, `STAR_JITTER = 5.0`).
