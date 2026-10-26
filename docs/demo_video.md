# Recording the demo video (peer review, at most 60 s)

```
python main.py --record
```

(or `dist/JustDance/JustDance --record`). Play as normal; close the game with q or ESC. Then the game makes
three videos in `recordings/<date_time>/` (next to the program for the executable), all H.264 MP4:

| File | What |
|---|---|
| `webcam.mp4` | the webcam image (mirrored, like on the screen) |
| `game.mp4` | the game, with the music of the song |
| `side_by_side.mp4` | webcam and game next to each other (1920 x 540), with the music: ready to upload |

The video plays at real speed and the music is in sync. Recording does not make the game slower
(measured: still 25 FPS). The song-selection previews are not in the sound. It needs `ffmpeg` (installed on
Ubuntu with `sudo apt install ffmpeg`); without it the raw videos (`.avi`) stay in the folder.

## What the 55 seconds of Rasputin show

Every song stops after 55 s (`SONG_MAX_TIME` in `config.py`, the music fades out in the last second), so the
video fits in 60 s. Rasputin is made so that its first 55 s show everything:

| Time in the song | What happens | Task |
|---|---|---|
| 0 - 11 s | normal moves, a duet (5 s) and a gold move (7.6 s) | T2, T4 |
| 11 s | animated move **Conductor**: follow the moving figure, chain bonus | T2 (motion), T4 |
| 18 s | **bonus star 1**: the first hand that touches it gets +100 | T4 |
| 23 s | animated move **Kazachok** | T2, T4 |
| 25 - 30 s | the **debug drawings** turn on by themselves: skeletons, boxes, FPS (`DEBUG_SHOW_AT`) | T2, T3 |
| 34 s | animated move **Wave** | T2, T4 |
| about 36 - 41 s | **bonus star 2** (18 s after star 1 is gone) | T4 |
| 45 s | animated move **Disco** | T2, T4 |
| 55 s | the song stops: the results (ranking, winner) | T4, T5 |

All the time: big heads, crown and player labels (T1), identity colours and the score bar (T3), the stage
instead of the room (T5).

**Watch the total time:** the song selection, registration and the 3 s countdown also take time. With 55 s of
song there are only 5 s left for them in a 60 s video. Either be quick (choose the song and register within a
few seconds), or start the recording just before the song and show the registration some other way, or lower
`SONG_MAX_TIME` (for example to 45).
