# Identity tracking (Task 3)

Every player keeps the same label ("Player 1", "Player 2") for the whole game, also when they move, cross each
other or leave the image for a moment. Players are recognised by the **colour of their shirt**. The code is in
[identity.py](../identity_tracking/identity.py).

Detection says **where** people are, identity tracking says **who** is who: the pose estimation (Task 2) finds the
bodies, the tracker gives them a label.

## How it works

```
keypoints of every person ──► box per person ──► shirt region ──► average colour + histogram
                                                                          │
                     remembered colour, histogram and position of each player ──► cost table
                                                                          │
                                                          Hungarian algorithm: best pairing
```

1. **`people_from_pose()`** puts a box around the keypoints of each person (YOLO, Task 2).
2. **`torso_box()`** finds the shirt: between the shoulders and the hips (middle 90%), so arms, trousers and
   background stay out. Hips not visible: the hips are estimated 1.3 shoulder widths below the shoulders. No
   shoulders: a fixed part of the box.
3. **`average_color()`** and **`color_histogram()`** describe the shirt (BGR).
4. **`IdentityTracker.update()`** computes the cost of every person–player pair:

   `cost = colour distance + 0.1 × position distance (px) + 100 × histogram distance`

   The histogram term (Bhattacharyya distance) is only used when the two players have similar colours (distance
   below 30), for example a striped and a grey shirt. The Hungarian algorithm
   (`scipy.optimize.linear_sum_assignment`) finds the best pairing overall, so two people never get the same label.
   A pair is only accepted when the colour distance is below 60; otherwise the person is a stranger ("?") and gets
   no effects.
5. The remembered colour and histogram are updated slowly (`0.9 × old + 0.1 × new`): they follow small lighting
   changes, but one bad frame does not spoil them.

The first two people the tracker sees after `reset()` (in the Registration) become Player 1 and Player 2.

## Leaving, crossing and similar shirts

- **Leave and re-enter.** A player who is not seen is not deleted. When they come back, anywhere in the image, their
  shirt colour gives them their label again. After 0.5 s without detection the game shows a "lost" warning.
- **Crossing.** The Hungarian algorithm keeps two players from getting the same label, and because the cost depends
  mostly on colour, the labels are right again as soon as they are apart.
- **Similar shirts** are the main weak point: then colour cannot tell the players apart. `colors_are_similar()`
  makes the game show "Please wear different colours".
- **Faces** (Task 1) are linked to a player when the face lies inside that player's box (the face closest to the
  nose keypoint). Only those faces get the head effects.

Limitations: the colours are compared in BGR, so shadows and lamps change them more than in HSV or Lab; there is no
prediction (Kalman filter) while a player is hidden; positions are in pixels only (no depth).

## Evaluation

Use [tests/eval_identity.py](../tests/eval_identity.py). It runs the pose estimation and the tracker of the game,
shows the measurements on screen (FPS, time per part, latency, accuracy, switches) and writes them to
`tests/results/<date>_<condition>/` (`frames.csv`, `trials.csv`, `summary.json`). Every run also adds one line
to `tests/results/summary.csv`, so different conditions end up in one table.

```
python tests/eval_identity.py --condition bright --subjects "A,B" --distance 2.5 --record tests/results/bright.mp4
python tests/eval_identity.py --video tests/results/bright.mp4 --condition bright   # the same test again
```

**Protocol.** Two players, different shirts, stand apart (the first frame with both players sets who is who:
check the OK labels, else press `k`). For each trial press the key, do the action, and press `e` when both stand
apart again:

| Key | Trial | What the players do |
|---|---|---|
| `c` | crossing | walk past each other and end on the other side (10x) |
| `o` | occlusion | one player stands half behind the other or an object for about 2 s (10x) |
| `r` | re-entry | one player walks out of the image and comes back on the same side (10x; other side: also press `s`) |

Other keys: `e` or `SPACE` ends the trial, `s` means the players swapped sides, `k` says the IDs shown now are
correct, `x` throws the current trial away, `p` pauses a video, and `q` or `ESC` stops and writes the results.
All options: `python tests/eval_identity.py --help`.

Repeat the runs for the impact factors: **lighting** (`--condition bright` / `dim`, the script also measures the
brightness), **distance** (`--distance` in metres; the player height in pixels is measured too), and **subjects**
(other pairs, similar shirts).

**What it measures.** Identity accuracy (% of the evaluated person-frames with the correct player), ID switches
(per trial and per minute), success rate (% of trials with both IDs correct at the end), recovery (% recovered and
the time from the end of the event until both IDs are correct), and the speed (pose, tracking and face time in ms,
FPS, latency from camera image to result). Only frames with exactly two people that do not overlap are evaluated:
then their side tells who is who. Spatial error in mm is not measured, because the game has no physical positions.
