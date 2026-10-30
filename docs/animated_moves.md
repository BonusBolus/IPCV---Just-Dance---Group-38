# Animated moves

Most moves are one pose to hold. An **animated move** is a short movement: a list of poses that the players
follow step by step, for example the Kazachok: left leg out, right leg out, left, right.

## In the song file

```json
{ "beat": 192, "poses": ["kazachok_left", "kazachok_right", "kazachok_left", "kazachok_right"], "name": "Kazachok" }
```

- Every pose is one **step** of one **music beat** (`step_beats`, default 1). So the steps stay on the beat of the
  music, also when the other moves use the faster `move_bpm`.
- When the song is loaded, the animated move becomes one move per step (`song.animated_move()`). Each step knows
  the pose before and after it.
- The poses must be in `poses.json` and must be solo poses (no duets).
- Rasputin uses four different animated moves: Conductor, Kazachok, Wave and Disco.

## On the screen

- **Magenta** is the colour of animated moves: on the timeline every step is a pictogram, so you see the whole
  movement coming, like a film strip.
- The NOW card shows the step ("NOW! 2/4") and the name of the move. The figure **moves**: it holds the pose for the
  first half of the step, then moves smoothly to the next pose (`pose_figure.animated_pose()`). So the players see
  where to go a moment before they have to be there.
- "Chain x3" above a player shows how many steps in a row they did right.

## Points: the chain

Every step is graded like a normal move (the same PoseGrader, Perfect / Good / OK / Miss, points per step).
A step is **correct** when:

1. its score is at least `CHAIN_MIN_SCORE` (65, Good), and
2. from the second step on: the player moved **the right way** to get there (the motion check below).

Every correct step in a row makes the chain longer and gives extra points: the 2nd correct step in a row
+20 (`CHAIN_BONUS`), the 3rd +40, the 4th +60. A wrong step ends the chain; the next animated move starts a new one.

## The motion check (Task 2 motion signal)

The poses only tell *where* the arms and legs are. The motion check also looks at *how the player moves*:

- `game_logic/motion.py` makes the **motion per player**: how far every body point moved since the last frame, in
  shoulder widths (so it does not depend on the distance to the camera), smoothed like the motion of Task 2.
- It uses the keypoints of the identity tracker (Task 3), so the motion always belongs to the right player. The
  PoseEstimator's own motion is per detected person in YOLO's order, and that order can change between frames
  when two people are in the image.
- For every step after the first, the movement of the hands, elbows, knees and feet from half a step before the
  step until half a step into it is compared with the movement of the figure between the two poses
  (`motion.flow()`, the cosine of the two movements: 1 = the same direction, -1 = the opposite way).
  The chain only goes on when this is at least `FLOW_MIN` (0.5). A player who does not move at all gets 0.
- `FLOW_CHECK = False` in `config.py` switches the motion check off.

With `--debug` the game shows the chain and the last direction match ("flow") of every player.
