# Identity tracking with average colour: how to approach it

Goal of task 3: every person on screen keeps the same label ("Player 1", "Player 2") for the
whole game, also when they move, cross each other or leave the frame for a moment.

First version: **recognise players by the colour of their shirt.**

## 1. The idea in one picture

```
frame ──► find people ──► boxes (who? unknown)
                             │
                             ▼
                  shirt region of each box ──► average colour ──► compare with the
                                                                  remembered colour
                                                                  of each player
                                                                        │
                                                                        ▼
                                                        closest colour = that player
```

Detection says **where** people are. Identity tracking says **who** is who. That split is
also in the assignment: task 2 finds the bodies, task 3 labels them.

## 2. Steps (do them in this order, test after each one)

Open [identity.py](../identity_tracking/identity.py) and run `python main.py`. The people come from
the pose estimation in [pose_tracking/pose_main.py](../pose_tracking/pose_main.py) (task 2):
`people_from_pose()` turns its keypoints into one box per person.

1. **`torso_box`**: shrink the person box to the shirt. Check it on screen: the thin white
   box should sit on your chest.
2. **`average_color`**: mean colour of the pixels in that box. The colour swatch next to the
   label should turn into your shirt colour once step 4 works. You can also `print` it.
3. **`color_distance`**: Euclidean distance between two (B, G, R) colours. Test it in a
   Python shell: black vs. white should be about 441, a colour vs. itself 0.
4. **`IdentityTracker.update`**: register new players, then match by smallest distance.
   Test: walk in alone (you become Player 1), let a friend walk in (Player 2), swap places.
   Do the labels swap too? They shouldn't.

## 3. Things to search / read about

| Topic | Why you need it |
|---|---|
| NumPy array slicing, `np.mean(axis=...)` | cut out the shirt region and average it |
| BGR vs. RGB in OpenCV | OpenCV stores colours as B, G, R (not R, G, B) |
| Euclidean distance, `np.linalg.norm` | how different two colours are |
| Colour spaces: **HSV** and **CIELAB** (`cv2.cvtColor`) | RGB mixes colour and brightness; a shadow changes the RGB values a lot. In HSV the hue barely changes with lighting, and in Lab distances match what humans see. Good first improvement. |
| Colour **histograms** (`cv2.calcHist`, `cv2.compareHist`) | the average of a striped shirt is a meaningless grey; a histogram keeps the distribution |
| Nearest-neighbour matching / assignment problem, **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) | prevents two people from getting the same label: it finds the best pairing overall |
| Exponential moving average | updating the remembered colour slowly (step 4 in `update`) |
| Pose estimation, YOLO-pose, COCO keypoints | how the pose estimation that provides your people works, in case the professor asks |

## 4. Pitfalls you will run into

- **Similar clothes.** Two people in black shirts cannot be told apart by colour alone. For the
  demo, ask players to wear different colours. Later, combine colour with position (step 5).
- **Background in the box.** The box around the keypoints also contains background. That's why you use only the shirt
  region, not the whole box.
- **Lighting.** Walking under a lamp changes the colour. Updating the colour slowly helps, and
  HSV or Lab helps more.
- **Pose estimation misses a frame.** Then a player has no box that frame. Don't delete the player:
  keep them in `self.players` and they get their label back when they are detected again.
  That is the "leave and re-enter" requirement.
- **Two boxes, same player.** With the simple nearest-colour rule both boxes can pick the same
  player. The Hungarian algorithm fixes this.
- **`max_distance`.** Too small and players become "?" under slightly different light; too
  large and a stranger gets a player's label. Print the distances while testing to choose it.

## 5. Next versions (after the simple one works)

1. **Colour + position.** People move only a little between two frames (1/30 s). A cost like
   `colour_distance + weight * position_distance` makes crossings much more stable.
2. **Hungarian assignment** instead of "each box picks its closest player".
3. **Boxes from the pose estimation (task 2).** Done: the shirt region is taken between the
   shoulders and hips (`torso_box`), with a fixed part of the box as fallback.
4. **Prediction (Kalman filter).** Predict where each player will be in the next frame, which
   helps while players cross or are hidden for a moment.
5. **Physical position.** The assignment asks for player positions in real units for task 4.
   With the pinhole camera model, distance ≈ focal length × real shoulder width (~0.4 m) ÷
   shoulder width in pixels.

## 6. How to evaluate it (needed for the report)

- **Identity switches:** walk past each other 10 times, and count how often the labels swap.
- **Re-entry:** leave the frame and come back 10 times, and count how often you get your old
  label back.
- **Lighting / clothing:** repeat with similar shirts, and in darker light, and report when it
  fails.
- **Speed:** time `update()` with `time.perf_counter()`. It should take well below 1 ms per
  frame; the pose estimation costs more.
- Record the test once (e.g. with your phone or OpenCV's `VideoWriter`) and replay it with
  `python main.py --video test.mp4`, so every version is tested on the same footage.
