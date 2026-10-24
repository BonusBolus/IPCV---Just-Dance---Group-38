# Face tracking and head effects

Every frame, the game finds the faces in the camera image and uses them for three effects:
**big heads**, a **gold crown** for the player who is ahead, and the **player label** ("Player 1", "Player 2") on
the forehead (on the crown, when that player has it).
This file explains how that works and where to change it.

## The files

| File | What is in it |
|---|---|
| `face_tracking/face_tracking.py` | Finds the faces (`get_face_properties()`) and smooths them over time (`smooth_face_properties()`) |
| `face_tracking/head_filter.py` | `enlarge_heads()`: makes every head bigger |
| `face_tracking/game_effects.py` | The crown (`add_crown()`), the player label (`add_player_label()`), which face belongs to which player (`faces_for_players()`), and the places on the head (`head_point()`, `crown_band()`, `crown_top()`) |
| `face_tracking/face_detection_yunet_2023mar.onnx` | The YuNet face detection model. It must be in this folder, or the game stops at start-up with a clear error. |

## Order in one frame

`Game.step()` in `game_logic/game.py`:

```
camera image
   ├──► face thread:  get_face_properties() ──► smooth_face_properties() ──► faces
   └──► main thread:  pose (YOLO) ──► IdentityTracker ──► players ──► state.update()

image = copy of the camera image
game.heads = faces_for_players(players, faces)    # 1. which face belongs to which player
player_faces = faces in game.heads that are not None
enlarge_heads(image, player_faces)                # 2. big heads, only for players
add_crown(image, [face of crown_holder()])        # 3. crown, only for the player who leads by 10 %
add_player_label(image, "Player N", face, colour) # 4. label on the crown, for every player with a face
scene.draw_edge_effects(...)                      # glow + sparkles at the screen edges
state.draw(game, image)                           # the screen of the current state (uses game.heads)
```

- Face tracking runs in a **second thread**, at the same time as YOLO. Both only read the camera image and spend most
  time outside Python, so this saves about 7 ms per frame. `faces_future.result()` waits for it before drawing.
- The head effects are drawn in **every state**, before the state draws its own screen. So text of a state (scores,
  banners) is always on top of the heads.

## 1. Finding faces: `face_tracking.py`

`get_face_properties(frame)` uses **YuNet** (`cv2.FaceDetectorYN`, needs OpenCV 4.5.4 or newer). The input size is set to
the size of each frame, so faces far away are not made smaller first. Faces with a score below `SCORE_THRESHOLD = 0.6`
are skipped. Each face is a dict:

```python
{
    "location":    (x, y),            # centre of the face box, pixels
    "size":        (width, height),   # size of the face box, pixels
    "orientation": {"roll": angle},   # tilt of the head in degrees, from the line between the eyes
}
```

`roll` is about **360 for a straight head** (not 0). The effects use it with `sin` / `cos` and rotations, so 360 and 0
give the same result.

`smooth_face_properties(faces, alpha=0.5, max_missed_frames=5)` makes the effects calm:

- Each face is paired with the **nearest face of the previous frame**, but only when it is closer than 1.5 × the face
  size. A new player never gets blended with another player.
- Location, size and roll become `alpha * new + (1 - alpha) * old`. Lower `alpha` = calmer, but slower to follow.
- A face that is not found for a moment (blur, a hand in front of it) **stays for up to 5 frames** at its last place.
  So the crown and the label do not blink.
- The smoothing keeps its memory in module variables (`_previous_faces`), so there is one face tracker for the whole
  game.

## 2. Big heads: `head_filter.py`

`enlarge_heads(image, faces, enlargement)` magnifies an oval around each face, with `cv2.remap`. The edge of the oval
fades out (feathering), so you do not see a hard border. The amount is `HEAD_ENLARGEMENT` in `config.py` (`1.35`);
`1.0` turns the effect off.

**Sharp big heads:** the webcam is read at 1920x1080 (`CAMERA_SIZE`), and the game works on a 1280x720 copy
(`GAME_SIZE`, see `camera.py`). `enlarge_heads(..., detail=frame.detail)` takes the enlarged head from the
1080p image, which has 1.5x more pixels. So a big head is sharp, and not a blown-up 720p face. The place and size
are the same as before. A webcam that cannot do 1080p just gives its own size, and then the effect works as before.

## 3. The crown: `add_crown()`

**Who gets the crown:** only the player whose score is **at least 10 % higher** than the other player's score
(`crown_holder()` in `game_logic/game.py`, the 10 % is `CROWN_LEAD` in `config.py`). So:

- both scores 0 (Registration, Countdown, the start of the song): **nobody** has the crown;
- scores close together (for example 500 and 460): **nobody** has the crown;
- player 1 has 0 points and player 2 has some: player 2 has the crown.

The crown goes to the leader's face (see section 4), sized from the face and tilted with the head. Its shape is set by constants at
the top of `game_effects.py`:

| Constant | Meaning | Value |
|---|---|---|
| `CROWN_WIDTH` | Crown width, × face width | 1.18 |
| `CROWN_HEIGHT` | Crown height, × face height | 0.52 |
| `CROWN_LIFT` | Crown centre above the face centre, × face height | 0.69 |
| `BAND_TOP`, `BAND_BOTTOM` | The **band**: the big yellow strip under the points, in crown heights from the crown centre (down is positive). `BAND_TOP` is just below the jewels. | 0.13, 0.5 |
| `TIP_TOP` | The highest point of the crown, in crown heights | -0.55 |

```
        /\    /\    /\          <- TIP_TOP
       /  \  /  \  /  \
      | ●    ●    ● |           <- jewels
      |  Player 1   |           <- band (BAND_TOP .. BAND_BOTTOM): the label goes here
      +-------------+
         ( face )
```

Change the crown shape only with these constants: the label and the progress bar use the same numbers, so they move
with the crown.

## 4. The player label: `faces_for_players()` + `add_player_label()`

The label used to hang above the **box of the body** from the IdentityTracker. That box changes size with arms and
legs, so the label floated around. Now it hangs on the **face**, which is much calmer.

**Which face belongs to which player** (`faces_for_players(players, faces)`):

1. A face can belong to a player only when its centre is **inside the player's box**.
2. When more faces fit, the one **closest to the player's nose** wins (top of the box if the nose is not seen).
3. Each face goes to **at most one** player.

It returns `{1: face or None, 2: face or None}`. `Game` keeps it in `game.heads`, so the states can use it too.
Faces that belong to nobody get **no effects at all**: no big head, no crown, no label. This filters out:

- **bystanders**: a third person, or someone in the background, is not a player;
- **false detections**: YuNet sometimes sees a "face" in a pattern, a shadow or an object. Such a face is almost never
  inside a player's box, near the nose that YOLO found. Without this filter, the smoothing even kept such a face on
  screen for up to 5 frames.

**The label** (`add_player_label(frame, text, face, colour)`):

- It is written in the centre of the crown band (`crown_band()`), in the **player colour** with a black outline, and
  **tilted with the head**.
- The text size is chosen so it fits the band (80 % of the band height, 90 % of the crown width), with a minimum size
  so it stays readable for faces far away.
- When the crown is drawn, the band is the background of the label. Without a crown, the label is on the **forehead**,
  at the same place.
- The text is small, so it is drawn 4× bigger (`SUPERSAMPLE`) and then shrunk. Without this, thin coloured text with an
  outline is hard to read.

## 5. Registration: progress bar above the crown

`scene/registration_screen.py` uses `game.heads`: the hands-up progress bar (or "READY") is drawn **just above the tips
of the crown** (`crown_top()`), as wide as the crown. When no face is found for a player, the label and the bar fall
back to the top of the player's box, so the player still sees them.

## Where do I change ...?

| I want to change ... | Where |
|---|---|
| how big the heads are | `HEAD_ENLARGEMENT` in `config.py` |
| how calm / fast the effects follow the head | `alpha` and `max_missed_frames` in `smooth_face_properties()` |
| which faces are found (small or unclear faces) | `SCORE_THRESHOLD` in `face_tracking.py` |
| the crown size, place or shape | the `CROWN_*`, `BAND_*`, `TIP_TOP` constants in `game_effects.py` |
| the crown colours | `add_crown()` |
| how far ahead a player must be for the crown | `CROWN_LEAD` in `config.py` |
| who gets the crown (the rule itself) | `crown_holder()` in `game_logic/game.py` |
| the label text size or outline | `add_player_label()` |
| the progress bar / "READY" above the crown | `scene/registration_screen.py` |
| which player gets which face | `faces_for_players()` |
