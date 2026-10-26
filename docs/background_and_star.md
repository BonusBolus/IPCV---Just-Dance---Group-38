# The stage and the bonus star

## 1. The stage: the room is replaced

The camera sees the players **and the room**. In Just Dance the dancers stand on a colourful stage, so we cut
the players out of the camera image and put them on a stage.

**Finding the players** (`background/segmentation.py`)

- A second YOLO model, `yolo26n-seg` (segmentation), finds every person and gives each one a **mask**: a small
  image where 1 = person and 0 = room. It runs with OpenVINO on a copy of 320 px wide (`SEGMENT_SIZE`). 480 px gives slightly cleaner edges, but
  costs 8-11 ms more per frame: the segmentation and the pose model compete for the CPU, so both get slower.
- It runs in its **own thread**, at the same time as the pose model and the faces, so the game is hardly slower.
- **Only the players stay** (`PlayerMasker`). A mask is kept when its box overlaps a player's box (`BOX_OVERLAP`).
  Someone who walks past in the back is not a player, so they disappear into the stage.
- **A player must never disappear**: that is worse than an extra person on the stage. So:
  - a player the tracker does not see for a moment keeps their place for 1 s (`PLAYER_KEEP_TIME`);
  - other people are only removed while the players are fixed (countdown, playing, results). In the song
    selection and the registration everyone stays, because people still have to step in there;
  - when no player is known, everyone stays;
  - `REMOVE_BYSTANDERS = False` keeps everyone, always.
- YOLO makes the image a multiple of 32 px with a border; `remove_padding()` cuts that border off the masks,
  so they fit the image again.
- The edge is moved a little inside the person (no light rim of the room) and made soft (`MASK_SOFTNESS`).

**The stage** (`background/stage.py`)

- A gradient from deep blue-purple at the top to pink at the bottom, two soft light beams from the top and a
  glow on the floor. This part does not move, so it is made **once** for the image size.
- A few big, soft circles of light (bokeh) float slowly. Their place only depends on the time.
- `place_players()` mixes the two: where the mask is 1 the camera image, where it is 0 the stage (about 2.5 ms).

**Big heads:** the head effect (`head_filter.py`) also makes the head bigger **in the mask**. Without that, the
stage would cut the big head back to its old size.

Turn it off with `BACKGROUND = False` in `config.py`: then you see the room again, and the second model is not loaded.

## 2. The bonus star

Now and then a gold star floats across the screen. **The first player who touches it with a hand gets +100 points**
(`STAR_BONUS`, a Perfect move is also 100).

| Rule | Setting |
|---|---|
| A star comes about every 20 s, plus or minus 5 s, so it is a surprise | `STAR_INTERVAL`, `STAR_JITTER` |
| It floats from one side to the other in 5 s; the next one comes from the other side | `STAR_TIME` |
| It floats at **hand height**: between just above the heads and the shoulders, and gently up and down. It follows the players when they move | (from the players' boxes and shoulders) |
| A wrist closer than 58 px (at 720p) grabs it. When both players touch it, the closest hand wins | `STAR_REACH` |
| No star in the last 3 s of the song, and none when nobody is seen | `STAR_END_FREE` |
| After a grab: rays fly out and "+100" rises in the player's colour | `STAR_BURST_TIME` |

The logic is in `game_logic/bonus_star.py` (when, where, who), the drawing in `scene/star.py`. The Playing state
adds the points to the player's score, so the bonus also counts for the crown and the results.

**Careful:** both players may reach for the same star. Leave enough space between the players.
