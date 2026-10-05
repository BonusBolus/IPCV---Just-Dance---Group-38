# Scene changes

The game has 4 screens: Registration, Countdown, Playing and Results (see [game_structure.md](game_structure.md)).
The `scene/` folder draws them. The rule: **the states decide what to show, the scene decides how it looks.**

Every part of the screen has **its own file** now, so you only need to read the file you want to change.

**Nothing old was removed or changed.** `scene.py` is the original, with one new method. `functions.py` is the original, with new helpers at the end.
We tested it: everything draws **the same pixels** as before.

## The files

| File | What is in it | Used by |
|---|---|---|
| `scene.py` | **The original** `Scene` class: title, score bar, current pose, `draw_player()`. **New:** `draw_ratings()` | Countdown, Playing |
| `functions.py` | **The original** text helpers. **New at the end:** colours and small building blocks | all files below |
| `pose_figure.py` | **New.** Draws a pose as a stick figure | `move_cards.py` |
| `move_cards.py` | **New.** The NOW and NEXT cards | Countdown, Playing |
| `registration_screen.py` | **New.** Everything on the Registration screen | Registration |
| `results_screen.py` | **New.** Everything on the Results screen | Results |

How the files use each other:

```
registration_screen.py ─┐
results_screen.py ──────┤
move_cards.py ──────────┼──> functions.py   (colours, text, bars, banners)
scene.py ───────────────┘
move_cards.py ─────────────> pose_figure.py (stick figure)
```

## Where do I change ...?

| I want to change ... | File |
|---|---|
| a colour (gold, green, the rating colours) | `functions.py`, at the top of the new part |
| the text outline, the progress bars, the red warning banner | `functions.py` |
| how a pose looks (lines, head, thickness) | `pose_figure.py` |
| the size or place of the NOW / NEXT cards, the pulse, the flash | `move_cards.py` |
| the player labels, "READY", "Waiting for Player 2..." | `registration_screen.py` |
| the scores, the winner text, "Raise both hands to play again" | `results_screen.py` |
| the title, the score bar, where "Perfect" / "Good" is shown | `scene.py` |

---

## `scene.py`: one new method

Only this was added (and `RATING_COLORS, draw_text` in the import line):

```python
def draw_ratings(self, frame, ratings):
    """"Perfect", "Good", ... under the score of each player. ratings: {player_id: "Perfect"}"""
    width = frame.shape[1]
    for player_id, rating in ratings.items():
        x = width // 2 + (-1 if player_id == 1 else 1) * width // 8
        draw_text(frame, rating, (x, self.score_bottom + 35), 1.0, RATING_COLORS[rating])
    return frame
```

It is in `scene.py` because the ratings are placed under the score bar (`self.score_bottom`).
`render(frame, players)` without a pose is used for the title and score bar during Countdown and Playing.

## `functions.py`: building blocks

The old functions (`mix_colors`, `put_text_left`, `put_text_right`, `put_text_center`, `scale_for_height`) are the same. New at the end:

| Name | What it does |
|---|---|
| `WHITE`, `GREY`, `GREEN`, `GOLD`, `RATING_COLORS` | Colours in OpenCV order (**BGR**) |
| `player_color(player)` | The colour of a player in BGR. The players dict has **RGB**, so always use this. |
| `draw_text(frame, text, center, size, color)` | Centred text with a black outline, so it is readable on every camera image. It draws the text twice: first thick and black, then in colour. |
| `draw_big_text(frame, text)` | Big text in the middle: "3", "2", "1", "Dance!" |
| `draw_bar(frame, top_left, size, fraction, color)` | A progress bar. `fraction` goes from 0 to 1. |
| `draw_banner(frame, text, color, row)` | A coloured bar with white text, for warnings. `row=1` puts it under the first one. |

`functions.py` now also imports `config`, for `config.PLAYER_COLOR` in `player_color()`.

## `pose_figure.py`: the stick figure

```python
draw_pose(frame, pose, center_x, nose_y, scale, color, thickness=1)
```

- `pose`: a pose from `poses.json`, like `{"nose": (0, 0), "left_wrist": (-1.3, -0.3), ...}`
- `center_x`, `nose_y`: where the nose goes on the screen (pixels)
- `scale`: pixels per pose unit
- `BONES` is the list of lines between the points. It is the same as the `combinations` list in `scene._add_current_pose()`.

## `move_cards.py`: NOW and NEXT

```python
draw_move_cards(frame, now_move, now_progress, next_move, next_progress, beat_phase, flash)
```

- **NOW** (big card): the move that counts now. The bar gets empty while the player holds the pose. A short flash shows that a new move started. Between moves: "Get ready".
- **NEXT** (small card): the coming move, visible from the countdown on. The bar fills up until the move starts. The figure is 8% bigger on the beat: that is the pulse.
- **Gold moves**: gold border and the label "GOLD".
- All numbers (`now_progress`, `next_progress`, `beat_phase`) come from `song.py`. This file only draws them.
- `draw_card()` draws one card. The card sizes and places are at the top of `draw_move_cards()`.

## `registration_screen.py`

```python
draw_registration(frame, players, ready, progress, colors_similar)
```

- Above each player: "Player 1" in the player colour, and the progress bar of the hands-up gesture. When the player is ready: "READY".
- A player that is not found yet: "Waiting for Player 2..." at the bottom.
- At the top: "Raise both hands when you are ready".
- `colors_similar=True`: the banner "Please wear different colours".
- `player["box"]` is `(x, y, w, h)` from the IdentityTracker.

## `results_screen.py`

```python
draw_results(frame, players, winner, progress, show_hint)
```

- The image gets darker, and the scores and the winner are shown in the middle.
- `winner` is the player dict of the winner, or `None` for a draw. The Results **state** decides who won; this file only shows it.
- `show_hint=True` (after 3 s): "Raise both hands to play again" and the progress bar.
- `draw_lines()` draws the lines of text. Each line is `(text, color, size)`.

---

## How the states call the scene

| State | Calls |
|---|---|
| Registration | `draw_registration()` |
| Countdown | `game.scene.render()`, `draw_move_cards()`, `draw_big_text()` |
| Playing | `game.scene.render()`, `draw_move_cards()`, `game.scene.draw_ratings()`, `draw_banner()`, `draw_big_text()` |
| Results | `draw_results()` |

Want to change how something looks? Only change the file in `scene/`. The game rules in `game_logic/states/` stay the same.
