# Can we play with 5 players?

Short answer: **yes, the computer can do it easily. The hard parts are the room, the camera and the screen design.**
This page was a study first. **It is built now** (branch `five-players`): see "7. What we built" at the end.

## 1. Speed: no problem

Measured on the laptop (photos with several people, and 5 fake players):

| Part | 2 players | 5 players |
|---|---|---|
| YOLO (finds all people in one go) | 8.9 ms | 8.3 ms (the same) |
| Identity tracker | 0.2 ms | 0.5 ms |
| Grading | 0.06 ms | 0.14 ms |
| Player labels | 0.11 ms | 0.14 ms |

More players cost less than 1 ms per frame. The game stays at about 22 FPS.

## 2. The room and the camera: the real limit

5 people next to each other need a lot of space, and the camera must see all of them.

| | Moves with arms close to the body | Moves with arms out (T-pose, star, ...) |
|---|---|---|
| Space per player | about 0.8 m | about 1.4 m (an arm span is about 1.7 m) |
| Width for 5 players | about 4 m | about 7 m |
| Distance to the webcam (about 75° wide) | about 2.7 m | about 4.5 m |

What that means for the game:

- **Bodies are fine.** At 4.5 m a person is still about half the image high; YOLO finds them easily.
- **Faces get small.** At 4.5 m a face is about 30 px wide at 1280x720, and only about 15 px in the 640 px copy that face
  tracking uses (`FACE_DETECT_WIDTH`). YuNet becomes unreliable below about 12-15 px, so the big heads and the crown can
  flicker. With 5 players face tracking should use the full image (slower: about 3x, see the speed fix).
- **Arms of neighbours overlap** in moves with arms out. Then YOLO can give an arm to the wrong person: wrong scores.
- **Most rooms are too small** for 5 players with arms out (7 m wide, 5 m deep).

## 3. Telling the players apart

The identity tracker uses the shirt colour and the position of each player.

- With 5 players we need **5 clearly different shirt colours**. The "Please wear different colours" warning already exists.
- More players means more chance that two of them cross or stand close together. Then labels can swap, and points go to
  the wrong player. This is the biggest risk for fair scores.
- `IdentityTracker(max_players=...)` already works for any number. It only knows colours for Player 1 and 2
  (`DEFAULT_COLORS`), so we need 3 more.

## 4. What must change in the code

**Easy (works for any number with small changes):**

- `PLAYER_IDS = (1, 2)` in `states/common.py` becomes `range(1, MAX_PLAYERS + 1)` (a new setting in `config.py`).
- `IdentityTracker(max_players=MAX_PLAYERS)` and 3 more player colours.
- Registration: start when **everyone who is registered** is ready, and at least 1 player (not "exactly 2").
- `crown_holder()`: the player with the highest score (now it compares exactly 2 players).
- Playing: most loops already go over `PLAYER_IDS`.

**New design needed (the screen):**

- **Score bar**: now a tug-of-war between 2 players (left and right). For 5: a **row of 5 score boxes** at the top,
  in the order the players stand (left to right), each with the player colour, the score and the rating under it.
- **Results**: a ranking (1st to 5th) instead of "Player 1 wins".
- **Title and edge glow**: now in 2 player colours. For 5: one colour, or all 5 colours.

**Rules needed (duets):**

Duet poses have a part for the LEFT person and a part for the RIGHT person, and a check that their hands touch.
With 5 players we must choose a rule, for example:

- **A. Pairs of neighbours**: from left to right, players 1+2 and 3+4 do the duet together; the 5th does the left part
  alone (no duet bonus).
- **B. Everyone does part 1**: duets become normal moves when there are more than 2 players (no duet bonus).
- **C. No duets** in songs with more than 2 players.

## 5. How much work?

| Part | Work |
|---|---|
| Setting `MAX_PLAYERS`, tracker, colours, registration, crown | small |
| Score row for up to 5 players + ratings under each box | medium |
| Results with a ranking | small |
| Duet rule (A, B or C) | small (B, C) to medium (A) |
| Face tracking on the full image when there are many players | small |
| Testing with 5 real people in a big room | the most work: we can only check the room and the camera there |

## 6. Advice

1. Make the number of players a **setting** (`MAX_PLAYERS`, from 1 to 5), and keep 2 as the normal game. Then nothing
   changes for 2 players.
2. Build the **score row** and the **ranking**, and choose duet rule **B** first (simple and fair).
3. **Test with 3 players first.** That already shows most problems (space, colours, crossing) and fits in most rooms.
   Go to 5 when 3 works well.

## 7. What we built

We chose duet rule **A** and the score row at the top. The game now works for 1 to 5 players:

- `config.py`: `MAX_PLAYERS = 5`, `MIN_PLAYERS = 1`, `DROP_TIME = 3.0`. 5 player colours: blue, red, green, yellow, purple.
- **Registration** waits for everyone who registered. Someone who is gone for 3 s (for example a person who walked past)
  is not waited for.
- **Score**: exactly 2 players keep the score bar. 1 or 3-5 players get a row of score boxes at the top, in the order they
  stand, with the rating under each box.
- **Duets**: pairs of neighbours, from left to right (1st+2nd, 3rd+4th). Each pair gets the duet bonus on its own.
  A 5th player does the left part alone and gets no bonus. The pairs follow where the players stand (only updated
  while every player is seen).
- **Results**: a ranking; the crown goes to the player with the highest score (no crown when it is shared).
- Tests: `tests/test_five_players.py` (registration, duet pairs with the real grader, ranking, score boxes).

Not tested yet with real people. Advice 3 still holds: try 3 players first.
