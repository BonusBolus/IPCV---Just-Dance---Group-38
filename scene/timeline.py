"""The move timeline at the bottom of the screen (Countdown and Playing).

  | [pose] ====|      [pose]=====        [pose]=====      |  [ NOW card ]
     hit line    <- the coming moves slide to the left

When a pictogram reaches the hit line, that is the moment to be in the pose. The bar behind it is how
long the pose must be held: while you hold it, the pictogram stays at the line and its bar gets shorter.
Gold moves are gold, duets cyan with a figure in each player's colour. The very next move always waits
at the right end of the track, so it is visible from the countdown on.
"""
import cv2

from scene.functions import DUET, GOLD, GREEN, GREY, WHITE
from scene.move_cards import draw_duet_poses
from scene.pose_figure import draw_pose

LOOKAHEAD = 4.0    # seconds of coming moves on the track


def timeline_layout(width, height):
    """(left, top, right, bottom) of the track and the x of the hit line. The NOW card is at the
    right edge (move_cards.py, show_next=False), so the track ends just before it."""
    u = height / 720
    bottom = height - 20
    top = bottom - int(120 * u)
    left, right = int(20 * u), width - 270
    return (left, top, right, bottom), left + int(70 * u)


def visible_moves(moves, t, lookahead=LOOKAHEAD):
    """The moves on the track now: started but not over, or starting within `lookahead` seconds."""
    return [move for move in moves if move.end > t and move.start < t + lookahead]


def move_color(move):
    return DUET if move.players == 2 else GOLD if move.gold else GREEN


def draw_timeline(frame, moves, t, beat_phase, player_colors=(WHITE, WHITE)):
    """moves: all moves of the song, t: song time (negative during the countdown), beat_phase: 0 on the beat."""
    height, width = frame.shape[:2]
    (left, top, right, bottom), hit_x = timeline_layout(width, height)
    if right - hit_x < 120:
        return frame                                   # the image is too narrow for a timeline
    speed = (right - hit_x) / LOOKAHEAD               # pixels per second
    size = int((bottom - top) * 0.86)                  # pictogram size
    middle_y = (top + bottom) // 2

    track = frame[top:bottom, left:right]
    track[:] = (track * 0.45).astype(frame.dtype)
    cv2.rectangle(frame, (left, top), (right, bottom), GREY, 1)

    coming = [move for move in moves if move.start > t]
    next_move = coming[0] if coming else None
    on_track = visible_moves(moves, t)
    if next_move is not None and next_move not in on_track:
        on_track.append(next_move)                     # the next move waits at the right end

    for move in reversed(on_track):                    # later moves first, so earlier ones are on top
        active = move.start <= t
        color = move_color(move)
        x = hit_x if active else int(hit_x + (move.start - t) * speed)
        if move is next_move:
            x = min(x, right - size // 2 - 4)
        elif x + size // 2 > right:
            continue                                   # not fully on the track yet

        # the hold bar: from the pictogram to the end of the move
        bar_end = min(int(hit_x + (move.end - t) * speed), right - 2)
        bar_h = max(4, size // 6)
        if bar_end > x:
            dim = tuple(int(c * (0.9 if active else 0.55)) for c in color)
            cv2.rectangle(frame, (x, middle_y - bar_h // 2), (bar_end, middle_y + bar_h // 2), dim, -1)

        # the pictogram
        x0, y0 = x - size // 2, middle_y - size // 2
        card = frame[y0:y0 + size, x0:x0 + size]
        card[:] = (card * 0.35).astype(frame.dtype)
        cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), color, 3 if active else 2)
        if move.players == 2 and move.pose2 is not None:
            draw_duet_poses(frame, (x0 + 4, y0 + 4, size - 8, size - 8), move.pose, move.pose2, player_colors, 1.0)
        else:
            # the figure uses x from -2.3 to 2.3 and y from -1.3 to 3.2 (pose units), like the cards
            scale = (size - 8) / 4.6
            draw_pose(frame, move.pose, x, middle_y - 0.95 * scale, scale, WHITE, 2)

    # the hit line, a little thicker on the beat
    thickness = 2 + int(3 * (1.0 - beat_phase))
    cv2.line(frame, (hit_x, top + 3), (hit_x, bottom - 3), WHITE, thickness, cv2.LINE_AA)
    return frame
