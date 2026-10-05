"""The NOW and NEXT cards, bottom right of the screen (Countdown and Playing).

NOW:  the move that counts now. Its bar gets empty while the player holds the pose.
      A short flash shows that a new move started. Between moves: "Get ready".
NEXT: the coming move, visible from the countdown on. Its bar fills up until the move
      starts, and the figure pulses a little on every beat.
Gold moves get a gold border and the label "GOLD".
"""
import cv2

from scene.functions import GOLD, GREEN, GREY, WHITE, draw_bar, draw_text
from scene.pose_figure import draw_pose


def draw_move_cards(frame, now_move, now_progress, next_move, next_progress, beat_phase, flash):
    """now_progress / next_progress: 0..1, beat_phase: 0 on the beat (all from song.py)."""
    height, width = frame.shape[:2]
    now_box = (width - 425, height - 310, 230, 290)
    next_box = (width - 180, height - 230, 160, 210)

    if now_move is None:
        draw_card(frame, now_box, "NOW", None, "Get ready", None, GREY)
    else:
        label = "NOW!  GOLD" if now_move.gold else "NOW!"
        color = GOLD if now_move.gold else GREEN
        draw_card(frame, now_box, label, now_move.pose, now_move.name, 1.0 - now_progress, color, flash=flash)

    if next_move is not None:
        label = "NEXT  GOLD" if next_move.gold else "NEXT"
        color = GOLD if next_move.gold else GREY
        zoom = 1.0 + 0.08 * (1.0 - beat_phase)   # biggest on the beat: the pulse
        draw_card(frame, next_box, label, next_move.pose, next_move.name, next_progress, color, zoom=zoom)
    return frame


def draw_card(frame, box, label, pose, caption, bar_fraction, color, flash=False, zoom=1.0):
    """One card: label at the top, the pose in the middle, the name and a bar at the bottom."""
    x, y, w, h = box
    region = frame[y:y + h, x:x + w]
    region[:] = (region * 0.5 + 120).astype(frame.dtype) if flash else (region * 0.4).astype(frame.dtype)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 6 if flash else 3)
    draw_text(frame, label, (x + w // 2, y + 25), 0.7, color)

    if pose is not None:
        # the figure uses x from -2.3 to 2.3 and y from -1.3 to 3.2 (pose units)
        scale = min((w - 20) / 4.6, (h - 100) / 4.5) * zoom
        center_y = y + 45 + (h - 100) // 2
        draw_pose(frame, pose, x + w // 2, center_y - 0.95 * scale, scale, WHITE, 3)

    draw_text(frame, caption, (x + w // 2, y + h - 40), 0.7, WHITE)
    if bar_fraction is not None:
        draw_bar(frame, (x + 10, y + h - 22), (w - 20, 12), bar_fraction, color)
