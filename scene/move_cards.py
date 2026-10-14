"""The NOW and NEXT cards, bottom right of the screen (Countdown and Playing).

NOW:  the move that counts now. Its bar gets empty while the player holds the pose.
      A short flash shows that a new move started. Between moves: "Get ready".
NEXT: the coming move, visible from the countdown on. Its bar fills up until the move
      starts, and the figure pulses a little on every beat.
Gold moves get a gold border and the label "GOLD".
Duet moves get a cyan border and the label "DUET", plus a figure in each player's colour.
"""
import cv2

from scene.functions import DUET, GOLD, GREEN, GREY, WHITE, draw_bar, draw_text
from scene.pose_figure import draw_pose


def draw_move_cards(frame, now_move, now_progress, next_move, next_progress, beat_phase, flash,
                    player_colors=(WHITE, WHITE)):
    """now_progress / next_progress: 0..1, beat_phase: 0 on the beat (all from song.py)."""
    height, width = frame.shape[:2]
    now_box = (width - 425, height - 310, 230, 290)
    next_box = (width - 180, height - 230, 160, 210)

    if now_move is None:
        draw_card(frame, now_box, "NOW", None, "Get ready", None, GREY)
    else:
        label = "NOW!  GOLD" if now_move.gold else "NOW!"
        if now_move.players == 2:
            label += "  DUET"
        color = DUET if now_move.players == 2 else GOLD if now_move.gold else GREEN
        draw_card(frame, now_box, label, now_move.pose, now_move.name, 1.0 - now_progress, color,
                  flash=flash, pose2=now_move.pose2 if now_move.players == 2 else None,
                  player_colors=player_colors)

    if next_move is not None:
        label = "NEXT  GOLD" if next_move.gold else "NEXT"
        if next_move.players == 2:
            label += "  DUET"
        color = DUET if next_move.players == 2 else GOLD if next_move.gold else GREY
        zoom = 1.0 + 0.08 * (1.0 - beat_phase)   # biggest on the beat: the pulse
        draw_card(frame, next_box, label, next_move.pose, next_move.name, next_progress, color,
                  zoom=zoom, pose2=next_move.pose2 if next_move.players == 2 else None,
                  player_colors=player_colors)
    return frame


def draw_card(frame, box, label, pose, caption, bar_fraction, color, flash=False, zoom=1.0,
              pose2=None, player_colors=(WHITE, WHITE)):
    """One card: label at the top, the pose in the middle, the name and a bar at the bottom."""
    x, y, w, h = box
    region = frame[y:y + h, x:x + w]
    region[:] = (region * 0.5 + 120).astype(frame.dtype) if flash else (region * 0.4).astype(frame.dtype)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 6 if flash else 3)
    if pose2 is not None and "GOLD" in label:
        cv2.rectangle(frame, (x + 5, y + 5), (x + w - 5, y + h - 5), GOLD, 1)
    draw_card_heading(frame, label, (x + w // 2, y + 25), w - 20, color)

    if pose is not None and pose2 is not None:
        draw_duet_poses(frame, (x + 12, y + 45, w - 24, h - 104),
                        pose, pose2, player_colors, zoom)
    elif pose is not None:
        # the figure uses x from -2.3 to 2.3 and y from -1.3 to 3.2 (pose units)
        scale = min((w - 20) / 4.6, (h - 100) / 4.5) * zoom
        center_y = y + 45 + (h - 100) // 2
        draw_pose(frame, pose, x + w // 2, center_y - 0.95 * scale, scale, WHITE, 3)

    draw_text(frame, caption, (x + w // 2, y + h - 40), 0.7, WHITE)
    if bar_fraction is not None:
        draw_bar(frame, (x + 10, y + h - 22), (w - 20, 12), bar_fraction, color)


def draw_card_heading(frame, label, center, max_width, color):
    """Fit the heading on either card, keeping GOLD gold on a cyan duet card."""
    words = label.split()
    widths = [cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
              for word in words]
    gap = 12
    factor = min(1.0, max_width / (sum(widths) + gap * (len(words) - 1)))
    size = 0.7 * factor
    widths = [cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, size, 2)[0][0]
              for word in words]
    gap = int(gap * factor)
    x = center[0] - (sum(widths) + gap * (len(words) - 1)) // 2
    for word, width in zip(words, widths):
        draw_text(frame, word, (x + width // 2, center[1]), size, GOLD if word == "GOLD" else color)
        x += width + gap


def draw_duet_poses(frame, box, pose, pose2, colors, zoom):
    """Fit both roles side by side, including room for the NEXT card's beat pulse."""
    x, y, w, h = box
    bounds = []
    for points in (pose, pose2):
        visible = [point for point in points.values() if point is not None]
        # Include the head radius and the neck added by draw_pose.
        visible += [(-0.2, -0.2), (0.2, 0.3)]
        bounds.append((min(p[0] for p in visible), max(p[0] for p in visible),
                       min(p[1] for p in visible), max(p[1] for p in visible)))
    left, right = bounds
    offset = left[1] - right[0] + 0.35
    min_x, max_x = left[0], offset + right[1]
    min_y, max_y = min(left[2], right[2]), max(left[3], right[3])
    scale = min(w / (max_x - min_x), h / (max_y - min_y)) * zoom / 1.08
    origin_x = x + w / 2 - (min_x + max_x) * scale / 2
    nose_y = y + h / 2 - (min_y + max_y) * scale / 2
    draw_pose(frame, pose, origin_x, nose_y, scale, colors[0], 2)
    draw_pose(frame, pose2, origin_x + offset * scale, nose_y, scale, colors[1], 2)
