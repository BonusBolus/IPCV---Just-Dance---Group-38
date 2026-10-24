"""Drawing the bonus star (see game_logic/bonus_star.py) and the burst with "+100" when a player grabs it."""
import math

import cv2
import numpy as np

from background.stage import soft_disc
from scene.functions import GOLD, WHITE, draw_text

GLOW_COLOR = (255, 215, 90)    # RGB, soft_disc() takes RGB
_glows = {}                     # radius -> glow image (made once per size)


def star_points(center, radius, angle):
    """The 10 corners of a 5-pointed star (outer and inner points), turned by `angle` (radians)."""
    points = []
    for k in range(10):
        r = radius if k % 2 == 0 else radius * 0.45
        a = angle - math.pi / 2 + k * math.pi / 5
        points.append((center[0] + r * math.cos(a), center[1] + r * math.sin(a)))
    return np.array(points, np.int32)


def _add_glow(frame, center, radius):
    if radius not in _glows:
        _glows[radius] = soft_disc(radius, GLOW_COLOR, 0.8)
    glow = _glows[radius]
    height, width = frame.shape[:2]
    x1, y1 = center[0] - radius, center[1] - radius
    sx, sy = max(0, -x1), max(0, -y1)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, center[0] + radius + 1), min(height, center[1] + radius + 1)
    if x2 > x1 and y2 > y1:
        frame[y1:y2, x1:x2] = cv2.add(frame[y1:y2, x1:x2], glow[sy:sy + y2 - y1, sx:sx + x2 - x1])


def draw_star(frame, center, size, t, alpha=1.0):
    """A gold star with a glow that turns slowly and pulses. size: radius in pixels; alpha: 0..1 (fade in/out)."""
    if alpha <= 0:
        return frame
    radius = int(size * (1.0 + 0.08 * math.sin(2 * math.pi * 2.0 * t)))
    height, width = frame.shape[:2]
    pad = int(radius * 2.4)
    x1, y1 = max(0, center[0] - pad), max(0, center[1] - pad)
    x2, y2 = min(width, center[0] + pad), min(height, center[1] + pad)
    if x2 <= x1 or y2 <= y1:
        return frame
    region = frame[y1:y2, x1:x2]
    layer = region.copy()
    local = (center[0] - x1, center[1] - y1)
    _add_glow(layer, local, int(radius * 2.2))
    points = star_points(local, radius, t * 1.5)
    cv2.fillPoly(layer, [points], GOLD, cv2.LINE_AA)
    cv2.polylines(layer, [points], True, WHITE, 2, cv2.LINE_AA)
    region[:] = layer if alpha >= 1.0 else cv2.addWeighted(layer, alpha, region, 1.0 - alpha, 0)
    return frame


def draw_star_burst(frame, center, text, color, progress):
    """After a grab: rays that fly out and "+100" that rises, in the player's colour. progress: 0..1."""
    if not 0 <= progress < 1:
        return frame
    fade = 1.0 - progress
    for k in range(8):
        a = k * math.pi / 4
        r1, r2 = 20 + 90 * progress, 40 + 130 * progress
        p1 = (int(center[0] + r1 * math.cos(a)), int(center[1] + r1 * math.sin(a)))
        p2 = (int(center[0] + r2 * math.cos(a)), int(center[1] + r2 * math.sin(a)))
        cv2.line(frame, p1, p2, GOLD, max(1, int(4 * fade)), cv2.LINE_AA)
    rise = int(60 * progress)
    size = 1.4 if progress > 0.15 else 1.4 + (0.15 - progress) * 4      # a short pop, like the ratings
    return draw_text(frame, text, (center[0], center[1] - 30 - rise), size, color, 3)
