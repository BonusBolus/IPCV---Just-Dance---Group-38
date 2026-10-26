"""The Just Dance stage behind the players: a minimal, colourful background instead of the room.

- a gradient from deep blue-purple at the top to pink at the bottom (made once per image size)
- two soft light beams from the top and a glow on the floor (also made once)
- a few big, soft circles of light (bokeh) that slowly float around (moved every frame)

place_players() puts the players (the segmentation mask) on the stage.
"""
import math

import cv2
import numpy as np

# colours as RGB, from the top of the screen to the bottom
GRADIENT = [(18, 10, 58), (70, 16, 110), (150, 28, 120)]
FLOOR_GLOW = (255, 70, 170)
BEAM = (190, 140, 255)
# bokeh: (colour RGB, size as part of the screen height, start x, start y, speed)
BOKEH = [((0, 220, 255), 0.22, 0.12, 0.30, 0.020), ((255, 60, 180), 0.30, 0.80, 0.25, 0.014),
         ((255, 210, 60), 0.16, 0.55, 0.12, 0.025), ((150, 90, 255), 0.26, 0.30, 0.65, 0.017),
         ((0, 220, 255), 0.14, 0.92, 0.70, 0.022), ((255, 60, 180), 0.18, 0.45, 0.85, 0.019)]
BOKEH_STRENGTH = 0.22


def bgr(rgb):
    return tuple(int(c) for c in rgb[::-1])


def soft_disc(radius, color, strength):
    """A round, soft light: bright in the middle, fading to nothing at the edge (uint8, BGR)."""
    size = 2 * radius + 1
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
    fade = np.clip(1.0 - np.sqrt(x * x + y * y) / radius, 0.0, 1.0) ** 2 * strength
    return (fade[..., None] * np.array(bgr(color), np.float32)).astype(np.uint8).reshape(size, size, 3)


def make_base(width, height):
    """The parts that do not move: gradient, light beams and floor glow."""
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    top, middle, bottom = (np.array(bgr(c), np.float32) for c in GRADIENT)
    upper = top + (middle - top) * np.clip(t * 2, 0, 1)[..., None]
    lower = middle + (bottom - middle) * np.clip(t * 2 - 1, 0, 1)[..., None]
    image = np.where((t < 0.5)[..., None], upper, lower) * np.ones((1, width, 1), np.float32)

    # two soft beams from the top corners towards the middle of the floor
    beams = np.zeros((height, width), np.float32)
    for top_x in (0.18, 0.82):
        poly = np.array([(top_x * width - 0.03 * width, 0), (top_x * width + 0.03 * width, 0),
                         (0.5 * width + 0.18 * width * (top_x - 0.5) * 2 + 0.12 * width, height),
                         (0.5 * width + 0.18 * width * (top_x - 0.5) * 2 - 0.12 * width, height)], np.int32)
        cv2.fillPoly(beams, [poly], 1.0)
    beams = cv2.GaussianBlur(beams, (0, 0), width * 0.02) * 0.10
    image += beams[..., None] * np.array(bgr(BEAM), np.float32)

    # a glow on the floor, low in the middle
    glow = np.zeros((height, width), np.float32)
    cv2.ellipse(glow, (width // 2, int(height * 0.97)), (int(width * 0.42), int(height * 0.12)), 0, 0, 360, 1.0, -1)
    glow = cv2.GaussianBlur(glow, (0, 0), height * 0.06) * 0.28
    image += glow[..., None] * np.array(bgr(FLOOR_GLOW), np.float32)
    return np.clip(image, 0, 255).astype(np.uint8)


class Stage:
    def __init__(self):
        self.size = None      # (width, height) the parts below were made for
        self.base = None
        self.discs = None

    def _prepare(self, width, height):
        if self.size != (width, height):
            self.size = (width, height)
            self.base = make_base(width, height)
            self.discs = [soft_disc(max(4, int(size * height)), color, BOKEH_STRENGTH)
                          for color, size, *_ in BOKEH]

    def backdrop(self, width, height, t):
        """The stage at time t (seconds): the base with the floating circles of light."""
        self._prepare(width, height)
        image = self.base.copy()
        for disc, (_, _, start_x, start_y, speed) in zip(self.discs, BOKEH):
            radius = disc.shape[0] // 2
            # float slowly around their start point (the position only depends on the time: no state)
            cx = int((start_x + 0.08 * math.sin(2 * math.pi * speed * t + start_y * 7)) * width)
            cy = int((start_y + 0.06 * math.sin(2 * math.pi * speed * 1.3 * t + start_x * 5)) * height)
            x1, y1, x2, y2 = cx - radius, cy - radius, cx + radius + 1, cy + radius + 1
            sx1, sy1 = max(0, -x1), max(0, -y1)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            part = disc[sy1:sy1 + y2 - y1, sx1:sx1 + x2 - x1]
            image[y1:y2, x1:x2] = cv2.add(image[y1:y2, x1:x2], part)
        return image

    def place_players(self, image, mask, t):
        """Changes `image` in place: where mask is 1 the camera image (the players), where it is 0 the stage."""
        height, width = image.shape[:2]
        stage = self.backdrop(width, height, t)
        image[:] = cv2.blendLinear(image, stage, mask, 1.0 - mask)
        return image
