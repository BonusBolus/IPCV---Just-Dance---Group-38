"""Task 1: procedurally drawn BGRA face stickers (no image files needed).

Every sticker comes with its anchor points in sticker pixels, so face_effects.py can map them
onto face landmarks with an affine warp. PNG files in assets/effects/ with the same name
override the drawn version (same anchor layout).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.config import ASSETS


@dataclass
class Sticker:
    rgba: np.ndarray
    anchors: np.ndarray  # (3, 2) float32: points that map onto face landmarks


def _canvas(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 4), np.uint8)


def _load_override(name: str) -> np.ndarray | None:
    path = ASSETS / "effects" / f"{name}.png"
    if Path(path).exists():
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3 and img.shape[2] == 4:
            return img
    return None


def sunglasses(frame_color, lens_color, name: str = "sunglasses") -> Sticker:
    """Glasses; anchors = (image-left eye outer corner, image-right eye outer corner, nose tip)."""
    w, h = 400, 150
    img = _load_override(name)
    if img is None:
        img = _canvas(w, h)
        for cx in (105, 295):
            cv2.ellipse(img, (cx, 72), (88, 58), 0, 0, 360, (*lens_color, 225), -1, cv2.LINE_AA)
            cv2.ellipse(img, (cx - 30, 52), (26, 12), -25, 0, 360, (255, 255, 255, 120), -1, cv2.LINE_AA)
            cv2.ellipse(img, (cx, 72), (88, 58), 0, 0, 360, (*frame_color, 255), 9, cv2.LINE_AA)
        cv2.line(img, (185, 60), (215, 60), (*frame_color, 255), 10, cv2.LINE_AA)
        cv2.line(img, (18, 55), (0, 45), (*frame_color, 255), 9, cv2.LINE_AA)
        cv2.line(img, (382, 55), (400, 45), (*frame_color, 255), 9, cv2.LINE_AA)
    # eye corners lie inside the lenses (glasses are ~1.4x wider than the eye span); the nose tip
    # is ~half an eye-span below the eye line
    return Sticker(img, np.array([[60, 72], [340, 72], [200, 212]], np.float32))


def star(color=(0, 215, 255), size: int = 160) -> Sticker:
    img = _load_override("star")
    if img is None:
        img = _canvas(size, size)
        c, r_out, r_in = size / 2, size * 0.48, size * 0.2
        pts = [(c + (r_out if i % 2 == 0 else r_in) * np.sin(i * np.pi / 5),
                c - (r_out if i % 2 == 0 else r_in) * np.cos(i * np.pi / 5)) for i in range(10)]
        pts = np.array(pts, np.int32)
        cv2.fillPoly(img, [pts], (*color, 255), cv2.LINE_AA)
        cv2.polylines(img, [pts], True, (255, 255, 255, 255), 4, cv2.LINE_AA)
    return Sticker(img, np.array([[0, 0], [size, 0], [0, size]], np.float32))


def crown() -> Sticker:
    """Crown; anchors = (bottom-left, bottom-right, top-centre)."""
    w, h = 300, 190
    img = _load_override("crown")
    if img is None:
        img = _canvas(w, h)
        pts = np.array([[10, 180], [290, 180], [290, 60], [222, 120], [150, 20], [78, 120], [10, 60]], np.int32)
        cv2.fillPoly(img, [pts], (0, 200, 255, 255), cv2.LINE_AA)
        cv2.polylines(img, [pts], True, (0, 120, 200, 255), 6, cv2.LINE_AA)
        cv2.rectangle(img, (10, 150), (290, 180), (0, 160, 230, 255), -1)
        for x, col in ((60, (60, 60, 230)), (150, (230, 120, 40)), (240, (60, 200, 60))):
            cv2.circle(img, (x, 165), 11, (*col, 255), -1, cv2.LINE_AA)
        for x, y in ((10, 60), (150, 20), (290, 60)):
            cv2.circle(img, (x, y), 12, (255, 255, 255, 255), -1, cv2.LINE_AA)
    return Sticker(img, np.array([[10, 180], [290, 180], [150, 20]], np.float32))
