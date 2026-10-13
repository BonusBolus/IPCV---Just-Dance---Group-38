"""Head-anchored visual effects for the game state.
"""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np


def _head_properties(face_properties: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    """Return head properties from a ``smooth_face_properties`` face record."""
    x, y = map(float, face_properties["location"])
    width, height = map(float, face_properties["size"])
    orientation = face_properties["orientation"]
    if not isinstance(orientation, Mapping):
        raise TypeError("smooth_face_properties returned a non-mapping orientation")
    angle = float(orientation["roll"])
    if width <= 0 or height <= 0:
        raise ValueError("smooth_face_properties returned a non-positive face size")
    return x, y, width, height, angle


def _rotated(points: np.ndarray, centre: tuple[float, float], angle: float) -> np.ndarray:
    radians = np.deg2rad(angle)
    rotation = np.array([[np.cos(radians), -np.sin(radians)],
                         [np.sin(radians), np.cos(radians)]])
    return (points @ rotation.T + np.asarray(centre)).round().astype(np.int32)


def add_crown(frame: np.ndarray, smooth_face_properties: Any) -> np.ndarray:
    """Draw a gold crown above a player's smoothed face position in ``frame``.

    The crown is sized from the face width and rotated using the smoothed head
    angle.
    """
    for face in smooth_face_properties:
        try:
            cx, cy, face_width, face_height, angle = _head_properties(face)
        except (TypeError, ValueError):
            # Face detectors commonly retain a zero-sized placeholder while a
            # face is lost between frames.  It is not drawable, but also is not
            # an error for the game loop.
            continue
        crown_width = face_width * 1.18
        crown_height = face_height * 0.52
        # Coordinates are relative to the crown's centre, positioned above the face.
        crown_centre = (cx + face_height * 0.69 * np.sin(angle * np.pi / 180), cy - face_height * 0.69 * np.cos(angle * np.pi / 180))
        w, h = crown_width / 2, crown_height
        polygon = np.array([
            [-w, h / 2], [-w, -h / 2], [-w * .52, h * .05], [-w * .22, -h * .55],
            [0, h * .05], [w * .22, -h * .55], [w * .52, h * .05], [w, -h / 2], [w, h / 2],
        ])
        points = _rotated(polygon, crown_centre, angle)

        # Now actually draw the crown on the frame:
        cv2.fillPoly(frame, [points], (0, 215, 255))
        cv2.polylines(frame, [points], True, (0, 130, 210), max(1, round(face_width / 38)), cv2.LINE_AA)
        for x in (-w * .52, 0, w * .52):
            jewel = _rotated(np.array([[x, h * .02]]), crown_centre, angle)[0]
            cv2.circle(frame, tuple(jewel), max(2, round(face_width / 18)), (40, 40, 220), -1, cv2.LINE_AA)
    return frame


def add_player_name(frame: np.ndarray, text: str, smooth_face_properties: Any) -> np.ndarray:
    """Write ``text`` above a player's head using their smoothed face properties."""
    for face in smooth_face_properties:
        try:
            cx, cy, face_width, face_height, angle = _head_properties(face)
        except (TypeError, ValueError):
            # Face detectors commonly retain a zero-sized placeholder while a
            # face is lost between frames.  It is not drawable, but also is not
            # an error for the game loop.
            continue
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(1, face_width / 100)
        thickness = max(1, round(scale * 2))
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)

        # The text is first written on a tile before putting it in the frame.
        tile = np.zeros((text_height + baseline + 12, text_width + 12, 3), dtype=frame.dtype)
        origin = (6, text_height + 4)
        # A thicker black text is put behind the white text to create white text with black borders.
        cv2.putText(tile, text, origin, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(tile, text, origin, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        # Calculate the position of the top left corner of the tile in the frame.
        top = int(round(cy - face_height * np.cos(angle * np.pi / 180) - tile.shape[0] / 2))
        left = int(round(cx + face_height * np.sin(angle * np.pi / 180) - tile.shape[1] / 2))
        y1, y2 = max(0, top), min(frame.shape[0], top + tile.shape[0])
        x1, x2 = max(0, left), min(frame.shape[1], left + tile.shape[1])
        # Draws the text in the frame, while making sure to not access pixels outside of the frame.
        if y1 < y2 and x1 < x2:
            source = tile[y1 - top:y2 - top, x1 - left:x2 - left]
            mask = np.any(source != 0, axis=2)
            frame_region = frame[y1:y2, x1:x2]
            frame_region[mask] = source[mask]
    return frame
