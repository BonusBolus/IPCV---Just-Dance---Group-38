from __future__ import annotations
import functools
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


# Crown shape, in units of the face size.
CROWN_WIDTH = 1.18       # x face width
CROWN_HEIGHT = 0.52      # x face height
CROWN_LIFT = 0.69        # x face height
BAND_TOP = 0.13          # just below the jewels
BAND_BOTTOM = 0.5
TIP_TOP = -0.55          # highest point of the crown
SUPERSAMPLE = 4          # labels are drawn this many times bigger, then shrunk (sharper small text)


def head_point(face: Mapping[str, Any], lift: float) -> tuple[float, float]:
    """Point ``lift`` face heights above the face centre, along the tilted head axis."""
    cx, cy, _, face_height, angle = _head_properties(face)
    radians = np.deg2rad(angle)
    return cx + face_height * lift * np.sin(radians), cy - face_height * lift * np.cos(radians)


def crown_band(face: Mapping[str, Any]) -> tuple[tuple[float, float], float, float, float]:
    """Centre, width, height and angle of the crown's band: the place for the player label."""
    _, _, face_width, face_height, angle = _head_properties(face)
    crown_height = face_height * CROWN_HEIGHT
    band_middle = (BAND_TOP + BAND_BOTTOM) / 2 * CROWN_HEIGHT
    centre = head_point(face, CROWN_LIFT - band_middle)
    return centre, face_width * CROWN_WIDTH, crown_height * (BAND_BOTTOM - BAND_TOP), angle


def crown_top(face: Mapping[str, Any]) -> tuple[float, float]:
    """Point just above the highest tip of the crown."""
    return head_point(face, CROWN_LIFT - TIP_TOP * CROWN_HEIGHT)


def faces_for_players(players: Mapping[int, Any], faces: list) -> dict:
    """Give every visible player the face that belongs to them, or None.

    A face belongs to a player when its centre lies inside the player's box. When more faces
    qualify, the one closest to the player's nose (or the top of the box) wins. Each face is
    given to at most one player."""
    pairs = []
    for player_id, player in players.items():
        if not player["visible"] or player["box"] is None:
            continue
        x, y, w, h = player["box"]
        nose = player["keypoints"].get("nose") or (x + w / 2, y + h * 0.1)
        for index, face in enumerate(faces):
            try:
                fx, fy, *_ = _head_properties(face)
            except (TypeError, ValueError):
                continue
            if x <= fx <= x + w and y <= fy <= y + h:
                pairs.append((np.hypot(fx - nose[0], fy - nose[1]), player_id, index))

    result = {player_id: None for player_id in players}
    used = set()
    for _, player_id, index in sorted(pairs):
        if result[player_id] is None and index not in used:
            result[player_id] = faces[index]
            used.add(index)
    return result


def add_crown(frame: np.ndarray, smooth_face_properties: Any) -> np.ndarray:
    """Draw a gold crown above a player's smoothed face position in ``frame``.

    The crown is sized from the face width and rotated using the smoothed head
    angle."""
    for face in smooth_face_properties:
        try:
            _, _, face_width, face_height, angle = _head_properties(face)
        except (TypeError, ValueError):
            continue
        crown_width = face_width * CROWN_WIDTH
        crown_height = face_height * CROWN_HEIGHT
        # Coordinates are relative to the crown's centre, positioned above the face.
        crown_centre = head_point(face, CROWN_LIFT)
        w, h = crown_width / 2, crown_height
        polygon = np.array([
            [-w, h * BAND_BOTTOM], [-w, -h / 2], [-w * .52, h * .05], [-w * .22, h * TIP_TOP],
            [0, h * .05], [w * .22, h * TIP_TOP], [w * .52, h * .05], [w, -h / 2], [w, h * BAND_BOTTOM],
        ])
        points = _rotated(polygon, crown_centre, angle)

        # Now actually draw the crown on the frame:
        cv2.fillPoly(frame, [points], (0, 215, 255))
        cv2.polylines(frame, [points], True, (0, 130, 210), max(1, round(face_width / 38)), cv2.LINE_AA)
        for x in (-w * .52, 0, w * .52):
            jewel = _rotated(np.array([[x, h * .02]]), crown_centre, angle)[0]
            cv2.circle(frame, tuple(jewel), max(2, round(face_width / 18)), (40, 40, 220), -1, cv2.LINE_AA)
    return frame


@functools.lru_cache(maxsize=64)
def _label_tile(text: str, color: tuple, scale: float, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """The label as a small tile and its mask, tilted by `angle`. Kept by add_player_label()."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    # The text is small, so it is written SUPERSAMPLE times bigger on a square tile (room to
    # rotate), together with a mask of its pixels, and shrunk afterwards: this keeps it sharp.
    big_scale = scale * SUPERSAMPLE
    thickness = max(2, round(big_scale * 1.8))
    outline = thickness + max(3, round(big_scale * 1.2))
    (text_width, text_height), baseline = cv2.getTextSize(text, font, big_scale, outline)
    size = int(np.hypot(text_width, text_height + baseline) / SUPERSAMPLE + 4) * SUPERSAMPLE
    tile = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    origin = ((size - text_width) // 2, (size + text_height) // 2)
    # A thicker black text is put behind the coloured text to give it a black border.
    cv2.putText(tile, text, origin, font, big_scale, (0, 0, 0), outline, cv2.LINE_AA)
    cv2.putText(tile, text, origin, font, big_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(mask, text, origin, font, big_scale, 255, outline, cv2.LINE_AA)

    # Tilt the tile with the head. getRotationMatrix2D turns the other way than _rotated().
    rotation = cv2.getRotationMatrix2D((size / 2, size / 2), -angle, 1.0)
    tile = cv2.warpAffine(tile, rotation, (size, size), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(mask, rotation, (size, size), flags=cv2.INTER_LINEAR)
    size //= SUPERSAMPLE
    tile = cv2.resize(tile, (size, size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
    tile.flags.writeable = False                      # the same tile is used again: never change it
    mask.flags.writeable = False
    return tile, mask


def add_player_label(frame: np.ndarray, text: str, face: Mapping[str, Any], color=(255, 255, 255)) -> np.ndarray:
    """Write ``text`` on the band of the crown (or the forehead), tilted with the head."""
    try:
        centre, band_width, band_height, angle = crown_band(face)
    except (TypeError, ValueError):
        return frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    (unit_width, unit_height), _ = cv2.getTextSize(text, font, 1.0, 2)
    scale = max(0.4, min(band_height * 0.8 / unit_height, band_width * 0.9 / unit_width))

    # Drawing the label is slow (see _label_tile), and it hardly changes from frame to frame, so the
    # finished label is kept: size in 2% steps and tilt in 1 degree steps (you do not see the difference).
    tile, mask = _label_tile(text, tuple(int(c) for c in color), round(scale / 0.02) * 0.02, round(angle))
    size = tile.shape[0]

    # Position of the top left corner of the tile in the frame.
    top = int(round(centre[1] - size / 2))
    left = int(round(centre[0] - size / 2))
    y1, y2 = max(0, top), min(frame.shape[0], top + size)
    x1, x2 = max(0, left), min(frame.shape[1], left + size)
    # Blend the text into the frame, while making sure to not access pixels outside of the frame.
    if y1 < y2 and x1 < x2:
        source = tile[y1 - top:y2 - top, x1 - left:x2 - left].astype(np.float32)
        alpha = mask[y1 - top:y2 - top, x1 - left:x2 - left, np.newaxis].astype(np.float32) / 255
        region = frame[y1:y2, x1:x2]
        region[:] = (alpha * source + (1 - alpha) * region).astype(frame.dtype)
    return frame
