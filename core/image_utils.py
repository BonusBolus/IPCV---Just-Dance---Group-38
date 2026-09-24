"""OpenCV drawing and image helpers shared by all tasks. All colours are BGR."""
from __future__ import annotations

import cv2
import numpy as np

from core.types import SKELETON_EDGES

FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GOLD = (0, 215, 255)

Rect = tuple[int, int, int, int]  # x, y, w, h in pixels


def to_px(xy, shape) -> tuple[int, int]:
    """Normalized (x, y) -> integer pixel coords for an image of `shape`."""
    h, w = shape[:2]
    return int(round(xy[0] * w)), int(round(xy[1] * h))


def bbox_iou(a, b) -> float:
    """Intersection over union of two (x, y, w, h) boxes."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def resize_to_width(image: np.ndarray, width: int, interpolation=cv2.INTER_AREA) -> np.ndarray:
    h, w = image.shape[:2]
    if w == width:
        return image
    return cv2.resize(image, (width, max(1, round(h * width / w))), interpolation=interpolation)


def _clip(rect: Rect, shape) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    H, W = shape[:2]
    return max(0, x), max(0, y), min(W, x + w), min(H, y + h)


def fill_rect_alpha(img: np.ndarray, rect: Rect, color, alpha: float = 0.6) -> None:
    """Semi-transparent filled rectangle, drawn in place."""
    x0, y0, x1, y1 = _clip(rect, img.shape)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    solid = np.empty_like(roi)
    solid[:] = color
    roi[:] = cv2.addWeighted(solid, alpha, roi, 1 - alpha, 0)


def darken(img: np.ndarray, factor: float = 0.4) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=factor)


def put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.8,
    color=WHITE,
    thickness: int = 2,
    *,
    anchor: str = "left",   # "left", "center" or "right" relative to org[0]; org[1] is the baseline
    bg=None,
    bg_alpha: float = 0.6,
    pad: int = 6,
    outline: bool = True,
) -> Rect:
    """Readable text: optional dark outline and background box. Returns the box that was covered."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    if anchor == "center":
        x -= tw // 2
    elif anchor == "right":
        x -= tw
    box = (x - pad, y - th - pad, tw + 2 * pad, th + base + 2 * pad)
    if bg is not None:
        fill_rect_alpha(img, box, bg, bg_alpha)
    if outline:
        cv2.putText(img, text, (x, y), FONT, scale, BLACK, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
    return box


def draw_bar(img: np.ndarray, rect: Rect, fraction: float, color, bg=(40, 40, 40)) -> None:
    """Horizontal progress/health bar."""
    x, y, w, h = rect
    fill_rect_alpha(img, rect, bg, 0.7)
    fw = int(round(w * float(np.clip(fraction, 0.0, 1.0))))
    if fw > 0:
        cv2.rectangle(img, (x, y), (x + fw, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), WHITE, 1, cv2.LINE_AA)


def vertical_gradient(w: int, h: int, top, bottom) -> np.ndarray:
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    col = np.asarray(top, np.float32) * (1 - t) + np.asarray(bottom, np.float32) * t
    return np.broadcast_to(col, (h, w, 3)).astype(np.uint8)


def _blend_patch(dst: np.ndarray, patch: np.ndarray, x0: int, y0: int, opacity: float) -> None:
    """Alpha-blend BGRA `patch` onto `dst` with its top-left corner at (x0, y0), clipped."""
    ph, pw = patch.shape[:2]
    cx0, cy0, cx1, cy1 = _clip((x0, y0, pw, ph), dst.shape)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    patch = patch[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    alpha = patch[..., 3:4].astype(np.float32) * (opacity / 255.0)
    roi = dst[cy0:cy1, cx0:cx1]
    roi[:] = (patch[..., :3] * alpha + roi * (1.0 - alpha)).astype(np.uint8)


def warp_rgba_affine(dst: np.ndarray, rgba: np.ndarray, src_pts, dst_pts, opacity: float = 1.0) -> None:
    """Warp a BGRA sticker so its 3 anchor points `src_pts` land on `dst_pts` (pixels), then blend.

    An affine map from 3 point pairs covers translation, rotation, scale, shear and
    foreshortening, so e.g. glasses anchored to both eye corners and the nose follow head turns.
    Only the bounding box of the warped sticker is processed.
    """
    src = np.asarray(src_pts, np.float32)
    dstp = np.asarray(dst_pts, np.float32)
    m = cv2.getAffineTransform(src, dstp)
    h, w = rgba.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32) @ m.T
    x0, y0 = np.floor(corners.min(axis=0)).astype(int)
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
    H, W = dst.shape[:2]
    if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H or (x1 - x0) * (y1 - y0) > 4 * W * H:
        return
    m[:, 2] -= (x0, y0)  # warp into the bounding box only
    patch = cv2.warpAffine(rgba, m, (int(x1 - x0), int(y1 - y0)), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    _blend_patch(dst, patch, int(x0), int(y0), opacity)


def overlay_rgba(
    dst: np.ndarray, rgba: np.ndarray, center: tuple[float, float],
    scale: float = 1.0, angle_deg: float = 0.0, opacity: float = 1.0,
) -> None:
    """Alpha-blend a BGRA sticker onto `dst` in place, centred at pixel `center`.

    Scaled by `scale` and rotated by `angle_deg` (counter-clockwise, OpenCV convention).
    Clipped at image borders. Load stickers with cv2.imread(path, cv2.IMREAD_UNCHANGED).
    """
    h, w = rgba.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, scale)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    if nw <= 0 or nh <= 0:
        return
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    warped = cv2.warpAffine(rgba, m, (nw, nh), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    _blend_patch(dst, warped, int(round(center[0] - nw / 2)), int(round(center[1] - nh / 2)), opacity)


def draw_skeleton(
    img: np.ndarray, keypoints: np.ndarray, confidence: np.ndarray, color,
    min_conf: float = 0.3, thickness: int = 3, rect: Rect | None = None,
) -> None:
    """Draw a COCO-17 skeleton. `rect` maps normalized coords into a sub-rectangle (e.g. a panel)."""
    H, W = img.shape[:2]
    ox, oy, rw, rh = rect if rect is not None else (0, 0, W, H)
    pts = [(int(round(ox + x * rw)), int(round(oy + y * rh))) for x, y in keypoints]
    ok = confidence >= min_conf
    for a, b in SKELETON_EDGES:
        if ok[a] and ok[b]:
            cv2.line(img, pts[a], pts[b], color, thickness, cv2.LINE_AA)
    for i in np.flatnonzero(ok):
        cv2.circle(img, pts[i], thickness + 1, color, -1, cv2.LINE_AA)
