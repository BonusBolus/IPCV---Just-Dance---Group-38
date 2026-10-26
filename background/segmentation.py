"""Which pixels belong to the players: person segmentation with YOLO (yolo26n-seg).

The model finds every person and gives each one a mask (a small image: 1 = person, 0 = room).
Only the masks that belong to a player are kept, so someone walking past in the back disappears
into the background. The result is one soft mask (0..1) at the size of the game image.

Like the pose model, it runs with OpenVINO (converted once, the folder is made next to the .pt file).
"""
import cv2
import numpy as np
from ultralytics import YOLO

import config
from pose_tracking.pose_processing import use_one_openvino_core


def load_segment_model():
    if not config.USE_OPENVINO:
        return YOLO(str(config.SEGMENT_MODEL_PATH))
    use_one_openvino_core()                 # a second OpenVINO Core can crash the program, see there
    folder = config.SEGMENT_MODEL_PATH.with_name(config.SEGMENT_MODEL_PATH.stem + "_openvino_model")
    if not folder.exists():
        print("Converting the segmentation model to OpenVINO (only the first time)...")
        YOLO(str(config.SEGMENT_MODEL_PATH)).export(format="openvino", dynamic=True)
    return YOLO(str(folder), task="segment")


def remove_padding(masks, image_shape):
    """YOLO makes the image a multiple of 32 px with a border (padding); the masks have that border too.
    Cut it off, the same way as Ultralytics (ops.scale_masks), so the masks fit the image again."""
    mask_h, mask_w = masks.shape[1:]
    image_h, image_w = image_shape[:2]
    gain = min(mask_h / image_h, mask_w / image_w)
    pad_w, pad_h = (mask_w - round(image_w * gain)) / 2, (mask_h - round(image_h * gain)) / 2
    top, left = round(pad_h - 0.1), round(pad_w - 0.1)
    return masks[:, top:mask_h - round(pad_h + 0.1), left:mask_w - round(pad_w + 0.1)]


def box_overlap(a, b):
    """How much two boxes (x1, y1, x2, y2) overlap: the shared area / the area of the smaller box (0..1)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / smaller if smaller > 0 else 0.0


class PersonSegmenter:
    def __init__(self, model=None):
        self.model = model or load_segment_model()

    def find_people(self, image):
        """Every person in the image: (boxes (n, 4) as x1, y1, x2, y2 in image pixels, small masks (n, h, w)).
        Runs in its own thread (see Game.step), at the same time as the pose model."""
        result = self.model.predict(image, imgsz=config.SEGMENT_SIZE, classes=[0], verbose=False)[0]
        if result.masks is None or len(result.masks) == 0:
            return np.zeros((0, 4), np.float32), None
        masks = remove_padding(result.masks.data.cpu().numpy(), image.shape)
        return result.boxes.xyxy.cpu().numpy(), masks


def player_mask(people, player_boxes, image_shape):
    """One soft mask (float32, 0..1, image size) with the people that belong to a player.

    people:       find_people() of this frame
    player_boxes: (x, y, w, h) of the players; a person belongs to a player when their boxes overlap
                  (BOX_OVERLAP). None or an empty list: everyone is kept (better an extra person on the
                  stage than a player who disappears)."""
    height, width = image_shape[:2]
    boxes, masks = people
    if masks is None:
        return np.zeros((height, width), np.float32)
    player_boxes = [(x, y, x + w, y + h) for x, y, w, h in (player_boxes or [])]
    keep = [i for i, box in enumerate(boxes)
            if not player_boxes or any(box_overlap(box, p) >= config.BOX_OVERLAP for p in player_boxes)]
    if not keep:
        return np.zeros((height, width), np.float32)
    mask = masks[keep].max(axis=0).astype(np.float32)
    # the edge a little inside the person (no light rim of the room), then soft; on the small mask
    # (0.3 ms instead of 8 ms on the game image), and making it bigger smooths it once more
    mask = np.clip((mask - 0.45) * 5.0, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), config.MASK_SOFTNESS * mask.shape[1] / width)
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)


class PlayerMasker:
    """player_mask() for the game, made careful: a player must never disappear from the stage.

    - A player the tracker does not see for a moment (someone stands close by, or walks past in front)
      keeps their last place for PLAYER_KEEP_TIME, so the person there stays on the stage.
    - Other people are only removed while the players are fixed (countdown, playing, results). In the
      song selection and the registration everyone stays: people still have to step in there.
    - REMOVE_BYSTANDERS = False in config.py keeps everyone, always."""

    def __init__(self):
        self.last_seen = {}     # player id -> (box, time)

    def mask(self, people, players, image_shape, now, players_fixed):
        for player in players:
            if player["visible"] and player["box"] is not None:
                self.last_seen[player["id"]] = (player["box"], now)
        if not (config.REMOVE_BYSTANDERS and players_fixed):
            return player_mask(people, None, image_shape)
        boxes = [box for box, seen in self.last_seen.values() if now - seen <= config.PLAYER_KEEP_TIME]
        return player_mask(people, boxes, image_shape)
