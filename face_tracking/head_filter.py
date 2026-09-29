import cv2
import numpy as np


def enlarge_heads(output, faces, enlargement=1.0):
    """Magnify each detected head with a Snapchat-style warp.

    Args:
        output: BGR image to modify.
        faces: Face dictionaries returned by ``smooth_face_properties``.
        enlargement: Head scale factor. ``1.0`` leaves the image unchanged;
            ``1.35`` makes the face/head area appear roughly 35% larger.

    Returns:
        The modified ``output`` image.
    """
    if enlargement <= 1.0:
        return output

    source = output.copy()
    image_height, image_width = output.shape[:2]

    for face in faces:
        center_x, center_y = face["location"]
        face_width, face_height = face["size"]

        # The destination is larger than the detected face. Sampling it from
        # nearer the centre enlarges the head while leaving the rest unchanged.
        radius_x = max(1, int(face_width * enlargement / 2))
        radius_y = max(1, int(face_height * enlargement * 0.6))
        x1 = max(0, center_x - radius_x)
        x2 = min(image_width, center_x + radius_x + 1)
        y1 = max(0, center_y - radius_y)
        y2 = min(image_height, center_y + radius_y + 1)

        if x1 >= x2 or y1 >= y2:
            continue

        x_coordinates, y_coordinates = np.meshgrid(
            np.arange(x1, x2, dtype=np.float32),
            np.arange(y1, y2, dtype=np.float32),
        )

        map_x = center_x + (x_coordinates - center_x) / enlargement
        map_y = center_y + (y_coordinates - center_y) / enlargement
        enlarged = cv2.remap(
            source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )

        # Feather only the edge of the oval so the warped region joins the
        # surrounding image smoothly instead of showing a visible boundary.
        distance = np.sqrt(
            ((x_coordinates - center_x) / radius_x) ** 2
            + ((y_coordinates - center_y) / radius_y) ** 2
        )
        
        alpha = np.clip((1.0 - distance) / 0.2, 0.0, 1.0)[..., np.newaxis]
        original_region = output[y1:y2, x1:x2]
        output[y1:y2, x1:x2] = (
            alpha * enlarged + (1.0 - alpha) * original_region
        ).astype(np.uint8)

    return output
