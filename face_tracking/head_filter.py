import cv2
import numpy as np


def enlarge_heads(output, faces, enlargement=1.0):
    """Magnify each detected head in place with a feathered warp.

    ``faces`` contains the dictionaries returned by ``smooth_face_properties``."""
    if enlargement <= 1.0:
        return output

    source = output.copy()
    image_height, image_width = output.shape[:2]

    for face in faces:
        center_x, center_y = face["location"]
        face_width, face_height = face["size"]

        # The destination oval is larger than the face. Mapping each destination
        # pixel closer to the centre makes the sampled content appear enlarged.
        radius_x = max(1, int(face_width * enlargement / 2))
        radius_y = max(1, int(face_height * enlargement / 2))
        # Get a rectangular box around the head, whilst staying inside the frame.
        x1 = max(0, center_x - radius_x)
        x2 = min(image_width, center_x + radius_x + 1)
        y1 = max(0, center_y - radius_y)
        y2 = min(image_height, center_y + radius_y + 1)

        # Skip when an invalid or empty region is detected.
        if x1 >= x2 or y1 >= y2:
            continue

        x_coordinates, y_coordinates = np.meshgrid(
            np.arange(x1, x2, dtype=np.float32),
            np.arange(y1, y2, dtype=np.float32),
        )

        # remap requires float32 coordinate maps. Keeping them as float32 also
        # avoids the OpenCV 5 assertion caused by NumPy float64 promotion.
        map_x = (center_x + (x_coordinates - center_x) / enlargement).astype(np.float32)
        map_y = (center_y + (y_coordinates - center_y) / enlargement).astype(np.float32)

        # Performing the actual remapping, does linear interpolation between pixels to enlarge heads.
        enlarged = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        # Fade the effect out at the oval edge to hide the region boundary.
        distance = np.sqrt(
            ((x_coordinates - center_x) / radius_x) ** 2
            + ((y_coordinates - center_y) / radius_y) ** 2
        )
        # Apply feathering to smooth the transition between head and environment.
        alpha = np.clip((1.0 - distance) / 0.2, 0.0, 1.0)[..., np.newaxis]
        original_region = output[y1:y2, x1:x2]
        output[y1:y2, x1:x2] = (alpha * enlarged + (1.0 - alpha) * original_region).astype(np.uint8)

    return output
