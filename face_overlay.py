"""Drawing helpers for face-tracking visualisations."""

import cv2
import numpy as np


def draw_face_outline(image, faces, color=(0, 255, 255), thickness=2):
    """Draw an oval outline over every detected face.

    Args:
        image: BGR OpenCV image to draw on. It is changed in place and returned.
        faces: Face dictionaries produced by ``get_face_properties``.
        color: BGR outline colour.
        thickness: Outline width in pixels.
    """
    for face in faces:
        center_x, center_y = face["location"]
        face_width, face_height = face["size"]
        orientation = face["orientation"]

        # Roll rotates the oval with the face. Yaw and pitch make the outline
        # slightly narrower/shorter when the face turns away from the camera.
        yaw = orientation["yaw"]
        pitch = orientation["pitch"]
        roll_radians = np.radians(orientation["roll"])
        half_width = max(1.0, face_width * (1.0 - min(abs(yaw), 75.0) / 300.0) / 2)
        half_height = max(1.0, face_height * (1.0 - min(abs(pitch), 75.0) / 450.0) / 2)

        cv2.ellipse(
            image,
            (int(center_x), int(center_y)),
            (int(half_width), int(half_height)),
            float(np.degrees(roll_radians)),
            0,
            360,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    return image
