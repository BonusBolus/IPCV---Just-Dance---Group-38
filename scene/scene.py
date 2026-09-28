import cv2

def createScene(frame):
    return addTitle(frame)

def addTitle(frame):
    """Add title to the frame."""
    width = frame.shape[1]
    cv2.putText(frame, "Just Dance", (width // 2 - 170, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (240, 0, 240), 2)
    return frame