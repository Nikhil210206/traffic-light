import cv2
import numpy as np
from utils import draw_label

# Define HSV ranges for Red, Yellow, Green
COLOR_RANGES = {
    "Red":    [(0, 100, 100), (10, 255, 255)],
    "Yellow": [(15, 100, 100), (35, 255, 255)],
    "Green":  [(40, 100, 100), (90, 255, 255)]
}

COLOR_BGR = {
    "Red": (0, 0, 255),
    "Yellow": (0, 255, 255),
    "Green": (0, 255, 0)
}

def detect_traffic_lights(frame):
    """
    Detect traffic light colors in a frame using HSV segmentation.
    Returns annotated frame.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, (lower, upper) in COLOR_RANGES.items():
        lower = np.array(lower)
        upper = np.array(upper)

        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:  # Filter small noise
                x, y, w, h = cv2.boundingRect(cnt)
                draw_label(frame, color_name, x, y, w, h, COLOR_BGR[color_name])

    return frame
