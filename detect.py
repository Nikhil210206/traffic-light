import cv2
import numpy as np
from utils import draw_label

# Define HSV ranges for Red, Yellow, Green
# Note: Red has two ranges to wrap around the hue spectrum
COLOR_RANGES = {
    "Red":    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    "Yellow": [((20, 100, 100), (30, 255, 255))],
    "Green":  [((40, 70, 70), (90, 255, 255))]
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

    for color_name, ranges in COLOR_RANGES.items():
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        for (lower, upper) in ranges:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))


        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:  # Filter small noise
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                # Add a simple shape constraint, traffic lights are often close to circular or square
                if 0.8 <= aspect_ratio <= 1.2:
                    draw_label(frame, color_name, x, y, w, h, COLOR_BGR[color_name])

    return frame