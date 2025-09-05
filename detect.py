import cv2
import numpy as np
from utils import draw_label

# Define a range for skin color in HSV to avoid face detection
SKIN_HSV_MIN = np.array([0, 48, 80], dtype="uint8")
SKIN_HSV_MAX = np.array([20, 255, 255], dtype="uint8")

# Define HSV ranges for Red, Yellow, Green
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
    Detects traffic light colors and returns the annotated frame and the detected state.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, SKIN_HSV_MIN, SKIN_HSV_MAX)
    
    detected_colors = []

    for color_name, ranges in COLOR_RANGES.items():
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        for (lower, upper) in ranges:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        
        if color_name == "Red":
            mask = cv2.subtract(mask, skin_mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                if 0.8 <= aspect_ratio <= 1.2:
                    draw_label(frame, color_name, x, y, w, h, COLOR_BGR[color_name])
                    detected_colors.append(color_name)

    # Determine the overall state based on detected colors (Red has priority)
    light_state = "None"
    if "Red" in detected_colors:
        light_state = "Red"
    elif "Yellow" in detected_colors:
        light_state = "Yellow"
    elif "Green" in detected_colors:
        light_state = "Green"
    
    return frame, light_state