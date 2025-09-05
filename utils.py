import cv2

def draw_label(frame, text, x, y, w, h, color):
    """
    Draw bounding box and label on the frame.
    """
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label = f"{text}"
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)
    return frame

def draw_status(frame, state):
    """
    Draw the main status message (STOP, GET READY, GO) on the frame.
    """
    if state == "Red":
        text = "STOP"
        color = (0, 0, 255)  # Red in BGR
    elif state == "Yellow":
        text = "GET READY"
        color = (0, 255, 255)  # Yellow in BGR
    elif state == "Green":
        text = "GO!"
        color = (0, 255, 0)  # Green in BGR
    else:
        return frame  # No status to draw

    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    cv2.putText(frame, text, (int((frame.shape[1] - text_width) / 2), 70), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA)
    return frame