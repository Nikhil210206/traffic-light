import cv2

def draw_label(frame, text, x, y, w, h, color):
    """
    Draw bounding box and label on the frame.
    """
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)
    return frame
