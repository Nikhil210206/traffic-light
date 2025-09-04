import cv2
import argparse
from detect import detect_traffic_lights
import os

def run_detection(source):
    if source == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        if not os.path.exists(source):
            print(f"Error: The file '{source}' does not exist.")
            return
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Cannot open source: {source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished processing the video or webcam feed.")
            break

        frame = detect_traffic_lights(frame)
        cv2.imshow("Traffic Light Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Light Detection using OpenCV")
    parser.add_argument("--source", type=str, default="webcam",
                        help="Source: 'webcam' or path to video/image")
    args = parser.parse_args()

    # If source is an image, read directly
    if args.source.lower().endswith((".jpg", ".png", ".jpeg")):
        if not os.path.exists(args.source):
            print(f"Error: The image file '{args.source}' does not exist.")
        else:
            img = cv2.imread(args.source)
            if img is None:
                print(f"Error: Cannot read image: {args.source}")
            else:
                result = detect_traffic_lights(img)
                cv2.imshow("Traffic Light Detection", result)
                print("Press any key to exit.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    else:
        run_detection(args.source)