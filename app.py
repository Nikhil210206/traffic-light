import streamlit as st
import cv2
import numpy as np
import os
from detect import detect_traffic_lights
from utils import draw_label
import tempfile

def main():
    st.title("🚦 Traffic Light Detection System")
    st.sidebar.title("Options")
    
    source = st.sidebar.radio("Select Source", ("Image", "Video", "Webcam"))

    if source == "Image":
        uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            # Display original image correctly
            img_rgb_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # <-- FIX
            st.sidebar.image(img_rgb_display, caption="Uploaded Image")

            if st.sidebar.button("Detect Traffic Lights"):
                # Process the BGR image
                result_img = detect_traffic_lights(img.copy())
                # Convert the final result to RGB for display
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB) # <-- FIX
                st.image(result_img_rgb, caption="Processed Image")

    elif source == "Video":
        uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()

            while cap.isOpened():
                ret, frame = cap.read() # Frame is in BGR
                if not ret:
                    break
                
                # Process the BGR frame
                processed_frame = detect_traffic_lights(frame.copy())
                # Convert the final processed frame to RGB for display
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB) # <-- FIX
                st_frame.image(processed_frame_rgb)
            
            cap.release()

    elif source == "Webcam":
        st.sidebar.write("Click the button below to start/stop the webcam feed.")
        run = st.sidebar.checkbox('Run Webcam')
        
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read() # Frame is in BGR
            if not ret:
                st.error("Failed to capture image from webcam.")
                break
            
            # Process the original BGR frame
            processed_frame = detect_traffic_lights(frame.copy())
            # Convert the final result to RGB for display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB) # <-- FIX
            FRAME_WINDOW.image(processed_frame_rgb)
        else:
            st.write('Webcam is stopped.')
            cap.release()

if __name__ == "__main__":
    main()