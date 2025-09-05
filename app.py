import streamlit as st
import cv2
import numpy as np
import os
from detect import detect_traffic_lights
from utils import draw_label
import tempfile

def main():
    st.title("🚦 Traffic Light Detection System")
    
    # Create a placeholder for the status message
    status_placeholder = st.empty()
    
    st.sidebar.title("Options")
    source = st.sidebar.radio("Select Source", ("Image", "Video", "Webcam"))

    if source == "Image":
        uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            img_rgb_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.sidebar.image(img_rgb_display, caption="Uploaded Image")

            if st.sidebar.button("Detect Traffic Lights"):
                # Get both the frame and the state
                result_img, light_state = detect_traffic_lights(img.copy())
                
                # Display the command based on the state
                if light_state == "Red":
                    status_placeholder.error("🔴 STOP")
                elif light_state == "Yellow":
                    status_placeholder.warning("🟡 GET READY")
                elif light_state == "Green":
                    status_placeholder.success("🟢 GO!")
                else:
                    status_placeholder.info("⚪ No light detected")

                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_img_rgb, caption="Processed Image")

    elif source == "Video" or source == "Webcam":
        run = True
        if source == "Webcam":
            run = st.sidebar.checkbox('Run Webcam', value=True)
            cap = cv2.VideoCapture(0)
        else: # Video
            uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
            if uploaded_file:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                cap = cv2.VideoCapture(tfile.name)
            else:
                run = False
        
        FRAME_WINDOW = st.image([])
        
        while run and 'cap' in locals() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Video finished or failed to capture frame.")
                break
            
            # Get both the frame and the state
            processed_frame, light_state = detect_traffic_lights(frame.copy())
            
            # Display the command based on the state
            if light_state == "Red":
                status_placeholder.error("🔴 STOP")
            elif light_state == "Yellow":
                status_placeholder.warning("🟡 GET READY")
            elif light_state == "Green":
                status_placeholder.success("🟢 GO!")
            else:
                status_placeholder.info("⚪ No light detected")
                
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(processed_frame_rgb)
        
        if not run and source == "Webcam":
            st.write('Webcam is stopped.')
        
        if 'cap' in locals() and cap.isOpened():
            cap.release()

if __name__ == "__main__":
    main()