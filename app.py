import streamlit as st
import cv2
import numpy as np
from detect import detect_traffic_lights
from utils import draw_status
# --- FIX 1: Add WebRtcMode to the import ---
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import tempfile
import av

# RTC Configuration for STUN servers
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def main():
    st.title("🚦 Traffic Light Detection System")
    st.sidebar.title("Options")
    
    source = st.sidebar.radio("Select Source", ("Image", "Video", "Live Webcam"))

    if source == "Image":
        handle_image_upload()
    elif source == "Video":
        handle_video_upload()
    elif source == "Live Webcam":
        handle_webrtc()

def handle_image_upload():
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        img_rgb_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.sidebar.image(img_rgb_display, caption="Uploaded Image")

        if st.sidebar.button("Detect Traffic Lights"):
            result_img, light_state = detect_traffic_lights(img.copy())
            result_img_with_status = draw_status(result_img, light_state)
            result_img_rgb = cv2.cvtColor(result_img_with_status, cv2.COLOR_BGR2RGB)
            st.image(result_img_rgb, caption="Processed Image")

def handle_video_upload():
    uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        FRAME_WINDOW = st.image([])
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Video finished.")
                break
            
            processed_frame, light_state = detect_traffic_lights(frame.copy())
            processed_frame_with_status = draw_status(processed_frame, light_state)
            processed_frame_rgb = cv2.cvtColor(processed_frame_with_status, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(processed_frame_rgb)
        
        cap.release()

class TrafficLightTransformer(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        
        # Process the frame to detect traffic lights
        processed_img, light_state = detect_traffic_lights(img)
        
        # Draw the status message on the frame
        final_img = draw_status(processed_img, light_state)
        
        return final_img

def handle_webrtc():
    st.sidebar.info("Click 'START' to activate your camera. The app will process the video stream in real-time.")
    
    webrtc_ctx = webrtc_streamer(
        key="traffic-light-detection",
        # --- FIX 2: Use WebRtcMode directly ---
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=TrafficLightTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if webrtc_ctx.state.playing:
        st.success("Webcam is running.")
    else:
        st.info("Webcam is stopped.")

if __name__ == "__main__":
    main()