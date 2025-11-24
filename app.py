import av
import cv2
import time
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Real-Time Detection", layout="wide")
st.title("📷 Real-Time YOLO Detection with Collect / Ignore")

# ---------------------------
# Load YOLO Model
# ---------------------------
model = YOLO("best.pt")   # <-- your trained model path

# ---------------------------
# Session State Initialization
# ---------------------------
if "pause" not in st.session_state:
    st.session_state.pause = False
if "pause_until" not in st.session_state:
    st.session_state.pause_until = 0
if "detected_item" not in st.session_state:
    st.session_state.detected_item = None
if "collected_count" not in st.session_state:
    st.session_state.collected_count = 0
if "det_frames" not in st.session_state:
    st.session_state.det_frames = 0  # counter for stable detection

# ---------------------------
# YOLO Video Processor
# ---------------------------
TARGET_CLASSES = ["book", "bag", "notebook"]  # your model classes
CONF_THRESHOLD = 0.6  # minimum confidence
STABLE_FRAMES = 5     # frames before pausing

class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # If paused, show frame but skip detection
        if st.session_state.pause:
            if time.time() < st.session_state.pause_until:
                cv2.putText(img, "PAUSED 5 SEC", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            else:
                # Resume detection
                st.session_state.pause = False
                st.session_state.detected_item = None

        # Run YOLO inference
        results = model(img, conf=CONF_THRESHOLD)[0]

        # Filter for target classes only
        detected_labels = []
        for box in results.boxes:
            cls_id = int(box.cls)
            label = model.names[cls_id]
            if label in TARGET_CLASSES:
                detected_labels.append(label)

        # Stable detection: pause only if detected several frames
        if detected_labels:
            st.session_state.det_frames += 1
        else:
            st.session_state.det_frames = 0

        if st.session_state.det_frames >= STABLE_FRAMES and not st.session_state.pause:
            st.session_state.pause = True
            st.session_state.pause_until = time.time() + 5
            st.session_state.detected_item = detected_labels[0]
            st.session_state.det_frames = 0

        # Draw bounding boxes
        output = results.plot()
        return av.VideoFrame.from_ndarray(output, format="bgr24")

# ---------------------------
# WebRTC Streamer
# ---------------------------
webrtc_streamer(
    key="yolo-live",
    mode=WebRtcMode.SENDRECV,  # ✅ correct enum
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "facingMode": {"exact": "user"}  # back camera
        },
        "audio": False
    },
)

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.title("📌 Controls")
st.sidebar.metric("Collected Items", st.session_state.collected_count)

if st.session_state.pause and st.session_state.detected_item:
    st.sidebar.warning(f"Detected: {st.session_state.detected_item}")

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Collect"):
        st.session_state.collected_count += 1
        st.session_state.pause = False
        st.session_state.detected_item = None
        st.experimental_rerun()  # refresh UI

    if col2.button("Ignore"):
        st.session_state.pause = False
        st.session_state.detected_item = None
        st.experimental_rerun()  # refresh UI


