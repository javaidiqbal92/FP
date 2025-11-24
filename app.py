import cv2
import time
import numpy as np
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

st.set_page_config(page_title="YOLO Item Collector", layout="wide")
st.title("🎥 YOLO Video Recorder with Collect / Ignore")

# ------------------------
# Load your YOLO model
# ------------------------
model = YOLO("best.pt")  # replace with your trained model

# ------------------------
# Initialize session state
# ------------------------
if "recording" not in st.session_state:
    st.session_state.recording = False

if "pause" not in st.session_state:
    st.session_state.pause = False

if "pause_until" not in st.session_state:
    st.session_state.pause_until = 0

if "detected_item" not in st.session_state:
    st.session_state.detected_item = None

if "collected_count" not in st.session_state:
    st.session_state.collected_count = {}

if "det_frames" not in st.session_state:
    st.session_state.det_frames = 0

# ------------------------
# YOLO Processor
# ------------------------
TARGET_CLASSES = ["book", "bag", "notebook"]
CONF_THRESHOLD = 0.6
STABLE_FRAMES = 5  # number of frames detection must be stable to pause

class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Skip detection if paused
        if st.session_state.pause:
            if time.time() < st.session_state.pause_until:
                cv2.putText(img, "PAUSED", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            else:
                st.session_state.pause = False
                st.session_state.detected_item = None

        # Only process if recording
        if st.session_state.recording:
            results = model(img, conf=CONF_THRESHOLD)[0]
            detected_labels = []
            for box in results.boxes:
                cls_id = int(box.cls)
                label = model.names[cls_id]
                if label in TARGET_CLASSES:
                    detected_labels.append(label)

            # Stable detection
            if detected_labels:
                st.session_state.det_frames += 1
            else:
                st.session_state.det_frames = 0

            if st.session_state.det_frames >= STABLE_FRAMES and not st.session_state.pause:
                st.session_state.pause = True
                st.session_state.pause_until = time.time() + 5
                st.session_state.detected_item = detected_labels[0]
                st.session_state.det_frames = 0

            output = results.plot()
        else:
            output = img

        return av.VideoFrame.from_ndarray(output, format="bgr24")

# ------------------------
# WebRTC Stream
# ------------------------
webrtc_streamer(
    key="yolo-collector",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={
        "video": {"facingMode": {"exact": "environment"}},
        "audio": False
    },
)

# ------------------------
# Sidebar Controls
# ------------------------
st.sidebar.title("Controls")
start = st.sidebar.button("Start Recording")
stop = st.sidebar.button("Stop Recording")

if start:
    st.session_state.recording = True

if stop:
    st.session_state.recording = False
    st.session_state.pause = False
    st.session_state.detected_item = None

st.sidebar.metric("Collected Items", sum(st.session_state.collected_count.values()))

# Show collect / ignore buttons if paused
if st.session_state.pause and st.session_state.detected_item:
    st.sidebar.warning(f"Detected: {st.session_state.detected_item}")
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Collect"):
        cls_name = st.session_state.detected_item
        st.session_state.collected_count[cls_name] = st.session_state.collected_count.get(cls_name,0)+1
        st.session_state.pause = False
        st.session_state.detected_item = None

    if col2.button("Ignore"):
        st.session_state.pause = False
        st.session_state.detected_item = None
