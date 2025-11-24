# import streamlit as st
# import av
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
# import cv2
# import threading


# # ------------------ Session State ------------------
# if "recording" not in st.session_state:
#     st.session_state.recording = False

# if "streaming" not in st.session_state:
#     st.session_state.streaming = False
# if "video_writer" not in st.session_state:
#     st.session_state.video_writer = None

# # ------------------ Start/Stop Functions ------------------
# def start_recording():
#     st.session_state.recording = True
#     st.session_state.streaming = True
    
#     st.success("Recording started!")

# def stop_recording():
#     st.session_state.recording = False
#     if st.session_state.video_writer is not None:
#         st.session_state.video_writer.release()
#         st.session_state.video_writer = None
#     st.success("Recording stopped!")

# # Set page configuration
# st.set_page_config(page_title="Three Pane Layout", layout="wide")

# # Title
# st.title("Detection Application")

# # Create three columns (panes)
# col1, col2, col3 = st.columns(3)

# # Pane 1 content
# with col1:
#     st.header("Control")
#     st.button("Start Recording", on_click=start_recording)
#     st.button("Stop Recording", on_click=stop_recording)

# # Pane 2 content
# with col2:
#     class VideoProcessor(VideoProcessorBase):
#         def __init__(self):
#             self.frame_shape = None

#         def recv(self, frame):
#             img = frame.to_ndarray(format="bgr24")

#             # Only save frames if recording is True
#             if st.session_state.recording:
#                 if st.session_state.video_writer is None:
#                     self.frame_shape = img.shape
#                     fourcc = cv2.VideoWriter_fourcc(*"XVID")
#                     st.session_state.video_writer = cv2.VideoWriter(
#                         "output.avi", fourcc, 20.0, (self.frame_shape[1], self.frame_shape[0])
#                     )
#                 st.session_state.video_writer.write(img)

#             return av.VideoFrame.from_ndarray(img, format="bgr24")

#     # Only launch webcam if streaming is True

#     st.set_page_config(page_title="Webcam Stream Demo", layout="wide")
#     st.header("Webcam Stream")
#     if st.session_state.streaming:
#         webrtc_streamer(
#             key="webcam",
#             video_processor_factory=VideoProcessor,
#             media_stream_constraints={"video": True, "audio": False},
#             async_processing=True,
#         )
#     else:
#         st.info("Click 'Start Recording' to open webcam")
   
# # Pane 3 content
# with col3:
#     st.header("Statistics")
#     if st.session_state.recording:
#         st.info("Recording: ON")
#     else:
#         st.info("Recording: OFF")

import av
import cv2
import time
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Real-Time Detection", layout="wide")
st.title("📷 Real-Time YOLO Detection with Collect / Ignore")

# Load your YOLO model
model = YOLO("best.pt")   # <-- put your model path here

# Initialize session state values
if "pause" not in st.session_state:
    st.session_state.pause = False

if "pause_until" not in st.session_state:
    st.session_state.pause_until = 0

if "detected_item" not in st.session_state:
    st.session_state.detected_item = None

if "collected_count" not in st.session_state:
    st.session_state.collected_count = 0


# --------------------
# VIDEO PROCESSOR
# --------------------
class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Check if paused
        if st.session_state.pause:
            if time.time() < st.session_state.pause_until:
                # Still paused → show "Paused" on screen
                cv2.putText(img, "PAUSED 5 SEC", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            else:
                # Resume
                st.session_state.pause = False
                st.session_state.detected_item = None

        # Run YOLO inference
        results = model(img)[0]
        boxes = results.boxes

        # Draw boxes
        output = results.plot()

        # Check if detection found
        if len(boxes) > 0 and not st.session_state.pause:
            cls_id = int(boxes[0].cls)
            label = model.names[cls_id]

            # Pause detection for 5 seconds
            st.session_state.pause = True
            st.session_state.pause_until = time.time() + 5
            st.session_state.detected_item = label

        return av.VideoFrame.from_ndarray(output, format="bgr24")


# --------------------
# WEBRTC STREAM
# --------------------
webrtc_streamer(
    key="yolo-live",
    mode="sendrecv",
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# --------------------
# SIDE PANEL UI
# --------------------
st.sidebar.title("📌 Controls")

st.sidebar.metric("Collected Items", st.session_state.collected_count)

if st.session_state.pause and st.session_state.detected_item:
    st.sidebar.warning(f"Detected: {st.session_state.detected_item}")

    col1, col2 = st.sidebar.columns(2)

    if col1.button("Collect"):
        st.session_state.collected_count += 1
        st.session_state.pause = False
        st.session_state.detected_item = None

    if col2.button("Ignore"):
        st.session_state.pause = False
        st.session_state.detected_item = None

