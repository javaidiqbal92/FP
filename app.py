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
import streamlit as st
from ultralytics import YOLO
import cv2
import time
import numpy as np
from PIL import Image

# ------------------------- CONFIG --------------------------
MODEL_PATH = "best.pt"  # <-- your YOLOv11 model path
TARGET_CLASSES = ["mobile-phone"]
# ------------------------------------------------------------

# Initialize model
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# Page configuration
st.set_page_config(
    page_title="YOLOv11 Detection App",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
def initialize_session_state():
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'pause_detection' not in st.session_state:
        st.session_state.pause_detection = False
    if 'paused_frame' not in st.session_state:
        st.session_state.paused_frame = None
    if 'timer_running' not in st.session_state:
        st.session_state.timer_running = False
    if 'timer_start_time' not in st.session_state:
        st.session_state.timer_start_time = 0
    if 'countdown_value' not in st.session_state:
        st.session_state.countdown_value = 5
    if 'collected_items' not in st.session_state:
        st.session_state.collected_items = {}
    if 'detected_class' not in st.session_state:
        st.session_state.detected_class = ""
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    if 'last_frame' not in st.session_state:
        st.session_state.last_frame = None
    if 'video_placeholder_key' not in st.session_state:
        st.session_state.video_placeholder_key = 0

initialize_session_state()

# ------------------ STREAMLIT UI SETUP -------------------

# Sidebar (Left Pane equivalent)
with st.sidebar:
    st.title("🎯 Detection Controls")
    
    st.subheader("Recording Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎥 Start Recording", use_container_width=True, type="primary"):
            st.session_state.is_recording = True
            st.session_state.cap = cv2.VideoCapture(0)
            # Set camera properties for better performance
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            st.session_state.cap.set(cv2.CAP_PROP_FPS, 30)
    with col2:
        if st.button("⏹️ Stop Recording", use_container_width=True, type="secondary"):
            st.session_state.is_recording = False
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None
    
    st.divider()
    
    st.subheader("📊 Collected Items")
    if st.session_state.collected_items:
        for item, count in st.session_state.collected_items.items():
            st.write(f"**{item}** : {count}")
    else:
        st.write("No items collected yet")
    
    # Clear items button
    if st.button("🗑️ Clear All Items", use_container_width=True):
        st.session_state.collected_items = {}

# Main content area
st.title("📱 YOLOv11 Object Detection")
st.markdown("---")

# Create a stable container for video
video_container = st.container()

with video_container:
    # Use a unique key for the video placeholder to prevent recreation
    video_placeholder = st.empty()
    
    # Display the last frame if available
    if st.session_state.last_frame is not None:
        video_placeholder.image(st.session_state.last_frame, use_column_width=True, channels="BGR")

# Collect/Ignore buttons in a separate container
button_container = st.container()
with button_container:
    if st.session_state.pause_detection:
        st.warning(f"📱 **{st.session_state.detected_class} detected!** Choose an action:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Collect", key="collect_btn", use_container_width=True, type="primary"):
                # Collect item logic
                cls_name = st.session_state.detected_class
                if cls_name not in st.session_state.collected_items:
                    st.session_state.collected_items[cls_name] = 1
                else:
                    st.session_state.collected_items[cls_name] += 1
                
                st.session_state.pause_detection = False
                st.session_state.timer_running = False
                st.success(f"✅ {cls_name} collected!")
        with col2:
            if st.button("❌ Ignore", key="ignore_btn", use_container_width=True, type="secondary"):
                # Ignore item logic
                st.session_state.pause_detection = False
                st.session_state.timer_running = False
                st.info("⏭️ Detection continued")

# Status indicator - use columns for stable layout
status_container = st.container()
with status_container:
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        status = "🔴 Recording" if st.session_state.is_recording else "⏸️ Stopped"
        st.metric("Status", status)
    with status_col2:
        detection_status = "⏸️ Paused" if st.session_state.pause_detection else "🔍 Detecting"
        st.metric("Detection", detection_status)
    with status_col3:
        if st.session_state.pause_detection:
            st.metric("Countdown", f"{st.session_state.countdown_value}s")
        else:
            st.metric("Countdown", "0s")

# ------------------ TIMER HANDLING -------------------------
def start_timer():
    st.session_state.timer_running = True
    st.session_state.timer_start_time = time.time()
    st.session_state.countdown_value = 5

def update_countdown():
    if st.session_state.timer_running:
        elapsed = time.time() - st.session_state.timer_start_time
        st.session_state.countdown_value = max(0, 5 - int(elapsed))
        
        if st.session_state.countdown_value <= 0:
            st.session_state.timer_running = False
            st.session_state.pause_detection = False

# ------------------ VIDEO PROCESSING -----------------------
def process_video_frame():
    if not st.session_state.is_recording or st.session_state.cap is None:
        return
    
    # Update countdown if timer is running
    if st.session_state.timer_running:
        update_countdown()
    
    # Read frame from camera
    ret, frame = st.session_state.cap.read()
    if not ret:
        st.error("❌ Failed to read from camera")
        st.session_state.is_recording = False
        return
    
    # Store the frame for persistent display
    display_frame = frame.copy()
    
    # Process frame based on detection state
    if st.session_state.pause_detection:
        # Use paused frame and overlay countdown
        if st.session_state.paused_frame is not None:
            display_frame = st.session_state.paused_frame.copy()
            # Draw countdown on frame
            cv2.putText(display_frame, f"Wait: {st.session_state.countdown_value}s", 
                       (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(display_frame, f"Detected: {st.session_state.detected_class}", 
                       (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        # Run YOLO detection
        results = model(frame, stream=True)
        detection_made = False
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                cls_name = model.names[cls]
                
                if cls_name in TARGET_CLASSES:
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{cls_name} {conf:.2f}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Pause detection if target class found
                    if not st.session_state.pause_detection:
                        st.session_state.pause_detection = True
                        st.session_state.paused_frame = frame.copy()
                        st.session_state.detected_class = cls_name
                        start_timer()
                        detection_made = True
                        break
            
            if detection_made:
                break
    
    # Update the video placeholder
    video_placeholder.image(display_frame, use_column_width=True, channels="BGR")
    
    # Store the last frame to prevent blinking
    st.session_state.last_frame = display_frame

# ------------------ MAIN APPLICATION LOGIC -----------------
# Process video frames if recording
if st.session_state.is_recording:
    process_video_frame()
    
    # Use Streamlit's automatic rerun with a delay
    time.sleep(0.03)  # ~30 FPS
    st.rerun()
else:
    # Show placeholder when not recording
    if st.session_state.last_frame is None:
        video_placeholder.info("👆 Click **Start Recording** to begin detection")

# Cleanup when app stops
if not st.session_state.is_recording and st.session_state.cap:
    st.session_state.cap.release()
    st.session_state.cap = None

# Instructions
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    1. Click **Start Recording** to begin webcam detection
    2. The app will detect objects using YOLOv11
    3. When a target object is detected, detection will pause
    4. Choose **Collect** to count the item or **Ignore** to continue
    5. Collected items appear in the sidebar
    6. Click **Stop Recording** to end the session
    """)

st.markdown("---")
st.caption("YOLOv11 Object Detection App | Built with Streamlit")