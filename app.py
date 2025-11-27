import streamlit as st
from ultralytics import YOLO
import cv2
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from gtts import gTTS
import io, base64, time

st.set_page_config(page_title="Exam Proctoring", layout="wide")
st.title("📝Cheating Material Detection Model")

# -- small CLASS_MAP for example, keep your own --
CLASS_MAP = {0: "Bag", 1: "book", 2: "mobile", 3: "Smart Watch", 4: "paper", 5: "Notebook", 6: "Calculator"}
HELPING_SET = set(CLASS_MAP.values())

@st.cache_resource
def load_model(path="best.pt"):
    return YOLO(path)

model = load_model()

def yolo_on_frame(bgr):
    h, w, _ = bgr.shape
    new_w = 480
    scale = new_w / w
    small = cv2.resize(bgr, (new_w, int(h * scale)))
    results = model.predict(small, conf=0.75, iou=0.45, verbose=False)
    res = results[0]
    ann_small = res.plot()
    annotated = cv2.resize(ann_small, (w, h))
    dets = []
    for box in res.boxes:
        cid = int(box.cls[0])
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 /= scale; y1 /= scale; x2 /= scale; y2 /= scale
        model_name = model.names[int(cid)] if hasattr(model, "names") else str(cid)
        friendly_name = CLASS_MAP.get(cid, model_name)
        dets.append({"Class ID": cid, "Class Name": friendly_name, "Confidence": round(conf_val,3), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return annotated, dets

# Session defaults
if "current" not in st.session_state:
    st.session_state.current = []

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.last = None
        self.latest = []
        self.frame_id = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_id += 1
        # only run detection every 5 frames
        if self.frame_id % 5 == 0 or self.last is None:
            annotated, dets = yolo_on_frame(img)
            self.last = annotated
            self.latest = dets
        return frame.from_ndarray(self.last, format="bgr24")

left, right = st.columns([2,1])

RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

with left:
    st.subheader("📸 Live Stream")
    webrtc_ctx = webrtc_streamer(
        key="stream",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,              # <-- simpler while debugging
        rtc_configuration=RTC_CONFIGURATION,
        desired_playing_state=True           # try to ensure player starts
    )

# read latest detections from transformer (if exists)
if webrtc_ctx and webrtc_ctx.video_transformer:
    latest = webrtc_ctx.video_transformer.latest
    if latest and len(latest) > 0:
        st.session_state.current = latest

with right:
    st.subheader("Detections")
    if st.session_state.current:
        st.dataframe(pd.DataFrame(st.session_state.current))
    else:
        st.info("No objects detected yet.")
