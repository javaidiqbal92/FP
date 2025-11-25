# import streamlit as st
# from ultralytics import YOLO
# import cv2
# import pandas as pd
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# from gtts import gTTS             # Text-to-speech
# import io                         # In-memory audio buffer
# from streamlit_autorefresh import st_autorefresh  # 🔁 auto refresh
# import time                       # ⏱ for 5-second repeat logic
# import base64                     # for embedding audio in HTML


# # --------------------------------------------
# # STREAMLIT CONFIG
# # --------------------------------------------
# st.set_page_config(page_title="Exam Proctoring", layout="wide")
# st.title("📝Cheating Material Detection Model")

# # 🔁 Auto-rerun every 5000 ms so detections + panels keep updating
# st_autorefresh(interval=5000, key="yolo-refresh")


# # --------------------------------------------
# # CLASS LABELS (prohibited items)
# # --------------------------------------------
# CLASS_MAP = {
#     0: "mobile-phone",
#     1: "calculator",
#     2: "Bag",
#     3: "Smart Watch",
#     4: "Notebook",
#     5: "Book",
#     6: "Notes",
# }
# HELPING_SET = set(CLASS_MAP.values())   # all are prohibited


# # --------------------------------------------
# # SIMPLE INSTRUCTION TEXT
# # --------------------------------------------
# def generate_llm_instructions(dets):
#     """
#     Build a simple sentence like:
#     'Phone is prohibited. Please collect it.'
#     or
#     'Phone and Bag are prohibited. Please collect them.'
#     """
#     if not dets:
#         return None

#     names = sorted({d["Class Name"] for d in dets if d["Class Name"] in HELPING_SET})
#     if not names:
#         return None

#     if len(names) == 1:
#         text = f"{names[0]} is prohibited. Please collect it."
#     else:
#         if len(names) == 2:
#             items = f"{names[0]} and {names[1]}"
#         else:
#             items = ", ".join(names[:-1]) + f" and {names[-1]}"
#         text = f"{items} are prohibited. Please collect them."

#     return text


# # --------------------------------------------
# # LOAD YOLO MODEL
# # --------------------------------------------
# @st.cache_resource
# def load_model(path):
#     return YOLO(path)

# model = load_model("best11.pt")


# def yolo_on_frame(bgr):
    
#     """
#     Run YOLO on a frame and return:
#       - annotated frame
#       - list of detections with Class ID, Class Name, Confidence, x1..y2
#     """
#     h, w, _ = bgr.shape

#     new_w = 480
#     scale = new_w / w
#     new_h = int(h * scale)

#     small = cv2.resize(bgr, (new_w, new_h))
#     results = model.predict(small, conf=0.75, iou=0.45, verbose=False)
#     res = results[0]

#     ann_small = res.plot()
#     annotated = cv2.resize(ann_small, (w, h))

#     dets = []
#     for box in res.boxes:
#         cid = int(box.cls[0])
#         conf_val = float(box.conf[0])
#         x1, y1, x2, y2 = box.xyxy[0].tolist()

#         x1 /= scale
#         y1 /= scale
#         x2 /= scale
#         y2 /= scale

#         # class name from model, then mapped to our friendly name if present
#         model_name = model.names[int(cid)] if hasattr(model, "names") else str(cid)
#         friendly_name = CLASS_MAP.get(cid, model_name)

#         dets.append({
#             "Class ID": cid,
#             "Class Name": friendly_name,
#             "Confidence": round(conf_val, 3),
#             "x1": x1, "y1": y1,
#             "x2": x2, "y2": y2
#         })

#     return annotated, dets


# # --------------------------------------------
# # SESSION STATE
# # --------------------------------------------
# if "current" not in st.session_state:
#     st.session_state.current = []

# if "counts" not in st.session_state:
#     st.session_state.counts = {name: 0 for name in CLASS_MAP.values()}

# # last instruction text + timing
# if "last_instruction_text" not in st.session_state:
#     st.session_state.last_instruction_text = None

# if "last_alert_time" not in st.session_state:
#     st.session_state.last_alert_time = 0.0

# # whether next detection is allowed to set a NEW instruction
# if "ready_for_new_instruction" not in st.session_state:
#     st.session_state.ready_for_new_instruction = True

# # store latest audio HTML (hidden auto-play)
# if "audio_html" not in st.session_state:
#     st.session_state.audio_html = ""

# # store summary DataFrame
# if "summary_df" not in st.session_state:
#     st.session_state.summary_df = None




# # --------------------------------------------
# # VIDEO PROCESSOR
# # --------------------------------------------
# class VideoProcessor(VideoTransformerBase):
    
#     def __init__(self):
#         self.last = None
#         self.latest = []
#         self.frame_id = 0

#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         self.frame_id += 1

#         if self.frame_id % 5 == 0 or self.last is None:
#             annotated, dets = yolo_on_frame(img)
#             self.last = annotated
#             self.latest = dets

#         return frame.from_ndarray(self.last, format="bgr24")


# # --------------------------------------------
# # LAYOUT
# # --------------------------------------------
# left, right = st.columns([2, 1])

# with left:
#     st.subheader("📸 Live Stream")
#     webrtc_ctx = webrtc_streamer(
#         key="stream",
#         mode=WebRtcMode.SENDRECV,
#         video_transformer_factory=VideoProcessor,
#         media_stream_constraints={"video": True, "audio": False},
#         async_processing=True,
        
#     )

# with right:
#     c1, c2 = st.columns(2)
#     with c1:
#         btn_collect = st.button("Collected", use_container_width=True)
#     with c2:
#         btn_ignore = st.button("Ignore", use_container_width=True)

#     btn_complete = st.button("Complete Scan", use_container_width=True)

#     st.subheader("📊 Current Detections")
#     box_dets = st.empty()

#     st.subheader("🧠 LLM Instructions")
#     box_llm = st.empty()

#     box_audio = st.empty()  # hidden autoplay audio

#     st.subheader("📈 Summary")
#     box_sum = st.empty()

#     msg = st.empty()

#     # Initialize session state
#     if "btn_disabled" not in st.session_state:
#         st.session_state.btn_disabled = False
    
#     btn_clear= st.button("Clear Summary",disabled=st.session_state.btn_disabled)
#     if btn_clear:
#         st.session_state.summary_df= None
#         st.session_state.btn_disabled = True

# # --------------------------------------------
# # UPDATE CURRENT DETECTIONS FROM VIDEO
# # --------------------------------------------
# if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.video_transformer:
#     latest = webrtc_ctx.video_transformer.latest
#     # ✅ Only update when YOLO gives something; keeps last stable otherwise
#     if latest is not None and len(latest) > 0:
#         st.session_state.current = latest

# dets = st.session_state.current


# # --------------------------------------------
# # CLEAR PANELS FUNCTION
# # --------------------------------------------
# def clear_panels():
#     # We do NOT clear last_instruction_text or summary_df
#     box_dets.empty()
#     box_audio.empty()
#     st.session_state.audio_html = ""
#     st.session_state.last_alert_time = 0.0
#     st.session_state.current = []
#     # Allow next detection to set a new instruction, but keep showing the old one
#     st.session_state.ready_for_new_instruction = True
    


# # --------------------------------------------
# # HANDLE BUTTONS (AND CLEAR PANELS)
# # --------------------------------------------
# skip_detection = False

# if btn_ignore:
#     clear_panels()
#     skip_detection = True
#     msg.info("Frame Ignored.")

# if btn_collect:
#     if not dets:
#         msg.warning("Nothing to collect.")
#     else:
#         for d in dets:
#             if d["Class Name"] in HELPING_SET:
#                 st.session_state.counts[d["Class Name"]] += 1
#         clear_panels()
#         skip_detection = True
#         msg.success("Collected.")

# if btn_complete:
#     # Build summary from counts and store in session_state
#     df = pd.DataFrame(
#         [{"Object": k, "Count": v} for k, v in st.session_state.counts.items()]
#     )
#     st.session_state.summary_df = df
    
#     clear_panels()
#     skip_detection = True
#     msg.success("Scan Completed.")

    


# # --------------------------------------------
# # SHOW CURRENT DETECTIONS TABLE
# # --------------------------------------------
# if not skip_detection:
#     if dets:
#         box_dets.dataframe(pd.DataFrame(dets), use_container_width=True)
#     else:
#         box_dets.info("No objects detected.")


# # --------------------------------------------
# # AUTO DETECTION MESSAGE + VOICE
# #   - LLM Instructions:
# #       set only once per "cycle" (until a button is clicked)
# #   - Audio repeats every 5s while prohibited items exist
# # --------------------------------------------
# if not skip_detection and dets:
#     helping_dets = [d for d in dets if d["Class Name"] in HELPING_SET]

#     if helping_dets:
#         # Only set instruction text when we are "ready" for a new one
#         if st.session_state.ready_for_new_instruction:
#             instruction_text = generate_llm_instructions(helping_dets)
#             if instruction_text:
#                 st.session_state.last_instruction_text = instruction_text
#                 st.session_state.ready_for_new_instruction = False

#         # Audio: repeat every 5 seconds while prohibited items are detected
#         now = time.time()
#         need_new_tts = (now - st.session_state.last_alert_time) >= 5

#         if need_new_tts and st.session_state.last_instruction_text:
#             st.session_state.last_alert_time = now

#             try:
#                 # generate TTS using the stored instruction
#                 tts = gTTS(text=st.session_state.last_instruction_text, lang="en")
#                 audio_bytes = io.BytesIO()
#                 tts.write_to_fp(audio_bytes)
#                 audio_bytes.seek(0)
#                 audio_data = audio_bytes.read()

#                 # encode as base64 and create hidden autoplay audio
#                 b64 = base64.b64encode(audio_data).decode()
#                 audio_html = f"""
#                 <audio autoplay style="display:none;">
#                     <source src="data:audio/mp3;base64,{b64}" type="audio/mpeg">
#                 </audio>
#                 """
#                 st.session_state.audio_html = audio_html
#             except Exception as e:
#                 msg.error(f"TTS Error: {e}")


# # --------------------------------------------
# # RENDER LLM INSTRUCTIONS + AUDIO
# # --------------------------------------------
# if st.session_state.last_instruction_text:
#     # ✅ This stays the same every rerun until a NEW detection cycle overwrites it
#     box_llm.markdown(st.session_state.last_instruction_text)

# if st.session_state.audio_html:
#     # hidden autoplay audio – no visible player
#     box_audio.markdown(st.session_state.audio_html, unsafe_allow_html=True)


# # --------------------------------------------
# # RENDER SUMMARY (PERSISTENT AFTER COMPLETE SCAN)
# # --------------------------------------------
# if st.session_state.summary_df is not None:
#     box_sum.table(st.session_state.summary_df)
    
import streamlit as st
from ultralytics import YOLO
import cv2
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from gtts import gTTS
import io
from streamlit_autorefresh import st_autorefresh
import time
import base64

st.set_page_config(page_title="Exam Proctoring", layout="wide")
st.title("📝Cheating Material Detection Model")
st_autorefresh(interval=5000, key="yolo-refresh")

CLASS_MAP = {
    0: "Bag",
    1: "book",
    2: "mobile",
    3: "Smart Watch",
    4: "paper",
    5: "Notebook",
    6: "Calculator",

}
HELPING_SET = set(CLASS_MAP.values())

def generate_llm_instructions(dets):
    if not dets:
        return None
    names = sorted({d["Class Name"] for d in dets if d["Class Name"] in HELPING_SET})
    if not names:
        return None
    if len(names) == 1:
        text = f"{names[0]} is prohibited. Please collect it."
    else:
        if len(names) == 2:
            items = f"{names[0]} and {names[1]}"
        else:
            items = ", ".join(names[:-1]) + f" and {names[-1]}"
        text = f"{items} are prohibited. Please collect them."
    return text

@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model("best.pt")

def yolo_on_frame(bgr):
    h, w, _ = bgr.shape
    new_w = 480
    scale = new_w / w
    new_h = int(h * scale)
    small = cv2.resize(bgr, (new_w, new_h))
    results = model.predict(small, conf=0.75, iou=0.45, verbose=False)
    res = results[0]
    ann_small = res.plot()
    annotated = cv2.resize(ann_small, (w, h))

    dets = []
    for box in res.boxes:
        cid = int(box.cls[0])
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 /= scale
        y1 /= scale
        x2 /= scale
        y2 /= scale
        model_name = model.names[int(cid)] if hasattr(model, "names") else str(cid)
        friendly_name = CLASS_MAP.get(cid, model_name)
        dets.append({
            "Class ID": cid,
            "Class Name": friendly_name,
            "Confidence": round(conf_val, 3),
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2
        })
    return annotated, dets

# ---------- session state defaults ----------
if "current" not in st.session_state:
    st.session_state.current = []
if "counts" not in st.session_state:
    st.session_state.counts = {name: 0 for name in CLASS_MAP.values()}
if "last_instruction_text" not in st.session_state:
    st.session_state.last_instruction_text = None
if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = 0.0
if "ready_for_new_instruction" not in st.session_state:
    st.session_state.ready_for_new_instruction = True
if "audio_html" not in st.session_state:
    st.session_state.audio_html = ""
if "summary_df" not in st.session_state:
    st.session_state.summary_df = None
if "btn_disabled" not in st.session_state:
    st.session_state.btn_disabled = False
# streaming flag: True means show/start stream, False stops/hides it
if "streaming" not in st.session_state:
    st.session_state.streaming = True

# ---------- Video processor ----------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.last = None
        self.latest = []
        self.frame_id = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_id += 1

        if self.frame_id % 5 == 0 or self.last is None:
            annotated, dets = yolo_on_frame(img)
            self.last = annotated
            self.latest = dets

        return frame.from_ndarray(self.last, format="bgr24")

# ensure webrtc_ctx exists in this run (set None if not created)
webrtc_ctx = None

# ---------- Layout ----------
left, right = st.columns([2, 1])

with left:
    st.subheader("📸 Live Stream")
    if st.session_state.streaming:
        # create the stream if streaming flag is True
        webrtc_ctx = webrtc_streamer(
            key="stream",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    else:
        st.info("Stream stopped. Click 'Start Stream' to resume scanning.")
        # show a button to restart streaming if you want
        if st.button("Start Stream"):
            st.session_state.streaming = True
            st.rerun()

with right:
    c1, c2 = st.columns(2)
    with c1:
        btn_collect = st.button("Collected", use_container_width=True)
    with c2:
        btn_ignore = st.button("Ignore", use_container_width=True)

    btn_complete = st.button("Complete Scan", use_container_width=True)

    st.subheader("📊 Current Detections")
    box_dets = st.empty()

    st.subheader("🧠 LLM Instructions")
    box_llm = st.empty()

    box_audio = st.empty()

    st.subheader("📈 Summary")
    box_sum = st.empty()

    msg = st.empty()

    btn_clear = st.button("Clear Summary", disabled=st.session_state.btn_disabled)
    if btn_clear:
        st.session_state.summary_df = None
        st.session_state.btn_disabled = True

# ---------- UPDATE CURRENT DETECTIONS FROM VIDEO ----------
if webrtc_ctx and webrtc_ctx.video_transformer and hasattr(webrtc_ctx.video_transformer, "latest"):
    latest = webrtc_ctx.video_transformer.latest
    if latest is not None and len(latest) > 0:
        st.session_state.current = latest
dets = st.session_state.current

# ---------- helper: clear panels ----------
def clear_panels():
    box_dets.empty()
    box_audio.empty()
    st.session_state.audio_html = ""
    st.session_state.last_alert_time = 0.0
    st.session_state.current = []
    st.session_state.ready_for_new_instruction = True

# ---------- HANDLE BUTTONS ----------
skip_detection = False

if btn_ignore:
    clear_panels()
    skip_detection = True
    msg.info("Frame Ignored.")

if btn_collect:
    if not dets:
        msg.warning("Nothing to collect.")
    else:
        for d in dets:
            if d["Class Name"] in HELPING_SET:
                st.session_state.counts[d["Class Name"]] += 1
        clear_panels()
        skip_detection = True
        msg.success("Collected.")

if btn_complete:
    # Build summary from counts and store in session_state
    df = pd.DataFrame([{"Object": k, "Count": v} for k, v in st.session_state.counts.items()])
    st.session_state.summary_df = df

    # STOP video stream robustly:
    # 1) set streaming flag False so future reruns won't re-create the streamer
    st.session_state.streaming = False

    # 2) attempt to stop the live webrtc context if it exists and is playing
    try:
        if webrtc_ctx is not None and getattr(webrtc_ctx, "state", None) is not None:
            if webrtc_ctx.state.playing:
                webrtc_ctx.stop()
    except Exception as e:
        # don't crash — just log a message to the UI
        msg.warning(f"Could not call webrtc_ctx.stop(): {e}")

    clear_panels()
    skip_detection = True
    msg.success("Scan Completed. Video stopped.")

# ---------- SHOW CURRENT DETECTIONS TABLE ----------
if not skip_detection:
    if dets:
        box_dets.dataframe(pd.DataFrame(dets), use_container_width=True)
    else:
        box_dets.info("No objects detected.")

# ---------- AUTO DETECTION MESSAGE + VOICE ----------
if not skip_detection and dets:
    helping_dets = [d for d in dets if d["Class Name"] in HELPING_SET]
    if helping_dets:
        if st.session_state.ready_for_new_instruction:
            instruction_text = generate_llm_instructions(helping_dets)
            if instruction_text:
                st.session_state.last_instruction_text = instruction_text
                st.session_state.ready_for_new_instruction = False

        now = time.time()
        need_new_tts = (now - st.session_state.last_alert_time) >= 5

        if need_new_tts and st.session_state.last_instruction_text:
            st.session_state.last_alert_time = now
            try:
                tts = gTTS(text=st.session_state.last_instruction_text, lang="en")
                audio_bytes = io.BytesIO()
                tts.write_to_fp(audio_bytes)
                audio_bytes.seek(0)
                audio_data = audio_bytes.read()
                b64 = base64.b64encode(audio_data).decode()
                audio_html = f"""
                <audio autoplay style="display:none;">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mpeg">
                </audio>
                """
                st.session_state.audio_html = audio_html
            except Exception as e:
                msg.error(f"TTS Error: {e}")

# ---------- RENDER LLM INSTRUCTIONS + AUDIO ----------
if st.session_state.last_instruction_text:
    box_llm.markdown(st.session_state.last_instruction_text)
if st.session_state.audio_html:
    box_audio.markdown(st.session_state.audio_html, unsafe_allow_html=True)

# ---------- RENDER SUMMARY ----------
if st.session_state.summary_df is not None:
    box_sum.table(st.session_state.summary_df)
