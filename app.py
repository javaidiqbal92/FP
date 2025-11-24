"""
Streamlit Pre-Exam Proctoring App
Single-file Streamlit app that:
- Runs a YOLO model (Ultralytics) on webcam/video stream
- When prohibited items are detected, uses an LLM (user-configurable) to produce an instruction
- Uses gTTS (or pyttsx3 locally) to speak the instruction
- Provides UI buttons: Ignore, Collected, Examination Completed

USAGE
- Put your trained YOLO weights (best.pt) path in MODEL_PATH
- Ensure that your YOLO model meets mAP > 0.85 on validation as requested
- Set environment variables or edit config for LLM usage (GOOGLE_API_KEY or other)
- Run: streamlit run streamlit_proctoring.py

LIMITATIONS / NOTES
- Browser autoplay of audio is often blocked. This app includes two TTS modes:
  1) SERVER_TTS (gTTS + st.audio) which embeds audio in the page (may require manual play)
  2) LOCAL_TTS (pyttsx3) which speaks on the machine running the Streamlit server (works if app runs locally)
- Replace LLM placeholder code with your preferred LLM client (Google Gen AI SDK, OpenAI, etc.).

Dependencies (pip)
ultralytics
streamlit
opencv-python
numpy
gTTS
pyttsx3 (optional, local TTS)
Pillow

"""

import os
import time
import io
import tempfile
from typing import List, Dict, Any

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# YOLO model
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# TTS
try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# ----------------------
# CONFIG / USER EDIT
# ----------------------
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "best.pt")  # path to trained weights
CONF_THRESH = float(os.getenv("CONF_THRESH", "0.85"))  # detection confidence threshold
IOU_THRESH = float(os.getenv("IOU_THRESH", "0.75"))

# Which classes are considered prohibited
PROHIBITED_CLASSES = [
    "mobile-phone"
]

# TTS mode: 'server' uses gTTS + embedded audio, 'local' uses pyttsx3 (speaks on server)
TTS_MODE = os.getenv("TTS_MODE", "server")

# LLM config placeholder
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google_genai")  # or 'openai'
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")  # if using Google Gen AI

# ----------------------
# Helper functions
# ----------------------

def load_model(path: str):
    if YOLO is None:
        raise RuntimeError("ultralytics not installed. `pip install ultralytics` to use YOLO inference.")
    if not os.path.exists(path):
        st.warning(f"Model weights not found at {path}. Please provide a valid path.")
        return None
    model = YOLO(path)
    return model


@st.cache_resource
def init_tts_engine():
    if pyttsx3 is None:
        return None
    engine = pyttsx3.init()
    return engine


def speak_text_local(text: str):
    engine = init_tts_engine()
    if engine is None:
        st.error("pyttsx3 not available. Install pyttsx3 for local TTS (pip install pyttsx3)")
        return
    engine.say(text)
    engine.runAndWait()


def synthesize_gtts_audio(text: str) -> bytes:
    """Return mp3 audio bytes using gTTS"""
    if gTTS is None:
        raise RuntimeError("gTTS not installed. `pip install gTTS` to enable server-side TTS.")
    tts = gTTS(text=text, lang="en")
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()


# Placeholder LLM wrapper. Replace internals with real API calls (Google GenAI, OpenAI, etc.)
def generate_instruction(detected_class: str) -> str:
    """Generate a short instruction string given the detected class.
    This function contains a simple local template fallback if LLM credentials are not provided.
    Replace the body with an actual LLM call (Google GenAI or other) if desired.
    """
    # If user provided Google API key, you would call the Google Gen AI SDK here.
    # Example (pseudocode):
    # from google.generativeai import client
    # client = GoogleClient(api_key=GOOGLE_API_KEY)
    # resp = client.generate_text(...)

    # Fallback deterministic templates (safe, no external calls)
    templates = {
        "Mobile-phone": "Prohibited: 'Mobile' is not allowed. Please remove it from the room immediately.",
        "backpack": "Prohibited: 'Backpack' is not allowed during the exam. Please remove it from the room.",
        "Calculator": "Prohibited: 'Calculator' is not allowed. Place it outside the exam area now.",
        "book": "Prohibited: 'Book' is not allowed during the exam. Please remove it from the desk.",
        "Notebook": "Prohibited: 'Notebook' is not allowed. Please remove it now.",
        "paper": "Prohibited: 'Notes/Paper' detected. Please clear the desk.",
        "smart watches - v1": "Prohibited: 'Smartwatch' is not allowed. Please remove it now.",
    }
    return templates.get(detected_class, f"Prohibited: '{detected_class}' is not allowed.")


# Utility to parse ultralytics results to a list of detections
def parse_results(results) -> List[Dict[str, Any]]:
    dets = []
    # results is a list (per frame) of Results objects
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls = int(box.cls.cpu().numpy()[0]) if hasattr(box, "cls") else int(box.cls)
            conf = float(box.conf.cpu().numpy()[0]) if hasattr(box, "conf") else float(box.conf)
            xyxy = box.xyxy.cpu().numpy()[0] if hasattr(box, "xyxy") else box.xyxy
            dets.append({
                "class_id": cls,
                "conf": conf,
                "xyxy": xyxy.tolist() if hasattr(xyxy, "tolist") else list(xyxy),
            })
    return dets


# ----------------------
# Streamlit UI / App
# ----------------------

st.set_page_config(page_title="Pre-Exam Proctoring", layout="wide")
st.title("Pre-Exam Proctoring — Live Detection of Prohibited Items")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("YOLO model path", MODEL_PATH)
    conf = st.slider("Confidence threshold", 0.0, 1.0, float(CONF_THRESH), 0.01)
    tts_mode = st.radio("TTS mode (audio playback)", ["server", "local"], index=0 if TTS_MODE == "server" else 1)
    start_camera = st.button("Start Camera")
    stop_camera = st.button("Stop Camera")
    st.markdown("---")
    st.caption("LLM integration: set GOOGLE_API_KEY as env var to enable Google Gen AI calls in generate_instruction().")

# Session state for detections
if "collected" not in st.session_state:
    st.session_state.collected = []  # list of dicts {class, time, conf}
if "ignored" not in st.session_state:
    st.session_state.ignored = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

# Load model button
if st.button("Load Model"):
    st.info(f"Loading model from {model_path} ...")
    model = load_model(model_path)
    if model is not None:
        st.success("Model loaded. Make sure your model has mAP > 0.85 on validation for production.")
        st.session_state.model = model

# Main columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Camera Feed")
    FRAME_WINDOW = st.image([])

with col2:
    st.subheader("Detections")
    st.write("Pending detection:")
    pending_box = st.empty()
    st.write("Collected items:")
    collected_box = st.empty()

# Buttons for pending detection actions
ignore_btn = st.button("Ignore")
collect_btn = st.button("Collected")
exam_done_btn = st.button("Examination Completed")

# Start / stop camera logic
if start_camera:
    st.session_state.camera_running = True
if stop_camera:
    st.session_state.camera_running = False

# OpenCV video capture (webcam)
cap = None
if st.session_state.camera_running:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam. Make sure a webcam is connected and accessible.")
        st.session_state.camera_running = False

# Main loop (pull frames and run inference)
if st.session_state.camera_running and hasattr(st.session_state, "model"):
    model = st.session_state.model
    try:
        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read frame from camera")
                break

            # Resize for speed (maintain aspect ratio) - model will re-scale internally
            h, w = frame.shape[:2]
            max_dim = 1280
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            # Convert BGR->RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run inference
            results = model.track(source=img, conf=conf, persist=False) if hasattr(model, "track") else model(img, conf=conf)

            # Parse detections
            detections = []
            # ultralytics returns a Results object or list; we handle common cases
            try:
                for r in results:
                    if r.boxes is None:
                        continue
                    for b in r.boxes:
                        cls_id = int(b.cls.cpu().numpy()[0]) if hasattr(b, "cls") else int(b.cls)
                        conf_v = float(b.conf.cpu().numpy()[0]) if hasattr(b, "conf") else float(b.conf)
                        xyxy = b.xyxy.cpu().numpy()[0] if hasattr(b, "xyxy") else b.xyxy
                        label = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
                        detections.append({"class": label, "conf": conf_v, "xyxy": xyxy.tolist()})
            except Exception:
                # fallback: try simpler parse
                try:
                    for r in results:
                        for d in r:
                            detections.append(d)
                except Exception:
                    pass

            # Draw boxes on frame for visualization
            vis = img.copy()
            for d in detections:
                x1, y1, x2, y2 = map(int, d["xyxy"][:4])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(vis, f"{d['class']} {d['conf']:.2f}", (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            FRAME_WINDOW.image(vis)

            # Check for prohibited items
            found = None
            for d in detections:
                if d["class"] in PROHIBITED_CLASSES and d["conf"] >= conf:
                    found = d
                    break

            if found is not None:
                # If new pending or different from previous
                prev = st.session_state.get("pending")
                if (prev is None) or (prev and prev.get("class") != found["class"]):
                    st.session_state.pending = {"class": found["class"], "conf": found["conf"], "time": time.time()}
                    # Generate instruction via LLM or template
                    instruction = generate_instruction(found["class"])
                    st.session_state.pending["instruction"] = instruction

                    # TTS
                    if tts_mode == "local" and pyttsx3 is not None:
                        speak_text_local(instruction)
                    else:
                        # Generate mp3 and show audio player (note: autoplay may be blocked by browser)
                        try:
                            audio_bytes = synthesize_gtts_audio(instruction)
                            st.audio(audio_bytes, format="audio/mp3")
                        except Exception as e:
                            st.warning("TTS failed: " + str(e))

                pending_box.json(st.session_state.pending)
            else:
                pending_box.write("No prohibited items detected.")
                st.session_state.pending = None

            # Handle button presses
            if ignore_btn and st.session_state.pending:
                st.session_state.ignored.append(st.session_state.pending)
                st.session_state.pending = None
                ignore_btn = False

            if collect_btn and st.session_state.pending:
                st.session_state.collected.append(st.session_state.pending)
                st.session_state.pending = None
                collect_btn = False

            # Show collected items
            if len(st.session_state.collected) > 0:
                collected_box.json(st.session_state.collected)
            else:
                collected_box.write("No items collected yet.")

            if exam_done_btn:
                # Show final report
                st.success("Examination completed. Final collected items returned below.")
                st.json(st.session_state.collected)
                # reset
                st.session_state.camera_running = False
                break

            # small sleep to yield
            time.sleep(0.05)

    finally:
        if cap is not None:
            cap.release()

else:
    if not hasattr(st.session_state, "model"):
        st.info("Load a YOLO model and start the camera to begin live proctoring.")
    else:
        st.info("Camera is not running. Click 'Start Camera' in the sidebar.")


# Footer: quick tips
st.markdown("---")
st.write("**Tips:** Ensure your model is trained with `imgsz` large enough for small objects (e.g., 960 or 1024). For best results, label tightly and include many hard examples.")


# End of file
