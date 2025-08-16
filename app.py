import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from inference import (
    load_plant_model,
    load_soil_model,
    predict_plant,
    predict_soil,
)
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av


# -------------------------------
# App Config
# -------------------------------
st.set_page_config(
    page_title="Plant & Soil Analyzer",
    layout="wide"
)

st.title("🌿 Real-Time Plant Disease & 🧪 Soil Type Detection")
st.write("Live camera analysis using your models (Keras H5 + TFLite)")


# -------------------------------
# Load Models & Labels
# -------------------------------
@st.cache_resource
def load_models():
    plant_model = load_plant_model("models/plant_disease_model.h5")
    soil_model = load_soil_model("models/soil_model.tflite")
    return plant_model, soil_model

plant_model, soil_model = load_models()

# Load labels
with open("labels/disease_labels.txt") as f:
    disease_labels = [line.strip() for line in f.readlines()]

with open("labels/soil_labels.txt") as f:
    soil_labels = [line.strip() for line in f.readlines()]


# -------------------------------
# Sidebar Options
# -------------------------------
st.sidebar.header("⚙️ Settings")
inference_mode = st.sidebar.radio(
    "Inference Mode", 
    ["Auto (Both)", "Plant only", "Soil only"], 
    index=0
)


# -------------------------------
# Video Processor for WebRTC
# -------------------------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_label = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        label_texts = []

        # Plant Prediction
        if plant_model and (inference_mode in ["Auto (Both)", "Plant only"]):
            plant_label, plant_conf = predict_plant(img, plant_model, disease_labels)
            label_texts.append(f"🌿 {plant_label} ({plant_conf:.2f})")

        # Soil Prediction
        if soil_model and (inference_mode in ["Auto (Both)", "Soil only"]):
            soil_label, soil_conf = predict_soil(img, soil_model, soil_labels)
            label_texts.append(f"🧪 {soil_label} ({soil_conf:.2f})")

        # Draw labels on frame
        y0 = 30
        for txt in label_texts:
            cv2.putText(img, txt, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            y0 += 40

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -------------------------------
# WebRTC Config
# -------------------------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# -------------------------------
# Start Camera Stream
# -------------------------------
st.subheader("🎥 Live Camera Feed")
webrtc_streamer(
    key="plant-soil-analyzer",
    mode=WebRtcMode.RECVONLY,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
)


# -------------------------------
# Fallback: Image Upload
# -------------------------------
st.subheader("📷 Upload an Image for Analysis")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)

    results = []

    if plant_model and (inference_mode in ["Auto (Both)", "Plant only"]):
        plant_label, plant_conf = predict_plant(img, plant_model, disease_labels)
        results.append(f"🌿 {plant_label} ({plant_conf:.2f})")

    if soil_model and (inference_mode in ["Auto (Both)", "Soil only"]):
        soil_label, soil_conf = predict_soil(img, soil_model, soil_labels)
        results.append(f"🧪 {soil_label} ({soil_conf:.2f})")

    if results:
        st.success(" | ".join(results))
