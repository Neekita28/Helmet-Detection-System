import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="PPE Safety Compliance System", layout="wide")

st.markdown("<h2 style='text-align: center;'>🦺 Real-Time Industrial PPE & Helmet Compliance Monitor</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Computer Vision System for Workplace Safety</p>", unsafe_allow_html=True)

# 1. Load Trained Model
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "best.pt")
    if not os.path.exists(model_path):
        if os.path.exists("best.pt"):
            return YOLO("best.pt")
        raise FileNotFoundError(f"Model weights not found at {model_path} or best.pt")
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model weights: {e}. Please ensure 'best.pt' is in the 'models/' folder or next to app.py.")
    st.stop()

# 2. Sidebar Controls
st.sidebar.header("⚙️ Controls")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.40, 0.05)
input_mode = st.sidebar.radio("Select Input Mode", ["Image Inspection", "Video Inspection"])

with st.sidebar.expander("🔍 Model Class Info", expanded=False):
    st.json(model.names)

# 3. Image Processing Mode
if input_mode == "Image Inspection":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file).convert("RGB")
        col1.image(image, caption="Original Workplace Frame", use_container_width=True)
        
        img_np = np.array(image)
        results = model.predict(img_np, conf=conf_threshold)[0]
        
        # Extract detected bounding boxes
        class_names = results.names
        detected_labels = [class_names[int(cls)].strip() for cls in results.boxes.cls]
        total_detections = len(detected_labels)
        
        # Draw bounding boxes
        annotated_frame = results.plot()
        col2.image(annotated_frame, caption="YOLOv8 Detection Feed", use_container_width=True)
        
        # Metrics Display
        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("👷 Detections Logged", total_detections)
        m2.metric("🏷️ Primary Class Detected", "head" if total_detections > 0 else "None")
        
        if total_detections > 0:
            st.warning(f"Detection logged: {total_detections} instance(s) of '{class_names[0]}' detected.")
        else:
            st.info("No targets detected with current confidence threshold.")

# 4. Video Processing Mode
elif input_mode == "Video Inspection":
    video_file = st.file_uploader("Upload Site Video (.mp4, .avi)", type=["mp4", "avi", "mov"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        st.info("Processing video feed...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, conf=conf_threshold)[0]
            annotated_frame = results.plot()
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
        cap.release()