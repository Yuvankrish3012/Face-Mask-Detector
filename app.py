import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import pickle
from PIL import Image

# Set page config as first command
st.set_page_config(page_title="😷 Face Mask Detector", layout="centered")

# Load model and label encoder using cache
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r"D:\ML PROJECTS\Face Mask Detection\mask_detector_model.h5")
    with open(r"D:\ML PROJECTS\Face Mask Detection\label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, label_encoder = load_model()

# Label and color maps
label_map = {
    0: "Incorrect Mask",
    1: "With Mask",
    2: "Without Mask"
}
color_map = {
    "With Mask": (0, 255, 0),
    "Without Mask": (0, 0, 255),
    "Incorrect Mask": (0, 255, 255)
}

# Title
st.markdown("<h1 style='text-align:center;'>😷 Face Mask Detection App</h1>", unsafe_allow_html=True)
st.markdown("Upload an image with human faces, and the model will detect mask status.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="📸 Original Image", use_container_width=True)

    # Load OpenCV face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(image_rgb, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.warning("😕 No faces detected.")
    else:
        for (x, y, w, h) in faces:
            face = image_rgb[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (128, 128)) / 255.0
            face_array = np.expand_dims(face_resized, axis=0)

            prediction = model.predict(face_array)
            class_index = np.argmax(prediction)
            label = label_map[class_index]
            confidence = np.max(prediction)

            # Draw bounding box and label
            cv2.rectangle(image_rgb, (x, y), (x+w, y+h), color_map[label], 2)
            cv2.putText(image_rgb, f"{label} ({confidence*100:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map[label], 2)

        # Display result
        st.image(image_rgb, caption="✅ Detection Result", use_container_width=True)
