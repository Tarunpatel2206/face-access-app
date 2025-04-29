import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import os

# Path to known images
KNOWN_PATH = "ImagesAttendence"

# Load known faces
def load_known_faces(path):
    images = []
    names = []
    for filename in os.listdir(path):
        img = face_recognition.load_image_file(f"{path}/{filename}")
        encode = face_recognition.face_encodings(img)
        if encode:
            images.append(encode[0])
            names.append(os.path.splitext(filename)[0])
    return images, names

known_encodings, known_names = load_known_faces(KNOWN_PATH)

# Streamlit UI
st.title("🔐 Face Access App")
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    img_np = np.array(img)

    unknown_encodings = face_recognition.face_encodings(img_np)

    if unknown_encodings:
        match_results = face_recognition.compare_faces(known_encodings, unknown_encodings[0])
        face_distances = face_recognition.face_distance(known_encodings, unknown_encodings[0])

        if any(match_results):
            best_match = np.argmin(face_distances)
            matched_name = known_names[best_match]
            st.success(f"✅ Access Granted to {matched_name}")
        else:
            st.error("🚫 Face not recognized. Access Denied.")
    else:
        st.warning("😕 No face detected. Please try a clearer image.")