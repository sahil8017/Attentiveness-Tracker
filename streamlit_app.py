import streamlit as st
import cv2
import os
import uuid
from roboflow import Roboflow
from playsound import playsound
import pandas as pd
from datetime import datetime
import numpy as np

# === CONFIG ===
st.set_page_config(page_title="Attentiveness Detector", layout="centered")

API_KEY = "CRO2jervxUZh1DbxqL37"
ALERT_CLASSES = ["sleepy", "bored"]
ALERT_SOUND_PATH = r"C:\Users\sahil\Desktop\attentiveness_detection\alert.mp3"  # Path to your alert.mp3
TEMP_IMAGE_NAME = lambda: f"frame_{uuid.uuid4().hex}.jpg"
CSV_LOG_PATH = r"C:\Users\sahil\Desktop\attentiveness_detection\attentiveness_log.csv"  # CSV path

# === INIT ROBOTFLOW MODEL ===
@st.cache_resource
def load_model():
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace().project("attention50k")
    model = project.version(2).model
    return model

model = load_model()

# === STREAMLIT UI ===
st.title("🎯 Real-Time Attentiveness Detection")
st.markdown("This app uses your webcam and a trained model to detect attentiveness (awake, sleepy, bored).")

start_check = st.button("✅ Let's Start Checking Attentiveness")

# === LOGGING SETUP ===
if not os.path.exists(CSV_LOG_PATH):
    df = pd.DataFrame(columns=["Time", "Class", "Confidence", "Frame_ID"])
    df.to_csv(CSV_LOG_PATH, index=False)

# === RUN DETECTION ===
if start_check:
    st.info("🔴 Press 'Stop' button or close browser tab to end.")
    frame_placeholder = st.empty()
    cap = cv2.VideoCapture(0)

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam not detected.")
            break

        frame_id += 1
        temp_path = TEMP_IMAGE_NAME()
        cv2.imwrite(temp_path, frame)

        # Roboflow prediction
        prediction = model.predict(temp_path, confidence=40, overlap=30).json()
        os.remove(temp_path)

        # Draw predictions
        for pred in prediction["predictions"]:
            x, y = int(pred["x"]), int(pred["y"])
            w, h = int(pred["width"]), int(pred["height"])
            class_name = pred["class"].lower()
            confidence = pred["confidence"]

            # Draw on frame
            cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {int(confidence * 100)}%", (x - w//2, y - h//2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Play alert sound
            if class_name in ALERT_CLASSES:
                try:
                    playsound(ALERT_SOUND_PATH, block=False)
                except Exception as e:
                    print("Sound Error:", e)

            # Log data to CSV
            log_df = pd.DataFrame([{
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Class": class_name,
                "Confidence": round(confidence, 2),
                "Frame_ID": frame_id
            }])
            log_df.to_csv(CSV_LOG_PATH, mode='a', header=False, index=False)

        # Show in browser
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Streamlit rerun exits the loop
        if not st.session_state.get("run_webcam", True):
            break

    cap.release()
    st.success("Webcam stopped.")
