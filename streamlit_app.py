import streamlit as st
import cv2
import os
import uuid
from roboflow import Roboflow
from playsound import playsound
import pandas as pd
from datetime import datetime
import numpy as np
import threading
import time # Import time for potential short sleep

# === CONFIG ===
st.set_page_config(page_title="Attentiveness Detector", layout="centered")

API_KEY = "CRO2jervxUZh1DbxqL37" # Your Roboflow API Key
ALERT_CLASSES = ["sleepy", "bored"]
ALERT_SOUND_PATH = r"C:\Users\sahil\Desktop\attentiveness_detection\alert.mp3" # Path to your alert.mp3
TEMP_IMAGE_NAME = lambda: f"frame_{uuid.uuid4().hex}.jpg"
CSV_LOG_PATH = r"C:\Users\sahil\Desktop\attentiveness_detection\attentiveness_log.csv" # CSV path

# === INIT ROBOTFLOW MODEL ===
@st.cache_resource # Cache the model loading to prevent re-loading on every rerun
def load_model():
    """Loads the Roboflow model for inference."""
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace().project("attention50k")
    model = project.version(2).model
    return model

model = load_model()

# === STREAMLIT UI ===
st.title("🎯 Real-Time Attentiveness Tracker")
st.markdown("This app uses your webcam and a trained model to detect attentiveness (awake, sleepy, bored).")

# Initialize session state variables for webcam control and camera object
if "run_webcam" not in st.session_state:
    st.session_state.run_webcam = False
if "cap" not in st.session_state:
    st.session_state.cap = None # To store the cv2.VideoCapture object

col1, col2 = st.columns([1,1])
with col1:
    start_button = st.button("✅ Start Tracking", use_container_width=True)
with col2:
    stop_button = st.button("🛑 Stop Tracking", use_container_width=True)

# Handle button clicks
if start_button:
    # If already running, do nothing to avoid re-initializing camera
    if not st.session_state.run_webcam:
        st.session_state.run_webcam = True
        st.rerun() # *** CHANGED HERE: st.experimental_rerun() -> st.rerun() ***
elif stop_button:
    if st.session_state.run_webcam:
        st.session_state.run_webcam = False
        # Explicitly release camera when stop is clicked
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None # Clear the camera object
        st.success("Webcam stream stopped. You can close this browser tab.")
        st.empty() # Clear the image placeholder immediately
        st.rerun() # *** CHANGED HERE: st.experimental_rerun() -> st.rerun() ***

# === LOGGING SETUP ===
# Create CSV log file if it doesn't exist
if not os.path.exists(CSV_LOG_PATH):
    df = pd.DataFrame(columns=["Time", "Class", "Confidence", "Frame_ID"])
    df.to_csv(CSV_LOG_PATH, index=False)

# === Function to play sound in a separate thread ===
def play_alert_sound(sound_path):
    """Plays an alert sound in a separate, blocking thread."""
    try:
        playsound(sound_path, block=True)
    except Exception as e:
        print(f"Error playing sound in thread: {e}")

# === RUN DETECTION ===
if st.session_state.run_webcam:
    st.info("🔴 Tracking in progress. Press 'Stop Tracking' button to end.")
    frame_placeholder = st.empty() # Placeholder for the webcam feed

    # Initialize webcam only if not already initialized
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)
        # Check if camera opened successfully
        if not st.session_state.cap.isOpened():
            st.error("Error: Could not open webcam. Please ensure it's not in use by another application.")
            st.session_state.run_webcam = False # Stop the loop if camera fails
            st.session_state.cap = None
            st.stop() # Stop the Streamlit execution if camera is not available

    frame_id = 0

    # The main webcam processing loop
    while st.session_state.run_webcam:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Webcam stream ended unexpectedly.")
            st.session_state.run_webcam = False # Stop the loop
            break

        frame_id += 1
        temp_path = TEMP_IMAGE_NAME()
        cv2.imwrite(temp_path, frame)

        # Roboflow prediction
        try:
            prediction = model.predict(temp_path, confidence=40, overlap=30).json()
        except Exception as e:
            print(f"Error during Roboflow prediction: {e}")
            prediction = {"predictions": []} # Fallback to empty predictions

        os.remove(temp_path) # Clean up temporary image file

        # Draw predictions on the frame
        for pred in prediction["predictions"]:
            x, y = int(pred["x"]), int(pred["y"])
            w, h = int(pred["width"]), int(pred["height"])
            class_name = pred["class"].lower()
            confidence = pred["confidence"]

            # Draw bounding box and label
            cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {int(confidence * 100)}%", (x - w//2, y - h//2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Play alert sound in a separate thread if a relevant class is detected
            if class_name in ALERT_CLASSES:
                sound_thread = threading.Thread(target=play_alert_sound, args=(ALERT_SOUND_PATH,))
                sound_thread.start()

            # Log data to CSV
            log_df = pd.DataFrame([{
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Class": class_name,
                "Confidence": round(confidence, 2),
                "Frame_ID": frame_id
            }])
            log_df.to_csv(CSV_LOG_PATH, mode='a', header=False, index=False)

        # Display the frame in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Small delay to reduce CPU usage and allow Streamlit responsiveness
        time.sleep(0.01)

    # After the loop breaks (either by 'Stop Tracking' or webcam error)
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None # Ensure it's cleared
    st.success("Webcam stopped.")
    # Clear the placeholder one last time to ensure no stale image
    frame_placeholder.empty()