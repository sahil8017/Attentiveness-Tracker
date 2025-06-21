import cv2
import os
from roboflow import Roboflow
from playsound import playsound  # pip install playsound

# === CONFIGURATION ===
API_KEY = "CRO2jervxUZh1DbxqL37"
ALERT_CLASSES = ["sleepy", "bored"]
ALERT_SOUND_PATH = r"C:\Users\sahil\Desktop\attentiveness_detection\alert.mp3"  # Correct path
TEMP_IMAGE_PATH = "frame.jpg"

# === INITIALIZE MODEL ===
rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project("attention50k")
model = project.version(2).model

# === START WEBCAM ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save current frame as temp image
    cv2.imwrite(TEMP_IMAGE_PATH, frame)

    # Get prediction from Roboflow
    prediction = model.predict(TEMP_IMAGE_PATH, confidence=40, overlap=30).json()

    for pred in prediction['predictions']:
        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        class_name = pred['class']
        confidence = pred['confidence']

        # Draw bounding box and label
        cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (255, 0, 255), 2)
        cv2.putText(frame, f"{class_name} {int(confidence * 100)}%", (x - w // 2, y - h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Play alert if sleepy or bored
        if class_name.lower() in ALERT_CLASSES:
            try:
                playsound(ALERT_SOUND_PATH)
            except Exception as e:
                print(f"Error playing sound: {e}")

    # Show webcam with predictions
    cv2.imshow("Real-Time Attentiveness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if os.path.exists(TEMP_IMAGE_PATH):
    os.remove(TEMP_IMAGE_PATH)
