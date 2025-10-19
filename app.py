from flask import Flask, render_template, request, jsonify
import cv2
import base64
import numpy as np
from roboflow import Roboflow
import pandas as pd
from datetime import datetime
from pathlib import Path
import uuid
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIGURATION ===
BASE_DIR = Path(__file__).resolve().parent
API_KEY = "CRO2jervxUZh1DbxqL37"  # Replace with your actual Roboflow API key
CSV_LOG_PATH = BASE_DIR / "attentiveness_log.csv"
PLOTS_DIR = BASE_DIR / "static" / "images"

# Ensure necessary directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# === INITIALIZATION ===
app = Flask(__name__)

# Initialize Roboflow model
rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project("attention50k")
model = project.version(2).model

# Initialize CSV log file if missing
if not CSV_LOG_PATH.exists():
    df = pd.DataFrame(columns=["Time", "Class", "Confidence", "Frame_ID"])
    df.to_csv(CSV_LOG_PATH, index=False)

frame_counter = {"count": 0}

# === ROUTES ===
@app.route('/')
def index():
    """Homepage"""
    return render_template('index.html')

@app.route('/detection')
def detection():
    """Live detection page"""
    return render_template('detection.html')

@app.route('/dashboard')
def dashboard():
    """Analytics dashboard page"""
    return render_template('dashboard.html')

# === API ENDPOINTS ===
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives a base64 image and returns Roboflow predictions.
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 image
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # Save temporary frame
        frame_counter["count"] += 1
        temp_path = BASE_DIR / f"temp_frame_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(temp_path), frame)

        # Roboflow prediction
        prediction = model.predict(str(temp_path), confidence=40, overlap=30).json()

        # Delete temp image
        if temp_path.exists():
            temp_path.unlink()

        # Log predictions
        for pred in prediction.get('predictions', []):
            log_df = pd.DataFrame([{
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Class": pred['class'].lower(),
                "Confidence": round(pred['confidence'], 2),
                "Frame_ID": frame_counter["count"]
            }])
            log_df.to_csv(CSV_LOG_PATH, mode='a', header=False, index=False)

        return jsonify({
            'success': True,
            'predictions': prediction.get('predictions', []),
            'frame_id': frame_counter["count"]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate_plots', methods=['GET'])
def generate_plots():
    """Generates analytics plots using seaborn and matplotlib."""
    try:
        if not CSV_LOG_PATH.exists():
            return jsonify({'error': 'No log data available'}), 404

        df = pd.read_csv(CSV_LOG_PATH)
        if df.empty:
            return jsonify({'error': 'Log file is empty'}), 404

        df['Time'] = pd.to_datetime(df['Time'])
        plt.close('all')

        # Plot 1: Confidence Over Time
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='Time', y='Confidence', hue='Class', marker='o', linewidth=2)
        plt.title('Confidence Over Time for Different Attentiveness Classes', fontsize=14)
        plt.xlabel('Time', fontsize=11)
        plt.ylabel('Confidence', fontsize=11)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Class')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'confidence_over_time.png', dpi=100, bbox_inches='tight')
        plt.close()

        # Plot 2: Class Distribution
        plt.figure(figsize=(8, 6))
        class_counts = df['Class'].value_counts()
        colors = sns.color_palette('Set2', len(class_counts))
        plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        plt.title('Distribution of Detected Classes', fontsize=14)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'class_distribution.png', dpi=100, bbox_inches='tight')
        plt.close()

        # Plot 3: Average Confidence by Class
        plt.figure(figsize=(10, 6))
        avg_conf = df.groupby('Class')['Confidence'].mean().sort_values(ascending=False)
        sns.barplot(x=avg_conf.index, y=avg_conf.values, palette='viridis')
        plt.title('Average Confidence Score by Class', fontsize=14)
        plt.xlabel('Class', fontsize=11)
        plt.ylabel('Average Confidence', fontsize=11)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'avg_confidence_by_class.png', dpi=100, bbox_inches='tight')
        plt.close()

        return jsonify({
            'success': True,
            'message': 'Plots generated successfully',
            'plots': [
                '/static/images/confidence_over_time.png',
                '/static/images/class_distribution.png',
                '/static/images/avg_confidence_by_class.png'
            ]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_stats', methods=['GET'])
def get_stats():
    """Returns basic statistics from the attentiveness log."""
    try:
        if not CSV_LOG_PATH.exists():
            return jsonify({'total_frames': 0, 'classes': {}})

        df = pd.read_csv(CSV_LOG_PATH)
        if df.empty:
            return jsonify({'total_frames': 0, 'classes': {}})

        stats = {
            'total_frames': len(df),
            'classes': df['Class'].value_counts().to_dict(),
            'avg_confidence': round(df['Confidence'].mean(), 2)
        }

        return jsonify(stats)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# === ENTRY POINT ===
if __name__ == '__main__':
    # Production-safe entry point
    app.run(host='0.0.0.0', port=5000)
