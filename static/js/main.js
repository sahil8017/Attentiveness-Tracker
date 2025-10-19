// Get DOM elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusDiv = document.getElementById('status');
const statusText = document.getElementById('statusText');
const frameCountDisplay = document.getElementById('frameCount');
const currentStateDisplay = document.getElementById('currentState');
const confidenceDisplay = document.getElementById('confidence');
const alertMessage = document.getElementById('alertMessage');
const alertSound = document.getElementById('alertSound');

// State variables
let stream = null;
let isTracking = false;
let frameCount = 0;
let detectionInterval = null;
const ALERT_CLASSES = ['sleepy', 'bored'];

// Update status display
function updateStatus(message, type = 'info') {
    statusText.textContent = message;
    statusDiv.className = 'mb-6 p-4 rounded-lg text-center font-semibold';
    
    if (type === 'success') {
        statusDiv.classList.add('bg-green-100', 'text-green-800');
    } else if (type === 'error') {
        statusDiv.classList.add('bg-red-100', 'text-red-800');
    } else if (type === 'warning') {
        statusDiv.classList.add('bg-yellow-100', 'text-yellow-800');
    } else {
        statusDiv.classList.add('bg-blue-100', 'text-blue-800');
    }
}

// Initialize webcam
async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        video.srcObject = stream;
        return true;
    } catch (error) {
        console.error('Error accessing webcam:', error);
        updateStatus('❌ Could not access webcam. Please check permissions.', 'error');
        return false;
    }
}

// Stop webcam
function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;
    }
}

// Capture frame from video and convert to base64
function captureFrame() {
    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert canvas to base64 image
    return canvas.toDataURL('image/jpeg', 0.8);
}

// Send frame to backend for prediction
async function detectAttentiveness() {
    if (!isTracking) return;
    
    try {
        const imageData = captureFrame();
        
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        });
        
        if (!response.ok) {
            throw new Error('Prediction request failed');
        }
        
        const data = await response.json();
        
        if (data.success) {
            frameCount++;
            frameCountDisplay.textContent = frameCount;
            
            // Clear previous drawings
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw predictions
            if (data.predictions && data.predictions.length > 0) {
                let highestConfidencePred = null;
                let maxConfidence = 0;
                
                data.predictions.forEach(pred => {
                    const x = pred.x;
                    const y = pred.y;
                    const w = pred.width;
                    const h = pred.height;
                    const className = pred.class.toLowerCase();
                    const confidence = pred.confidence;
                    
                    // Track highest confidence prediction
                    if (confidence > maxConfidence) {
                        maxConfidence = confidence;
                        highestConfidencePred = pred;
                    }
                    
                    // Choose color based on class
                    let boxColor = '#00ff00'; // green for awake
                    if (className === 'sleepy') {
                        boxColor = '#ffaa00'; // orange for sleepy
                    } else if (className === 'bored') {
                        boxColor = '#ff0000'; // red for bored
                    }
                    
                    // Draw bounding box
                    ctx.strokeStyle = boxColor;
                    ctx.lineWidth = 3;
                    ctx.strokeRect(x - w/2, y - h/2, w, h);
                    
                    // Draw label background
                    const label = `${className} ${Math.round(confidence * 100)}%`;
                    ctx.font = 'bold 16px Arial';
                    const textWidth = ctx.measureText(label).width;
                    ctx.fillStyle = boxColor;
                    ctx.fillRect(x - w/2, y - h/2 - 30, textWidth + 10, 25);
                    
                    // Draw label text
                    ctx.fillStyle = '#ffffff';
                    ctx.fillText(label, x - w/2 + 5, y - h/2 - 10);
                    
                    // Check if alert needed
                    if (ALERT_CLASSES.includes(className)) {
                        showAlert();
                        playAlertSound();
                    } else {
                        hideAlert();
                    }
                });
                
                // Update stats display with highest confidence prediction
                if (highestConfidencePred) {
                    currentStateDisplay.textContent = highestConfidencePred.class.toUpperCase();
                    confidenceDisplay.textContent = Math.round(highestConfidencePred.confidence * 100) + '%';
                }
            } else {
                currentStateDisplay.textContent = 'No Detection';
                confidenceDisplay.textContent = '-';
                hideAlert();
            }
        }
    } catch (error) {
        console.error('Error during detection:', error);
        updateStatus('⚠️ Detection error. Retrying...', 'warning');
    }
}

// Show alert message
function showAlert() {
    alertMessage.classList.remove('hidden');
}

// Hide alert message
function hideAlert() {
    alertMessage.classList.add('hidden');
}

// Play alert sound
function playAlertSound() {
    if (alertSound) {
        alertSound.currentTime = 0;
        alertSound.play().catch(err => {
            console.log('Could not play alert sound:', err);
        });
    }
}

// Start detection
async function startDetection() {
    if (isTracking) return;
    
    updateStatus('🔄 Starting webcam...', 'info');
    startBtn.disabled = true;
    
    const webcamStarted = await startWebcam();
    
    if (webcamStarted) {
        isTracking = true;
        updateStatus('🟢 Detection Active - Tracking in progress', 'success');
        startBtn.disabled = true;
        stopBtn.disabled = false;
        
        // Start detection loop (every 1 second)
        detectionInterval = setInterval(detectAttentiveness, 1000);
    } else {
        startBtn.disabled = false;
    }
}

// Stop detection
function stopDetection() {
    if (!isTracking) return;
    
    isTracking = false;
    clearInterval(detectionInterval);
    stopWebcam();
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    updateStatus('⏸️ Detection stopped. Click "Start Detection" to resume.', 'info');
    startBtn.disabled = false;
    stopBtn.disabled = true;
    
    currentStateDisplay.textContent = '-';
    confidenceDisplay.textContent = '-';
    hideAlert();
}

// Event listeners
startBtn.addEventListener('click', startDetection);
stopBtn.addEventListener('click', stopDetection);

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopDetection();
});