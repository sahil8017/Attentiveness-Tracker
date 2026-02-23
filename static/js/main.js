// ============================================================
// Attentiveness Tracker — Main Detection Script
// Session management, temporal smoothing, adaptive detection
// ============================================================

// === DOM ELEMENTS ===
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
const attentionScoreDisplay = document.getElementById('attentionScore');
const alertMessage = document.getElementById('alertMessage');
const alertSound = document.getElementById('alertSound');
const timeline = document.getElementById('timeline');
const sessionTimer = document.getElementById('sessionTimer');
const sessionBadge = document.getElementById('sessionBadge');
const sessionIdDisplay = document.getElementById('sessionIdDisplay');
const blurOverlay = document.getElementById('blurOverlay');
const blurText = document.getElementById('blurText');
const detectionIntervalSelect = document.getElementById('detectionInterval');
const alertSoundToggle = document.getElementById('alertSoundToggle');

// === STATE ===
let stream = null;
let isTracking = false;
let frameCount = 0;
let sessionId = null;
let detectionInterval = null;
let sessionStartTime = null;
let timerInterval = null;
let attentiveFrames = 0;
let totalFrames = 0;
let activeAbortController = null; // Prevent overlapping requests
let consecutiveErrors = 0;
const MAX_CONSECUTIVE_ERRORS = 5;
const timelineSegments = [];
const ALERT_CLASSES = ['sleepy', 'bored'];
const ALERT_COOLDOWN_MS = 5000;
let lastAlertSoundTime = 0;

// Client-side smoothing buffer
const clientBuffer = [];
const CLIENT_BUFFER_SIZE = 5;

// Color map for states
const STATE_COLORS = {
    awake: { bg: 'from-emerald-500/20 to-emerald-500/5', text: 'text-emerald-400', box: '#34d399', timeline: '#34d399' },
    sleepy: { bg: 'from-amber-500/20 to-amber-500/5', text: 'text-amber-400', box: '#fbbf24', timeline: '#fbbf24' },
    bored: { bg: 'from-red-500/20 to-red-500/5', text: 'text-red-400', box: '#f87171', timeline: '#f87171' }
};

// === UTILITY FUNCTIONS ===

function updateStatus(message, type = 'info') {
    statusText.textContent = message;
    statusDiv.className = 'mb-5 p-3 rounded-xl text-center font-semibold text-sm transition-all duration-300';

    const styles = {
        success: 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400',
        error: 'bg-red-500/10 border border-red-500/30 text-red-400',
        warning: 'bg-amber-500/10 border border-amber-500/30 text-amber-400',
        info: 'bg-white/5 border border-white/10 text-white/60'
    };
    statusDiv.className += ' ' + (styles[type] || styles.info);
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
    const secs = (seconds % 60).toString().padStart(2, '0');
    return `${mins}:${secs}`;
}

function clientSmooth(rawClass) {
    clientBuffer.push(rawClass);
    if (clientBuffer.length > CLIENT_BUFFER_SIZE) clientBuffer.shift();

    const counts = {};
    clientBuffer.forEach(c => { counts[c] = (counts[c] || 0) + 1; });
    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
}

function updateAttentionScore() {
    if (totalFrames === 0) {
        attentionScoreDisplay.textContent = '—';
        return;
    }
    const score = Math.round((attentiveFrames / totalFrames) * 100);
    attentionScoreDisplay.textContent = score;
}

function addTimelineSegment(state) {
    timelineSegments.push(state);
    renderTimeline();
}

function renderTimeline() {
    const total = timelineSegments.length;
    if (total === 0) return;

    timeline.innerHTML = '';
    timelineSegments.forEach((state) => {
        const seg = document.createElement('div');
        const color = STATE_COLORS[state]?.timeline || '#6b7280';
        seg.style.flex = '1';
        seg.style.backgroundColor = color;
        seg.style.minWidth = '2px';
        seg.style.transition = 'background-color 0.3s';
        timeline.appendChild(seg);
    });
}

function startSessionTimer() {
    sessionStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
        sessionTimer.textContent = formatTime(elapsed);
    }, 1000);
}

function stopSessionTimer() {
    clearInterval(timerInterval);
    timerInterval = null;
}

// === WEBCAM ===

async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user',
                frameRate: { ideal: 15 } // Lower framerate to reduce CPU
            }
        });
        video.srcObject = stream;
        return true;
    } catch (error) {
        console.error('Error accessing webcam:', error);
        updateStatus('❌ Could not access webcam. Please check permissions.', 'error');
        return false;
    }
}

function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;
    }
}

// === FRAME CAPTURE ===

function captureFrame() {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    // Reduced quality 0.6 (from 0.8) for faster encode/transfer
    return canvas.toDataURL('image/jpeg', 0.6);
}

// === DETECTION ===

async function detectAttentiveness() {
    if (!isTracking) return;

    // Abort any in-flight request to prevent overlap
    if (activeAbortController) {
        activeAbortController.abort();
    }
    activeAbortController = new AbortController();

    try {
        const imageData = captureFrame();

        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData, session_id: sessionId }),
            signal: activeAbortController.signal
        });

        if (!response.ok) throw new Error('Prediction request failed');

        const data = await response.json();
        consecutiveErrors = 0; // Reset error counter on success

        if (data.success) {
            // Handle fully skipped frames (no previous state available)
            if (data.skipped) {
                blurOverlay.className = 'absolute top-3 right-3 px-3 py-1 rounded-full text-xs font-mono bg-amber-500/20 text-amber-400 border border-amber-500/30';
                blurText.textContent = `⚠ Blurry (${data.blur_score})`;
                blurOverlay.classList.remove('hidden');
                // DON'T return early — keep detection loop running
                return;
            }

            // Show blur indicator (reduced opacity) if blurry but using cached state
            if (data.blurry) {
                blurOverlay.className = 'absolute top-3 right-3 px-3 py-1 rounded-full text-xs font-mono bg-amber-500/10 text-amber-300/60 border border-amber-500/20';
                blurText.textContent = `⚡ Cached (${data.blur_score})`;
                blurOverlay.classList.remove('hidden');
            } else {
                blurOverlay.classList.add('hidden');
            }

            frameCount++;
            totalFrames++;
            frameCountDisplay.textContent = frameCount;

            // Clear canvas for fresh drawing
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (data.predictions && data.predictions.length > 0) {
                let topPred = null;
                let maxConf = 0;

                data.predictions.forEach(pred => {
                    const x = pred.x, y = pred.y, w = pred.width, h = pred.height;
                    const rawClass = pred.class;
                    const smoothedClass = pred.smoothed_class || rawClass;
                    const confidence = pred.confidence;

                    if (confidence > maxConf) {
                        maxConf = confidence;
                        topPred = { ...pred, smoothedClass: smoothedClass };
                    }

                    const colors = STATE_COLORS[smoothedClass] || STATE_COLORS.awake;

                    // Draw bounding box with glow
                    ctx.shadowColor = colors.box;
                    ctx.shadowBlur = 15;
                    ctx.strokeStyle = colors.box;
                    ctx.lineWidth = 2.5;
                    ctx.beginPath();
                    ctx.roundRect(x - w / 2, y - h / 2, w, h, 8);
                    ctx.stroke();
                    ctx.shadowBlur = 0;

                    // Draw label pill
                    const label = `${smoothedClass} ${Math.round(confidence * 100)}%`;
                    ctx.font = 'bold 13px Inter, sans-serif';
                    const textWidth = ctx.measureText(label).width;
                    const pillX = x - w / 2;
                    const pillY = y - h / 2 - 28;

                    ctx.fillStyle = colors.box;
                    ctx.beginPath();
                    ctx.roundRect(pillX, pillY, textWidth + 16, 22, 6);
                    ctx.fill();

                    ctx.fillStyle = '#000';
                    ctx.fillText(label, pillX + 8, pillY + 15);
                });

                // Update display with top prediction
                if (topPred) {
                    const finalClass = clientSmooth(topPred.smoothedClass);
                    currentStateDisplay.textContent = finalClass.toUpperCase();
                    confidenceDisplay.textContent = Math.round(topPred.confidence * 100) + '%';

                    // Apply state color
                    const stateColors = STATE_COLORS[finalClass] || STATE_COLORS.awake;
                    currentStateDisplay.className = `text-2xl font-bold ${stateColors.text}`;

                    // Track attention
                    if (finalClass === 'awake') attentiveFrames++;
                    updateAttentionScore();
                    addTimelineSegment(finalClass);
                }

                // Handle alerts
                if (data.trigger_alert) {
                    showAlert();
                } else {
                    hideAlert();
                }
            } else {
                currentStateDisplay.textContent = 'No Face';
                currentStateDisplay.className = 'text-2xl font-bold text-white/40';
                confidenceDisplay.textContent = '—';
                hideAlert();
            }
        }
    } catch (error) {
        if (error.name === 'AbortError') return; // Intentional abort

        consecutiveErrors++;
        console.error('Detection error:', error);

        if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
            updateStatus(`⚠ Multiple errors. Check connection. Retrying...`, 'warning');
        }
        // Keep detection loop running — don't crash on transient errors
    } finally {
        activeAbortController = null;
    }
}

// === ALERTS ===

function showAlert() {
    alertMessage.classList.remove('hidden');
    const now = Date.now();
    if (alertSoundToggle.checked && (now - lastAlertSoundTime) > ALERT_COOLDOWN_MS) {
        playAlertSound();
        lastAlertSoundTime = now;
    }
}

function hideAlert() {
    alertMessage.classList.add('hidden');
}

function playAlertSound() {
    if (alertSound) {
        alertSound.currentTime = 0;
        alertSound.play().catch(err => console.log('Alert sound blocked:', err));
    }
}

// === SESSION MANAGEMENT ===

async function createSession() {
    try {
        const res = await fetch('/api/sessions', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            sessionId = data.session_id;
            sessionIdDisplay.textContent = sessionId.replace('session-', '');
            sessionBadge.classList.remove('hidden');
            return true;
        }
    } catch (e) {
        console.error('Failed to create session:', e);
    }
    return false;
}

async function endCurrentSession() {
    if (!sessionId) return;
    try {
        const res = await fetch(`/api/sessions/${sessionId}/end`, { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            attentionScoreDisplay.textContent = data.attention_score;
        }
    } catch (e) {
        console.error('Failed to end session:', e);
    }
}

// === START / STOP ===

async function startDetection() {
    if (isTracking) return;

    updateStatus('🔄 Initializing...', 'info');
    startBtn.disabled = true;

    const webcamStarted = await startWebcam();
    if (!webcamStarted) {
        startBtn.disabled = false;
        return;
    }

    const sessionCreated = await createSession();
    if (!sessionCreated) {
        updateStatus('❌ Failed to create session', 'error');
        stopWebcam();
        startBtn.disabled = false;
        return;
    }

    // Reset state
    isTracking = true;
    frameCount = 0;
    attentiveFrames = 0;
    totalFrames = 0;
    consecutiveErrors = 0;
    clientBuffer.length = 0;
    timelineSegments.length = 0;
    timeline.innerHTML = '';
    frameCountDisplay.textContent = '0';
    attentionScoreDisplay.textContent = '—';

    updateStatus('🟢 Detection Active', 'success');
    startBtn.disabled = true;
    stopBtn.disabled = false;
    startSessionTimer();

    const interval = parseInt(detectionIntervalSelect.value, 10);
    detectionInterval = setInterval(detectAttentiveness, interval);
}

async function stopDetection() {
    if (!isTracking) return;

    isTracking = false;
    clearInterval(detectionInterval);

    // Abort any in-flight request
    if (activeAbortController) {
        activeAbortController.abort();
        activeAbortController = null;
    }

    stopWebcam();
    stopSessionTimer();

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    await endCurrentSession();

    updateStatus('⏸ Session complete. Click "Start Detection" for a new session.', 'info');
    startBtn.disabled = false;
    stopBtn.disabled = true;
    currentStateDisplay.textContent = '—';
    currentStateDisplay.className = 'text-2xl font-bold text-white/80';
    confidenceDisplay.textContent = '—';
    hideAlert();
}

// === SETTINGS ===

detectionIntervalSelect.addEventListener('change', () => {
    if (isTracking && detectionInterval) {
        clearInterval(detectionInterval);
        const interval = parseInt(detectionIntervalSelect.value, 10);
        detectionInterval = setInterval(detectAttentiveness, interval);
    }
});

// === EVENT LISTENERS ===
startBtn.addEventListener('click', startDetection);
stopBtn.addEventListener('click', stopDetection);
window.addEventListener('beforeunload', () => { stopDetection(); });