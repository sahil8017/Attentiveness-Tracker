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

// === AUTH GUARD ===
// Redirect to login if not authenticated
if (typeof AUTH !== 'undefined' && !AUTH.requireAuth()) {
    // Auth guard triggered redirect — halt script execution.
    // We use a self-executing async to block without throw.
    (async () => { await new Promise(() => { }); })();
}

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
let activeAbortController = null;
let consecutiveErrors = 0;
const MAX_CONSECUTIVE_ERRORS = 5;
const timelineSegments = [];
const ALERT_CLASSES = ['sleepy', 'bored'];
const ALERT_COOLDOWN_MS = 5000;
let lastAlertSoundTime = 0;

// Device ID for multi-device isolation
const deviceId = localStorage.getItem('device_id') || (() => {
    const id = crypto.randomUUID();
    localStorage.setItem('device_id', id);
    return id;
})();

// Video resolution (set dynamically after webcam starts)
let videoWidth = 640;
let videoHeight = 480;

// Client-side smoothing buffer
const clientBuffer = [];
const CLIENT_BUFFER_SIZE = 5;

// Color map for states
const STATE_COLORS = {
    awake: { bg: 'from-emerald-500/20 to-emerald-500/5', text: 'text-emerald-600 dark:text-emerald-300', box: '#34d399', timeline: '#34d399' },
    sleepy: { bg: 'from-amber-500/20 to-amber-500/5', text: 'text-amber-600 dark:text-amber-300', box: '#fbbf24', timeline: '#fbbf24' },
    bored: { bg: 'from-rose-500/20 to-rose-500/5', text: 'text-rose-600 dark:text-rose-300', box: '#fb7185', timeline: '#fb7185' }
};

// === UTILITY FUNCTIONS ===

function updateStatus(message, type = 'info') {
    statusText.textContent = message;
    statusDiv.className = 'mb-4 sm:mb-6 p-3 rounded-xl text-center font-medium text-sm transition-all duration-200';

    const styles = {
        success: 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-600 dark:text-emerald-300',
        error: 'bg-rose-500/10 border border-rose-500/30 text-rose-600 dark:text-rose-300',
        warning: 'bg-amber-500/10 border border-amber-500/30 text-amber-600 dark:text-amber-300',
        info: 'bg-gray-50 dark:bg-zinc-900 border border-gray-200 dark:border-white/10 text-gray-500 dark:text-zinc-400'
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
    attentionScoreDisplay.className = score >= 70
        ? 'text-3xl font-semibold text-emerald-600 dark:text-emerald-300'
        : score >= 40
            ? 'text-3xl font-semibold text-amber-600 dark:text-amber-300'
            : 'text-3xl font-semibold text-rose-600 dark:text-rose-300';
    attentionScoreDisplay.setAttribute('data-score', score);
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

// === CANVAS SIZING ===

function syncCanvasSize() {
    if (video.videoWidth && video.videoHeight) {
        videoWidth = video.videoWidth;
        videoHeight = video.videoHeight;
        canvas.width = videoWidth;
        canvas.height = videoHeight;
    }
}

// === WEBCAM ===

async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user',
                frameRate: { ideal: 15 }
            }
        });
        video.srcObject = stream;

        await new Promise((resolve) => {
            video.onloadedmetadata = () => {
                syncCanvasSize();
                resolve();
            };
        });

        return true;
    } catch (error) {
        console.error('Error accessing webcam:', error);
        updateStatus('Camera access failed. Please check permissions.', 'error');
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
    return canvas.toDataURL('image/jpeg', 0.6);
}

// === API HELPERS ===
// Use AUTH.apiFetch for all API calls to include JWT token

async function apiFetch(url, options = {}) {
    if (typeof AUTH !== 'undefined') {
        return AUTH.apiFetch(url, options);
    }
    // Fallback for no auth
    return fetch(url, options);
}

// Safe wrapper: returns null on abort instead of throwing
async function safeApiFetch(url, options = {}) {
    try {
        return await apiFetch(url, options);
    } catch (err) {
        if (err.name === 'AbortError') return null;
        throw err;
    }
}

// === DETECTION ===

async function detectAttentiveness() {
    if (!isTracking) return;

    if (activeAbortController) {
        activeAbortController.abort();
    }
    activeAbortController = new AbortController();

    try {
        const imageData = captureFrame();

        const response = await safeApiFetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData, session_id: sessionId }),
            signal: activeAbortController.signal
        });

        // Aborted — skip this frame
        if (!response) return;

        if (!response.ok) throw new Error('Prediction request failed');

        const data = await response.json();
        consecutiveErrors = 0;

        if (data.success) {
            if (data.skipped) {
                blurOverlay.className = 'absolute top-3 right-3 px-3 py-1 rounded-full text-xs font-mono bg-amber-500/10 text-amber-600 dark:text-amber-300 border border-amber-500/30';
                blurText.textContent = `Frame blurred (${data.blur_score})`;
                blurOverlay.classList.remove('hidden');
                return;
            }

            if (data.blurry) {
                blurOverlay.className = 'absolute top-3 right-3 px-3 py-1 rounded-full text-xs font-mono bg-amber-500/5 text-amber-600/70 dark:text-amber-200/70 border border-amber-500/20';
                blurText.textContent = `Using cached frame (${data.blur_score})`;
                blurOverlay.classList.remove('hidden');
            } else {
                blurOverlay.classList.add('hidden');
            }

            frameCount++;
            totalFrames++;
            frameCountDisplay.textContent = frameCount;

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

                    ctx.shadowColor = colors.box;
                    ctx.shadowBlur = 15;
                    ctx.strokeStyle = colors.box;
                    ctx.lineWidth = 2.5;
                    ctx.beginPath();
                    ctx.roundRect(x - w / 2, y - h / 2, w, h, 8);
                    ctx.stroke();
                    ctx.shadowBlur = 0;

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

                if (topPred) {
                    const finalClass = clientSmooth(topPred.smoothedClass);
                    currentStateDisplay.textContent = finalClass.toUpperCase();
                    confidenceDisplay.textContent = Math.round(topPred.confidence * 100) + '%';

                    const stateColors = STATE_COLORS[finalClass] || STATE_COLORS.awake;
                    currentStateDisplay.className = `text-2xl font-bold ${stateColors.text}`;

                    if (finalClass === 'awake') attentiveFrames++;
                    updateAttentionScore();
                    addTimelineSegment(finalClass);
                }

                if (data.trigger_alert) {
                    showAlert();
                } else {
                    hideAlert();
                }
            } else {
                currentStateDisplay.textContent = 'No face';
                currentStateDisplay.className = 'text-2xl font-bold text-gray-400 dark:text-zinc-500';
                confidenceDisplay.textContent = '—';
                hideAlert();
            }
        }
    } catch (error) {
        if (error.name === 'AbortError') return;

        consecutiveErrors++;
        console.error('Detection error:', error);

        if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
            updateStatus('Multiple errors detected. Retrying connection...', 'warning');
        }
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
        const res = await apiFetch('/api/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ device_id: deviceId }),
        });
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
        const res = await apiFetch(`/api/sessions/${sessionId}/end`, { method: 'POST' });
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

    updateStatus('Initializing detection...', 'info');
    startBtn.disabled = true;

    const webcamStarted = await startWebcam();
    if (!webcamStarted) {
        startBtn.disabled = false;
        return;
    }

    const sessionCreated = await createSession();
    if (!sessionCreated) {
        updateStatus('Failed to create session.', 'error');
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
    attentionScoreDisplay.className = 'text-3xl font-semibold text-indigo-600 dark:text-indigo-300';

    updateStatus('Detection active', 'success');
    startBtn.disabled = true;
    stopBtn.disabled = false;
    startSessionTimer();
    document.querySelector('.hud-shell')?.classList.add('hud-active');

    const interval = parseInt(detectionIntervalSelect.value, 10);
    detectionInterval = setInterval(detectAttentiveness, interval);
}

async function stopDetection() {
    if (!isTracking) return;

    isTracking = false;
    clearInterval(detectionInterval);

    if (activeAbortController) {
        activeAbortController.abort();
        activeAbortController = null;
    }

    stopWebcam();
    stopSessionTimer();

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    await endCurrentSession();

    updateStatus('Session complete. Click "Start Detection" for a new session.', 'info');
    startBtn.disabled = false;
    stopBtn.disabled = true;
    document.querySelector('.hud-shell')?.classList.remove('hud-active');
    currentStateDisplay.textContent = '—';
    currentStateDisplay.className = 'text-2xl font-semibold text-gray-800 dark:text-zinc-100';
    confidenceDisplay.textContent = '—';
    attentionScoreDisplay.className = 'text-3xl font-semibold text-indigo-600 dark:text-indigo-300';
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