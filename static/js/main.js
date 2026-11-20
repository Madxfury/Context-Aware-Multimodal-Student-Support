// Escape stored profile/session text before inserting it into HTML templates.
function escapeHTML(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[char]));
}

// ═══════════════════════════════════════════════════════════
//  Samvya — main.js
// ═══════════════════════════════════════════════════════════

function iconMarkup(name, className = '') {
  return '<svg class="icon ' + className + '" aria-hidden="true" focusable="false"><use href="/static/images/icons.svg#' + name + '"></use></svg>';
}

// ── Emotion config ───────────────────────────────────────────
const EMOTION_CFG = {
  Angry: { icon: 'angry', label: 'Angry' },
  Disgust: { icon: 'frown', label: 'Disgust' },
  Fear: { icon: 'shield-alert', label: 'Fear' },
  Happy: { icon: 'smile', label: 'Happy' },
  Sad: { icon: 'frown', label: 'Sad' },
  Surprise: { icon: 'zap', label: 'Surprise' },
  Neutral: { icon: 'meh', label: 'Neutral' },
};

// ── State ─────────────────────────────────────────────────────
let currentUser = null;
let userSessions = [];
let latestScore = null;
let latestRisk = null;
let latestConfidence = null;
let profileTrigger = null;

let lockedEmotion = null;
let isLocked = false;
let scanStartTime = null;
let isScanning = false;
let heatmapEnabled = false;
let cameraReadyTimer;
let video, captureCanvas, captureCtx, overlayCanvas, overlayCtx;

// Voice and reply state for the current conversation.
let voiceCapture = null;
let activeChatRequest = null;
let activeNarration = null;
let finishingCheckIn = false;
let chatNotice = '';
let saveQueue = Promise.resolve();

// Chat history state
let chatHistory = [];
let wearablePoll = null;
let wearableSummary = null;
let wearableShown = false;  // true once a snapshot from the server has been rendered

// ── Initialise ────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', initApp);

async function initApp() {
  video = document.getElementById('video');
  captureCanvas = document.createElement('canvas');
  captureCtx = captureCanvas.getContext('2d');
  overlayCanvas = document.getElementById('overlay-canvas');
  overlayCtx = overlayCanvas.getContext('2d');

  initChatUI();
  document.getElementById('start-camera').addEventListener('click', startCameraAndScan);
  document.getElementById('chat-camera-pause').addEventListener('click', toggleMonitorPause);
  document.getElementById('chat-camera-stop').addEventListener('click', turnOffChatCamera);
  // The counsellor voice is a per-device preference.
  const voiceChoice = document.getElementById('counsellor-voice');
  try {
    const saved = localStorage.getItem('samvya-counsellor-voice');
    if (saved === 'male' || saved === 'female') voiceChoice.value = saved;
  } catch {}
  voiceChoice.addEventListener('change', () => {
    try { localStorage.setItem('samvya-counsellor-voice', voiceChoice.value); } catch {}
  });
  document.getElementById('retry-report-save').addEventListener('click', retryReportSave);
  const profileOptions = document.getElementById('profile-options');
  const ageField = document.getElementById('input-user-age');
  document.getElementById('profile-options-button').addEventListener('click', () => profileOptions.showModal());
  document.getElementById('returning-profiles-button').addEventListener('click', () => {
    document.getElementById('returning-users-box').showModal();
  });
  document.getElementById('profile-options-done').addEventListener('click', () => {
    if (ageField.reportValidity()) profileOptions.close();
  });
  document.querySelectorAll('[data-close-profile-sheet]').forEach(button => {
    button.addEventListener('click', () => button.closest('dialog').close());
  });
  document.getElementById('profile-form').addEventListener('invalid', event => {
    const sheet = event.target.closest('dialog');
    if (sheet && !sheet.open) sheet.showModal();
  }, true);
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && currentUser && !document.querySelector('dialog[open]')
        && !document.getElementById('profile-modal').hidden) {
      closeProfileModal();
    }
  });
  
  // Developer Toggle
  const devToggle = document.getElementById('dev-mode-toggle');
  if (devToggle) {
    devToggle.addEventListener('change', (e) => {
      if (e.target.checked) {
        document.getElementById('app-container').classList.add('debug-active');
        document.getElementById('debug-panel').style.display = 'flex';
      } else {
        document.getElementById('app-container').classList.remove('debug-active');
        document.getElementById('debug-panel').style.display = 'none';
      }
    });
  }

  // Load returning profiles list
  loadExistingUsers();
  setStatus('Your camera is off.');
}

// ── User Management & Profile Gating ───────────────────────────
async function loadExistingUsers() {
  try {
    const res = await fetch('/api/users');
    const data = await res.json();
    const users = data.users || [];
    const container = document.getElementById('user-chips-list');
    const button = document.getElementById('returning-profiles-button');
    button.hidden = users.length === 0;

    if (users.length > 0 && container) {
      container.innerHTML = '';
      users.forEach(u => {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'button secondary compact user-chip';
        chip.innerHTML = iconMarkup('user') + '<span>' + escapeHTML(u.name) + '</span><small>' + escapeHTML(u.role || 'Student') + '</small>';
        chip.onclick = () => selectUserChip(u);
        container.appendChild(chip);
      });
    }
  } catch (err) {
    console.error("Failed to load existing users:", err);
  }
}

async function selectUserChip(user) {
  document.getElementById('returning-users-box').close();
  currentUser = user;
  userSessions = [];
  document.getElementById('input-user-name').value = user.name;
  document.getElementById('input-user-role').value = user.role || '';
  document.getElementById('input-user-age').value = user.age || '';
  document.getElementById('input-user-concern').value = user.notes || '';

  activateUserProfile(user);
  try {
    const response = await fetch('/api/users/' + encodeURIComponent(user.id) + '/sessions');
    const data = await response.json();
    if (currentUser?.id === user.id) {
      userSessions = data.sessions || [];
      document.getElementById('nav-session-count').textContent = userSessions.length;
    }
  } catch (error) {
    console.error('Unable to load past check-ins:', error);
  }
}

async function handleProfileSubmit(event) {
  if (event) event.preventDefault();
  const nameField = document.getElementById('input-user-name');
  const name = nameField.value.trim();
  const errorMessage = document.getElementById('profile-error');
  errorMessage.hidden = true;
  if (!name) {
    errorMessage.textContent = 'Enter the name you would like us to use.';
    errorMessage.hidden = false;
    nameField.focus();
    return;
  }
  
  const role = document.getElementById('input-user-role').value.trim() || 'Student';
  const age = document.getElementById('input-user-age').value.trim();
  const notes = document.getElementById('input-user-concern').value.trim();
  const submit = document.querySelector('#profile-form button[type="submit"]');
  if (submit.disabled) return;
  submit.disabled = true;
  submit.querySelector('span').textContent = 'Opening your check-in…';

  try {
    const res = await fetch('/api/users', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, role, age, notes }),
      signal: AbortSignal.timeout(6500)
    });
    const data = await res.json();
    if (!res.ok || !data.user) {
      errorMessage.textContent = data.error || 'Your profile could not be opened. Please try again.';
      errorMessage.hidden = false;
      return;
    }
    currentUser = data.user;
    userSessions = data.recent_sessions || [];
    activateUserProfile(currentUser);
  } catch (err) {
    console.error("Profile submit error:", err);
    errorMessage.textContent = 'Samvya could not open your profile. Check the connection and try again.';
    errorMessage.hidden = false;
  } finally {
    submit.disabled = false;
    submit.querySelector('span').textContent = 'Begin my check-in';
  }
}

function activateUserProfile(user) {
  // Update header badge
  const badge = document.getElementById('user-profile-badge');
  const nameEl = document.getElementById('nav-user-name');
  const roleEl = document.getElementById('nav-user-role');
  const countEl = document.getElementById('nav-session-count');
  
  if (badge) badge.style.display = 'flex';
  if (nameEl) nameEl.textContent = user.name;
  if (roleEl) roleEl.textContent = user.role || 'Student';
  if (countEl) countEl.textContent = (userSessions || []).length;
  
  closeProfileModal(false);
  restart();
}

function openProfileModal() {
  stopChatActivity();
  profileTrigger = document.activeElement;
  document.getElementById('profile-modal').hidden = false;
  document.getElementById('close-profile').hidden = !currentUser;
  document.getElementById('profile-error').hidden = true;
  document.getElementById('workspace').inert = true;
  document.getElementById('workspace-header').inert = true;
  loadExistingUsers();
  document.getElementById('input-user-name').focus();
}

function closeProfileModal(restoreFocus = true) {
  document.getElementById('profile-modal').hidden = true;
  document.getElementById('workspace').inert = false;
  document.getElementById('workspace-header').inert = false;
  if (restoreFocus) profileTrigger?.focus();
}

async function startCameraAndScan() {
  const scanSession = sessionId;
  const button = document.getElementById('start-camera');
  const label = document.getElementById('camera-button-text');
  button.disabled = true;
  label.textContent = 'Starting camera…';
  setStatus('Waiting for camera permission.');

  try {
    // HD video: the expression models need about 200+ pixels across the face (see face-tracker.js).
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false
    });
    if (scanSession !== sessionId || isLocked) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }
    video.onloadedmetadata = () => {
      if (scanSession !== sessionId || isLocked) return;
      clearTimeout(cameraReadyTimer);
      captureCanvas.width = video.videoWidth;
      captureCanvas.height = video.videoHeight;
      overlayCanvas.width = video.videoWidth;
      overlayCanvas.height = video.videoHeight;
      document.getElementById('screen-scan').classList.add('camera-active');
      label.textContent = 'Camera on';
      setStatus('Centre your face in the guide.');
      beginCheckIn(scanSession);
    };
    video.srcObject = stream;
    cameraReadyTimer = setTimeout(() => {
      if (video.srcObject === stream) cameraError('The camera is taking too long. Try again or continue without it.');
    }, 8000);
    stream.getVideoTracks().forEach(track => {
      track.addEventListener('ended', () => {
        if (video.srcObject === stream) cameraError('The camera disconnected. Try again or continue without it.');
      }, { once: true });
    });
  } catch (error) {
    if (scanSession !== sessionId || isLocked) return;
    cameraError('Camera unavailable. Check your permissions or continue without it.');
    console.error('Camera error:', error);
  }
}

function cameraError(message) {
  stopScanning();
  stopCamera();
  scanStartTime = null;
  document.getElementById('lock-row').style.display = 'none';
  document.getElementById('start-camera').disabled = false;
  document.getElementById('camera-button-text').textContent = 'Try camera again';
  setStatus(message);
}

function continueWithoutCamera() {
  cameraStatus = 'off';
  lockedEmotion = null;
  latestScore = latestRisk = latestConfidence = null;
  isLocked = true;
  stopScanning();
  stopCamera();
  showScreen('screen-chat');
}

// ── Camera check-in ───────────────────────────────────────────
// With the face tracker (static/js/face-tracker.js), each frame is checked in the browser and
// only clear frames are classified: a level face crop is sent to /detect. The check-in ends
// after CHECKIN_FRAMES clear frames (or CHECKIN_TIMEOUT), and the server reports a label only
// when the model is confident; otherwise the cue is "unclear". Without the tracker, whole
// frames are sent and the server finds the face.
const CHECKIN_FRAMES = 12;
const CHECKIN_TIMEOUT = 30000;
const SEND_INTERVAL = 250;
const TRACK_INTERVAL = 66;
const MONITOR_INTERVAL = 3000;
const cropCanvas = document.createElement('canvas');
let trackerModule = null;
let faceLandmarker = null;
let trackerFailed = false;
let tracker = null;
let lastTrackAt = 0;
let lastSentAt = 0;
let detectInFlight = false;
let lastCheckIn = null;
let lastHeatmap = null;
let cameraStatus = 'off';        // off | clear | unclear
let monitorTimer = null;
let monitorPaused = false;
let monitorFrames = 0;

async function loadFaceTracker() {
  if (faceLandmarker || trackerFailed) return faceLandmarker;
  try {
    trackerModule = await import('/static/js/face-tracker.js');
    faceLandmarker = await trackerModule.createFaceLandmarker('VIDEO');
  } catch (error) {
    trackerFailed = true;
    console.warn('Face tracker unavailable; the server will find the face.', error);
  }
  return faceLandmarker;
}

async function beginCheckIn(scanSession) {
  setStatus('Preparing the face tracker…');
  await loadFaceTracker();
  if (scanSession !== sessionId || isLocked || !video.srcObject) return;
  scanStartTime = Date.now();
  lastCheckIn = null;
  lastHeatmap = null;
  document.getElementById('lock-row').style.display = 'flex';
  updateCheckInProgress(0);
  isScanning = true;
  if (faceLandmarker) {
    tracker = new trackerModule.FaceTracker(faceLandmarker);
    requestAnimationFrame(() => trackLoop(scanSession));
  } else {
    document.getElementById('keep-camera-option').hidden = true;  // the chat check needs the tracker
    scanLoop(scanSession);
  }
}

function stopScanning() {
  isScanning = false;
}

// Check one frame of `source`; crops it into cropCanvas when it is clear.
function assessFrame(source, now) {
  const view = tracker.track(source, now);
  if (view.face && !view.reason) {
    trackerModule.alignedCrop(source, view.geometry, cropCanvas);
    const light = trackerModule.brightness(cropCanvas);
    if (light < trackerModule.LIMITS.minBrightness) view.reason = 'dark';
    else if (light > trackerModule.LIMITS.maxBrightness) view.reason = 'bright';
  }
  return view;
}

function trackLoop(scanSession) {
  if (!isScanning || scanSession !== sessionId) return;
  const now = performance.now();
  if (now - lastTrackAt >= TRACK_INTERVAL && video.readyState >= 2) {
    lastTrackAt = now;
    const view = assessFrame(video, now);
    drawFaceOverlay(view);
    setStatus(view.reason ? trackerModule.QUALITY_HINTS[view.reason] : 'Clear view. Hold this for a moment.');
    if (!view.reason && !detectInFlight && now - lastSentAt >= SEND_INTERVAL) {
      lastSentAt = now;
      sendFace('checkin', view.cues, scanSession);
    }
    if (Date.now() - scanStartTime >= CHECKIN_TIMEOUT && !detectInFlight) {
      finishCheckIn(lastCheckIn);
      return;
    }
  }
  requestAnimationFrame(() => trackLoop(scanSession));
}

function drawFaceOverlay(view) {
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  if (!view.face) return;
  const { box, roll, centre } = view.geometry;
  overlayCtx.save();
  // The box is in the levelled frame; rotate it back onto the picture.
  overlayCtx.translate(centre.x, centre.y);
  overlayCtx.rotate(roll);
  overlayCtx.translate(-centre.x, -centre.y);
  if (heatmapEnabled && lastHeatmap) {
    overlayCtx.globalAlpha = 0.75;
    overlayCtx.drawImage(lastHeatmap, box.minX, box.minY, box.width, box.height);
    overlayCtx.globalAlpha = 1;
  }
  overlayCtx.strokeStyle = view.reason ? '#d89a3a' : (getComputedStyle(document.body).getPropertyValue('--accent').trim() || '#34745f');
  overlayCtx.lineWidth = Math.max(2, box.width / 90);
  overlayCtx.strokeRect(box.minX, box.minY, box.width, box.height);
  overlayCtx.restore();
}

async function sendFace(phase, cues, scanSession) {
  detectInFlight = true;
  const image = cropCanvas.toDataURL('image/jpeg', 0.9);
  try {
    const res = await fetch('/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image, face_crop: true, phase, cues, session_id: scanSession,
        heatmap: phase === 'checkin' && heatmapEnabled
      }),
      signal: AbortSignal.timeout(10000)
    });
    const data = await res.json();
    if (scanSession !== sessionId) return;
    if (!res.ok) throw new Error(data.error || 'Camera check-in unavailable');
    if (phase === 'checkin') handleCheckInFrame(data);
    else handleMonitorFrame(data);
  } catch (error) {
    if (scanSession !== sessionId) return;
    if (phase === 'checkin' && isScanning) {
      cameraError('The camera check-in is unavailable. You can continue to the conversation.');
    } else if (phase === 'chat') {
      updateMonitorCard('Camera check unavailable right now.');
    }
    console.error('Detect error:', error);
  } finally {
    detectInFlight = false;
  }
}

function handleCheckInFrame(data) {
  if (isLocked || !data.face_detected) return;
  lastCheckIn = data;
  if (data.final_ms !== undefined) latestScore = data.final_ms;
  if (data.risk !== undefined) latestRisk = data.risk;
  if (data.confidence !== undefined) latestConfidence = data.confidence;
  if (data.heatmap_b64) {
    const img = new Image();
    img.onload = () => { lastHeatmap = img; };
    img.src = 'data:image/jpeg;base64,' + data.heatmap_b64;
  }
  setEmotionBadge(data.clear ? data.emotion : null, data.confidence);
  updateCheckInProgress(data.clear_frames);
  if (data.clear_frames >= CHECKIN_FRAMES) finishCheckIn(data);
}

function updateCheckInProgress(frames) {
  const shown = Math.min(frames, CHECKIN_FRAMES);
  document.getElementById('timer-fill').style.width = `${(shown / CHECKIN_FRAMES) * 100}%`;
  document.getElementById('lock-text').textContent = `Clear views ${shown} / ${CHECKIN_FRAMES}`;
}

// The server decides: a label only when the model is confident over enough clear frames.
function finishCheckIn(data) {
  if (isLocked) return;
  if (data && data.clear) {
    latestScore = data.final_ms;
    latestRisk = data.risk;
    latestConfidence = data.confidence;
    cameraStatus = 'clear';
    lockEmotion(data.emotion);
  } else {
    latestScore = latestRisk = null;
    latestConfidence = data ? data.confidence : null;
    cameraStatus = 'unclear';
    lockEmotion(null);
  }
}

// Without the face tracker: send whole frames and let the server find the face.
async function scanLoop(scanSession) {
  if (!isScanning || scanSession !== sessionId) return;
  await captureAndDetect(scanSession);
  if (!isScanning || scanSession !== sessionId) return;
  if (Date.now() - scanStartTime >= CHECKIN_TIMEOUT) {
    finishCheckIn(lastCheckIn);
    return;
  }
  setTimeout(() => scanLoop(scanSession), 110);
}

async function captureAndDetect(scanSession) {
  if (!video.videoWidth || video.readyState < 2) return;
  try {
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    const b64 = captureCanvas.toDataURL('image/jpeg', 0.75);
    const res = await fetch('/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: b64, heatmap: heatmapEnabled, session_id: scanSession, phase: 'checkin' }),
      signal: AbortSignal.timeout(10000)
    });
    const data = await res.json();
    if (scanSession !== sessionId || !isScanning) return;
    if (!res.ok) throw new Error(data.error || 'Camera check-in unavailable');
    handleDetection(data);
  } catch (error) {
    if (scanSession !== sessionId || !isScanning) return;
    cameraError('The camera check-in is unavailable. You can continue to the conversation.');
    console.error('Detect error:', error);
  }
}

async function fetchDebugState() {
  if (!document.getElementById('app-container').classList.contains('debug-active')) return;
  try {
    const res = await fetch('/api/debug/state?session_id=' + encodeURIComponent(sessionId));
    const state = await res.json();
    renderDebugState(state);
  } catch (err) {
    console.error("Debug state error:", err);
  }
}

function handleDetection(data) {
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  if (!data.face_detected) {
    setStatus('No face detected — look at the camera.');
    return;
  }
  if (data.face_box) {
    const [x, y, w, h] = data.face_box;
    overlayCtx.strokeStyle = getComputedStyle(document.body).getPropertyValue('--accent').trim() || '#34745f';
    overlayCtx.lineWidth = 2;
    overlayCtx.strokeRect(x, y, w, h);
  }
  setStatus('Face found. Hold still for a moment.');
  handleCheckInFrame(data);
}

// ── Camera during the conversation (optional) ─────────────────
function startMonitor() {
  const preview = document.getElementById('chat-camera-video');
  preview.srcObject = video.srcObject;
  preview.play().catch(() => {});
  monitorPaused = false;
  monitorFrames = 0;
  tracker = new trackerModule.FaceTracker(faceLandmarker);
  document.getElementById('chat-camera').hidden = false;
  document.getElementById('chat-camera-pause').textContent = 'Pause';
  updateMonitorCard('Camera check on. One view every few seconds.');
  scheduleMonitor(sessionId, 1000);
}

function scheduleMonitor(scanSession, delay) {
  clearTimeout(monitorTimer);
  monitorTimer = setTimeout(() => monitorTick(scanSession), delay);
}

async function monitorTick(scanSession) {
  const preview = document.getElementById('chat-camera-video');
  if (scanSession !== sessionId || !preview.srcObject) return;
  if (monitorPaused || document.hidden || preview.readyState < 2) {
    scheduleMonitor(scanSession, MONITOR_INTERVAL);
    return;
  }
  try {
    // Look for one clear frame for up to 1.5 s.
    const until = performance.now() + 1500;
    let view = null;
    while (performance.now() < until && scanSession === sessionId) {
      view = assessFrame(preview, performance.now());
      if (!view.reason) break;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    if (scanSession !== sessionId) return;
    if (view && !view.reason) await sendFace('chat', view.cues, scanSession);
    else updateMonitorCard('Waiting for a clear view: ' + trackerModule.QUALITY_HINTS[view?.reason || 'no_face'].toLowerCase());
  } catch (error) {
    console.error('Camera check error:', error);
    updateMonitorCard('Camera check is retrying.');
  }
  if (scanSession === sessionId) scheduleMonitor(scanSession, MONITOR_INTERVAL);
}

function handleMonitorFrame(data) {
  if (!data.face_detected) return;
  monitorFrames += 1;
  // data.cue describes the last two minutes of clear views (server side, services/camera_cues.py).
  const cue = data.cue || {};
  const reading = cue.clear && cue.emotion
    ? `Expression in the last 2 min: mostly ${cue.emotion.toLowerCase()}`
    : `Expression in the last 2 min: ${cue.frames >= 5 ? 'unclear' : 'still reading'}`;
  updateMonitorCard(`${reading} · ${monitorFrames} clear ${monitorFrames === 1 ? 'view' : 'views'}`);
}

function updateMonitorCard(text) {
  document.getElementById('chat-camera-status').textContent = text;
}

function toggleMonitorPause() {
  monitorPaused = !monitorPaused;
  document.getElementById('chat-camera-pause').textContent = monitorPaused ? 'Resume' : 'Pause';
  updateMonitorCard(monitorPaused ? 'Camera check paused.' : 'Camera check on. One view every few seconds.');
}

function stopMonitor() {
  clearTimeout(monitorTimer);
  monitorTimer = null;
  const preview = document.getElementById('chat-camera-video');
  if (preview) preview.srcObject = null;
  const card = document.getElementById('chat-camera');
  if (card) card.hidden = true;
}

function turnOffChatCamera() {
  stopMonitor();
  stopCamera();
}

function setEmotionBadge(emotion, confidence) {
  const cfg = EMOTION_CFG[emotion];
  document.getElementById('e-label').textContent = cfg
    ? `${cfg.label} · ${confidence.toFixed(0)}%`
    : 'Reading your expression…';
  document.getElementById('conf-fill').style.width = `${Math.min(confidence || 0, 100)}%`;
}

function setStatus(msg) {
  document.getElementById('status-text').textContent = msg;
}

// ── Lock emotion & transition ─────────────────────────────────
function lockEmotion(emotion) {
  lockedEmotion = emotion;
  isLocked = true;
  stopScanning();
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  const keepCamera = document.getElementById('keep-camera').checked && faceLandmarker && video.srcObject;
  if (keepCamera) startMonitor();
  else stopCamera();
  document.body.setAttribute('data-emotion', emotion || 'Neutral');
  setStatus(emotion
    ? `Facial expression read as mostly ${emotion.toLowerCase()}.`
    : 'The camera view was not clear enough for an expression reading.');
  saveSessionProgress();
  showScreen('screen-chat');
}

function stopCamera() {
  clearTimeout(cameraReadyTimer);
  stopMonitor();
  if (video && video.srcObject) {
    const tracks = video.srcObject.getTracks();
    tracks.forEach(track => track.stop());
    video.srcObject = null;
    video.onloadedmetadata = null;
  }
  document.getElementById('screen-scan').classList.remove('camera-active');
}

// ── SCREEN 2: CHAT ───────────────────────────────────────
function initChatUI() {
  const sendButton = document.getElementById('btn-send');
  const chatInput = document.getElementById('chat-input');

  sendButton.addEventListener('click', () => {
    const text = chatInput.value.trim();
    if (!text || finishingCheckIn) return;
    chatInput.value = '';
    sendChatMessage(text);
    chatInput.focus();
  });

  chatInput.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.isComposing) {
      event.preventDefault();
      sendButton.click();
    }
  });

  document.querySelectorAll('[data-starter]').forEach(button => {
    button.addEventListener('click', () => {
      chatInput.value = button.dataset.starter;
      chatInput.focus();
    });
  });
  document.getElementById('btn-voice').addEventListener('click', toggleRecording);
  document.getElementById('chat-activity-action').addEventListener('click', () => {
    if (voiceCapture) {
      cancelVoiceCapture();
      document.getElementById('btn-voice').focus();
    } else if (activeNarration?.phase === 'ready') {
      playNarration(activeNarration);
    } else {
      stopNarration();
      chatInput.focus();
    }
    updateChatActivity();
  });
  document.getElementById('screen-chat').addEventListener('keydown', event => {
    if (event.key !== 'Escape' || !voiceCapture) return;
    event.preventDefault();
    cancelVoiceCapture();
    updateChatActivity();
    document.getElementById('btn-voice').focus();
  });
  window.addEventListener('pagehide', () => {
    stopChatActivity();
    stopWearablePolling();
  });
  window.addEventListener('pageshow', event => {
    if (event.persisted && currentUser && !document.getElementById('screen-results').classList.contains('active')) {
      startWearablePolling();
    }
  });
}

function updateChatActivity() {
  const status = document.getElementById('chat-activity-status');
  const panel = document.getElementById('chat-activity');
  const action = document.getElementById('chat-activity-action');
  const microphone = document.getElementById('btn-voice');
  const recording = voiceCapture?.phase === 'recording';
  const micLabel = recording ? 'Stop recording and send' : 'Record a voice note';

  microphone.disabled = finishingCheckIn || Boolean(voiceCapture && !recording);
  microphone.classList.toggle('recording', recording);
  microphone.title = micLabel;
  microphone.setAttribute('aria-label', micLabel);
  microphone.querySelector('use').setAttribute('href', '/static/images/icons.svg#' + (recording ? 'stop' : 'mic'));
  document.getElementById('btn-send').disabled = finishingCheckIn;
  document.getElementById('finish-session').disabled = finishingCheckIn;

  const phase = voiceCapture?.phase || (activeChatRequest && 'replying')
    || activeNarration?.phase || (chatNotice && 'notice');
  const labels = {
    starting: 'Starting microphone…',
    recording: 'Listening… Your message sends when you finish speaking.',
    transcribing: 'Transcribing…',
    replying: 'Preparing a reply…',
    loading: 'Preparing audio…',
    speaking: 'Samvya is speaking',
    ready: 'Audio ready',
    notice: chatNotice
  };
  const label = labels[phase] || '';
  panel.hidden = !phase;
  panel.dataset.phase = phase || '';
  if (status.textContent !== label) status.textContent = label;
  if (!voiceCapture) updateVoicePreview('');
  const actionLabel = voiceCapture ? 'Cancel' : activeNarration?.phase === 'ready'
    ? 'Play reply' : activeNarration ? 'Stop audio' : '';
  action.hidden = !actionLabel;
  if (action.textContent !== actionLabel) action.textContent = actionLabel;
  if (actionLabel === 'Cancel') action.setAttribute('aria-label', 'Cancel voice note');
  else action.removeAttribute('aria-label');
}

function updateVoicePreview(text) {
  const preview = document.getElementById('chat-voice-preview');
  if (preview.textContent !== text) preview.textContent = text;
  preview.hidden = !text;
}

async function toggleRecording() {
  if (voiceCapture?.phase === 'recording') {
    finishVoiceCapture(voiceCapture);
    return;
  }
  if (voiceCapture || finishingCheckIn) return;
  stopChatActivity();

  const capture = new ChatVoiceInput(
    text => {
      if (voiceCapture !== capture) return;
      updateVoicePreview(text);
    },
    () => finishVoiceCapture(capture)
  );
  voiceCapture = capture;
  capture.phase = 'starting';
  updateChatActivity();
  try {
    await capture.start();
    if (voiceCapture !== capture) return;
    capture.phase = 'recording';
    updateChatActivity();
  } catch {
    if (voiceCapture !== capture) return;
    cancelVoiceCapture();
    chatNotice = 'Microphone access is unavailable. You can still type a message.';
    updateChatActivity();
  }
}

async function finishVoiceCapture(capture) {
  if (voiceCapture !== capture || capture.phase !== 'recording') return;
  capture.phase = 'transcribing';
  updateChatActivity();
  try {
    const transcript = await capture.finish();
    if (voiceCapture !== capture) return;
    voiceCapture = null;
    sendChatMessage(transcript);
  } catch (error) {
    if (voiceCapture !== capture) return;
    cancelVoiceCapture();
    chatNotice = error instanceof TypeError || error.name === 'TimeoutError'
      ? 'Transcription is unavailable. Try again or type a message.'
      : error.message;
    updateChatActivity();
  }
}

function cancelVoiceCapture() {
  const capture = voiceCapture;
  voiceCapture = null;
  capture?.cancel();
}

function stopNarration() {
  const narration = activeNarration;
  if (!narration) return;
  activeNarration = null;
  narration.controller.abort();
  if (narration.utterance && 'speechSynthesis' in window) window.speechSynthesis.cancel();
  if (narration.audio) {
    narration.audio.pause();
    narration.audio.removeAttribute('src');
    narration.audio.load();
  }
  if (narration.url) URL.revokeObjectURL(narration.url);
}

async function playNarration(narration) {
  try {
    await narration.audio.play();
    if (activeNarration !== narration) return;
    narration.phase = 'speaking';
  } catch (error) {
    if (activeNarration !== narration) return;
    if (error.name === 'NotAllowedError') narration.phase = 'ready';
    else if (narration.fallback) narration.fallback();
    else stopNarration();
  }
  updateChatActivity();
}

// Read a reply with the browser's built-in voice: used when the ElevenLabs voice is unavailable
// (for example when its monthly character quota is used up).
function speakWithBrowser(narration, text) {
  if (!text || !('speechSynthesis' in window)) return false;
  const synth = window.speechSynthesis;
  const utterance = new SpeechSynthesisUtterance(text);
  const devanagari = /[\u0900-\u097F]/.test(text);
  utterance.lang = devanagari ? (/(आहे|नाही|मला|तुम्ही|आणि)/.test(text) ? 'mr-IN' : 'hi-IN') : 'en-IN';
  const voices = synth.getVoices();
  utterance.voice = voices.find(voice => voice.lang === utterance.lang)
    || voices.find(voice => voice.lang.startsWith(utterance.lang.slice(0, 2))) || null;
  const finish = () => {
    if (activeNarration !== narration) return;
    stopNarration();
    updateChatActivity();
  };
  utterance.addEventListener('end', finish);
  utterance.addEventListener('error', finish);
  narration.utterance = utterance;
  narration.phase = 'speaking';
  synth.cancel();
  synth.speak(utterance);
  updateChatActivity();
  return true;
}

// Replies stream from the server (ElevenLabs, Microsoft neural voices or offline Piper; see
// services/speech_synthesis.py), so playback starts with the first audio. If no server voice
// can speak, the browser's own voice reads the reply.
async function startNarration(ticket, text) {
  const narration = { controller: new AbortController(), phase: 'loading' };
  activeNarration = narration;
  updateChatActivity();
  const audio = new Audio();
  narration.audio = audio;
  let started = false;
  let fellBack = false;
  const finish = () => {
    if (activeNarration !== narration) return;
    const restoreFocus = document.activeElement === document.getElementById('chat-activity-action');
    stopNarration();
    updateChatActivity();
    if (restoreFocus) document.getElementById('chat-input').focus();
  };
  narration.fallback = () => {
    if (fellBack || activeNarration !== narration) return;
    fellBack = true;
    audio.pause();
    audio.removeAttribute('src');
    narration.audio = null;
    if (!speakWithBrowser(narration, text)) {
      stopNarration();
      updateChatActivity();
    }
  };
  audio.addEventListener('playing', () => { started = true; }, { once: true });
  audio.addEventListener('ended', finish, { once: true });
  audio.addEventListener('error', () => {
    if (activeNarration !== narration || fellBack) return;
    if (started) finish();
    else narration.fallback();
  }, { once: true });
  audio.src = '/chat/speech/stream?ticket=' + encodeURIComponent(ticket);
  await playNarration(narration);
}

function stopChatActivity() {
  activeChatRequest?.controller.abort();
  activeChatRequest = null;
  cancelVoiceCapture();
  stopNarration();
  chatNotice = '';
  updateChatActivity();
}

function appendMessage(sender, text) {
  const container = document.getElementById('chat-messages');
  const message = document.createElement('div');
  message.className = 'chat-bubble ' + sender;
  const author = document.createElement('span');
  author.className = 'message-author';
  author.textContent = sender === 'user' ? 'You' : 'Samvya';
  const body = document.createElement('p');
  body.textContent = text;
  message.append(author, body);
  container.appendChild(message);
  container.scrollTop = container.scrollHeight;
  return message;
}

async function sendChatMessage(text) {
  stopChatActivity();
  if (text.length > 4000) {
    chatNotice = 'Enter a message of up to 4,000 characters.';
    updateChatActivity();
    return;
  }
  const chatSession = sessionId;
  const request = { controller: new AbortController() };
  activeChatRequest = request;
  const userMessage = { role: 'user', content: text };
  chatHistory.push(userMessage);
  const bubble = appendMessage('user', text);
  document.getElementById('chat-starters').hidden = true;
  updateChatActivity();

  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        emotion: lockedEmotion || 'Neutral',
        history: chatHistory,
        session_id: chatSession,
        user_id: currentUser?.id || null,
        counsellor_voice: document.getElementById('counsellor-voice').value === 'male' ? 'male' : 'female',
        defer_audio: true
      }),
      signal: AbortSignal.any([request.controller.signal, AbortSignal.timeout(30000)])
    });
    const data = await response.json().catch(() => ({}));
    if (activeChatRequest !== request || chatSession !== sessionId) return;
    if (!response.ok || !data.response) {
      if (data.debug_state) renderDebugState(data.debug_state);
      throw new Error(data.error || 'Chat request failed');
    }

    activeChatRequest = null;
    if (data.debug_state) renderDebugState(data.debug_state);
    chatHistory.push({ role: 'assistant', content: data.response });
    appendMessage('bot', data.response);
    if (data.speech_ticket) startNarration(data.speech_ticket, data.response);
    else updateChatActivity();
    saveSessionProgress();
  } catch (error) {
    if (activeChatRequest !== request || chatSession !== sessionId) return;
    activeChatRequest = null;
    const draft = document.getElementById('chat-input');
    if (!draft.value) {
      draft.value = text;
      chatHistory.splice(chatHistory.indexOf(userMessage), 1);
      bubble.remove();
    }
    chatNotice = error.name === 'Error' && error.message !== 'Chat request failed'
      ? error.message
      : 'I couldn’t connect just now. Please try again.';
    updateChatActivity();
    console.error('Chat error:', error);
  }
}

// Report links are only protected by this ID, so it must not be guessable.
function newSessionId() {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return 'session_' + Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
}

let sessionId = newSessionId();

function sensorValue(value, unit, digits = 0, missing = 'Not recorded') {
  return Number.isFinite(value) ? value.toFixed(digits) + ' ' + unit : missing;
}

function sensorVector(values, unit, missing = 'Not recorded') {
  return values ? values.map(value => (Math.abs(value) < 0.005 ? 0 : value).toFixed(2)).join(' / ') + ' ' + unit : missing;
}

function formatAge(seconds) {
  if (!Number.isFinite(seconds) || seconds < 60) return Math.max(0, Math.round(seconds || 0)) + ' s ago';
  if (seconds < 3600) return Math.round(seconds / 60) + ' min ago';
  return Math.round(seconds / 3600) + ' h ago';
}

function signed(value, digits, unit) {
  return (value > 0 ? '+' : value < 0 ? '−' : '±') + Math.abs(value).toFixed(digits) + unit;
}

function conductanceLabel(comparison) {
  return { higher: 'higher than at the start', lower: 'lower than at the start', similar: 'similar to the start' }[comparison] ||
    'similar to the start';
}

function activityLabel(movement) {
  return movement < 0.15 ? 'Still' : movement < 1 ? 'Light movement' : 'Moving';
}

// Each reading keeps its last good value: "live" when just measured, otherwise the
// last measurement with its age. Nothing is estimated or filled in.
const PULSE_HINTS = {
  movement: 'Keep your hand still',
  contact: 'Place the sensor on your skin',
  device: 'Wearable not sending data',
  measuring: 'Needs about 10 s of stillness'
};

function vitalDisplay(name, vital, digits, unit) {
  vital = vital || { state: 'waiting' };
  if (Number.isFinite(vital.value)) {
    let state = vital.state === 'live' ? 'Live' : 'Last reading · ' + formatAge(vital.age_s);
    if (name === 'motion' && vital.state === 'live') state = activityLabel(vital.value);
    if (name === 'pulse' && vital.state !== 'live' && vital.reason) state += ' · ' + PULSE_HINTS[vital.reason];
    return { value: vital.value.toFixed(digits) + ' ' + unit, state, live: vital.state === 'live' ? 'live' : 'held' };
  }
  if (name === 'spo2' && vital.state === 'unsupported') {
    return { value: 'Not at this spot', state: 'Works on the inner wrist or a fingertip', live: 'waiting' };
  }
  if (name === 'temperature' && vital.state === 'sensor_missing') {
    return { value: 'Not connected', state: 'Check the sensor wiring', live: 'waiting' };
  }
  if (name === 'pulse') {
    return { value: 'Measuring…', state: PULSE_HINTS[vital.reason] || PULSE_HINTS.measuring, live: 'waiting' };
  }
  return { value: 'Waiting…', state: 'Not measured yet', live: 'waiting' };
}

function renderTrends(insights) {
  const section = document.getElementById('wearable-trends');
  const list = document.getElementById('wearable-trend-list');
  const rows = [];
  if (insights) {
    const trendWord = trend => trend ? ', ' + trend.direction + ' over the last minute' : '';
    if (insights.pulse) {
      const pulse = insights.pulse;
      rows.push(['Pulse', 'Average ' + pulse.average.toFixed(0) + ' BPM (' + pulse.min.toFixed(0) + '–' + pulse.max.toFixed(0) +
        '), ' + signed(pulse.change, 0, ' BPM') + ' vs start' + trendWord(pulse.trend)]);
      rows.push(['Pulse coverage', 'Measured ' + pulse.coverage + '% of the time (the hand must be still)']);
    }
    if (insights.temperature) {
      const temperature = insights.temperature;
      rows.push(['Skin temperature', temperature.current.toFixed(2) + ' °C, ' + signed(temperature.change, 2, ' °C') +
        ' vs start' + trendWord(temperature.trend)]);
    }
    if (insights.gsr && insights.gsr.change_percent !== null) {
      const gsr = insights.gsr;
      // The GSR voltage falls when skin conductance rises, so describe the conductance.
      const conductanceTrend = gsr.skin_conductance_trend ? ', ' + gsr.skin_conductance_trend + ' over the last minute' : '';
      rows.push(['GSR', gsr.current + ' mV (' + signed(gsr.change_percent, 1, '%') + ' vs start): skin conductance ' +
        conductanceLabel(gsr.skin_conductance_vs_start) + conductanceTrend]);
    }
    if (insights.movement) {
      const movement = insights.movement;
      rows.push(['Movement', 'Still ' + movement.still_percent + '% of the check-in, moving ' + movement.active_percent + '%']);
    }
    if (insights.elapsed_s < 60 && rows.length) rows.push(['Baseline', 'Collecting the first minute…']);
  }
  section.hidden = !rows.length;
  const html = rows.map(([label, text]) => '<div><dt>' + escapeHTML(label) + '</dt><dd>' + escapeHTML(text) + '</dd></div>').join('');
  if (list.innerHTML !== html) list.innerHTML = html;
}

function renderWearable(snapshot) {
  const status = document.getElementById('wearable-status');
  const note = document.getElementById('wearable-pulse-note');
  const vitals = snapshot.vitals || {};
  const sample = snapshot.sample || snapshot.last_sample || {};
  const statusLabels = {
    live: 'Live via USB',
    waiting: 'Waiting for device data',
    disconnected: 'No device connected · showing last readings',
    stale: 'Readings paused · showing last readings',
    disabled: 'USB readings off',
    error: 'Connection to Samvya lost · showing last readings'
  };
  if (status.textContent !== statusLabels[snapshot.status]) status.textContent = statusLabels[snapshot.status];
  status.dataset.status = snapshot.status;
  note.textContent = vitals.pulse && vitals.pulse.state !== 'live' && vitals.pulse.reason === 'movement'
    ? 'Pulse is measured while your hand is still. The last reading stays on screen until a new one is ready.'
    : '';
  note.hidden = !note.textContent;
  const shown = {
    hr: vitalDisplay('pulse', vitals.pulse, 0, 'BPM'),
    spo2: vitalDisplay('spo2', vitals.spo2, 1, '%'),
    temp: vitalDisplay('temperature', vitals.temperature, 2, '°C'),
    gsr: vitalDisplay('gsr', vitals.gsr, 0, 'mV'),
    motion: vitalDisplay('motion', vitals.motion, 2, 'm/s²')
  };
  Object.entries(shown).forEach(([key, display]) => {
    const output = document.getElementById('wearable-' + key);
    const state = document.getElementById('wearable-' + key + '-state');
    if (output.textContent !== display.value) output.textContent = display.value;
    if (state.textContent !== display.state) state.textContent = display.state;
    state.dataset.state = display.live;
  });
  const details = {
    ir: sensorValue(sample.ir, 'counts'),
    'gsr-adc': sensorValue(sample.gsr_adc, 'ADC'),
    xyz: sensorVector(sample.accel, 'm/s²'),
    gyro: sensorVector(sample.gyro, 'rad/s')
  };
  Object.entries(details).forEach(([key, value]) => {
    const output = document.getElementById('wearable-' + key);
    if (output.textContent !== value) output.textContent = value;
  });
  document.getElementById('wearable-details').hidden = !snapshot.sample && !snapshot.last_sample;
  renderTrends(snapshot.insights);
  renderWearableDebug(snapshot.sample ? { source: 'USB', ...sample, vitals } : null);
}

function startWearablePolling() {
  stopWearablePolling();
  document.getElementById('wearable-bar').hidden = false;
  const poll = { controller: new AbortController() };
  wearablePoll = poll;
  const checkIn = sessionId;
  async function refresh() {
    try {
      const response = await fetch('/api/wearable?session_id=' + encodeURIComponent(checkIn), {
        signal: AbortSignal.any([poll.controller.signal, AbortSignal.timeout(3500)])
      });
      if (!response.ok) throw new Error('Wearable unavailable');
      const snapshot = await response.json();
      if (wearablePoll !== poll) return;
      wearableSummary = snapshot.summary;
      renderWearable(snapshot);
      wearableShown = true;
    } catch {
      if (wearablePoll !== poll) return;
      // Keep the last readings on screen; only the connection status changes.
      if (wearableShown) {
        const status = document.getElementById('wearable-status');
        status.textContent = 'Connection to Samvya lost · showing last readings';
        status.dataset.status = 'error';
      } else {
        renderWearable({ status: 'error' });
      }
    }
    poll.timer = setTimeout(refresh, 1000);
  }
  wearableShown = false;
  renderWearable({ status: 'waiting' });
  refresh();
}

function stopWearablePolling() {
  if (!wearablePoll) return;
  wearablePoll.controller.abort();
  clearTimeout(wearablePoll.timer);
  wearablePoll = null;
}

function showChat() {
  document.getElementById('chat-messages').replaceChildren();
  document.getElementById('chat-starters').hidden = false;
  document.getElementById('chat-input').value = '';
  document.getElementById('app-container').classList.add('focus-mode');

  const name = currentUser?.name || 'there';
  const greeting = userSessions.length
    ? 'Welcome back, ' + name + '. What feels important to talk about today?'
    : 'Hi ' + name + '. I’m Samvya, your AI wellbeing companion. What’s been on your mind?';

  chatHistory = [{ role: 'assistant', content: greeting }];
  appendMessage('bot', greeting);
}

function showScreen(id) {
  document.querySelectorAll('.right-panel .screen').forEach(screen => screen.classList.toggle('active', screen.id === id));
  document.querySelectorAll('#journey [data-stage]').forEach(item => {
    if ('screen-' + item.dataset.stage === id) item.setAttribute('aria-current', 'step');
    else item.removeAttribute('aria-current');
  });
  if (id === 'screen-chat') showChat();
  const heading = id === 'screen-idle'
    ? document.getElementById('scan-heading')
    : document.querySelector('#' + id + ' h1');
  window.scrollTo(0, 0);
  heading?.focus({ preventScroll: true });
}

// ── Finalize & Report Workflow ─────────────────────────────────
async function finishSessionAndShowReport() {
  if (finishingCheckIn) return;
  finishingCheckIn = true;
  stopChatActivity();
  stopWearablePolling();
  stopCamera();   // also ends the optional camera check during the chat
  
  const reportSession = sessionId;
  const finishBtn = document.getElementById('finish-session');
  const finishLabel = finishBtn.innerHTML;
  finishBtn.innerHTML = iconMarkup('loader', 'spinning') + '<span>Preparing…</span>';
  
  let summaryInsight = "You completed a wellbeing check-in.";
  try {
    const summaryRes = await fetch('/api/sessions/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: reportSession,
        transcript: chatHistory,
        emotion: lockedEmotion || 'Not captured',
        mental_health_score: latestScore,
        user_name: currentUser ? currentUser.name : 'Student'
      }),
      signal: AbortSignal.timeout(45000)   // the detailed counsellor brief takes longer than a recap
    });
    const summaryData = await summaryRes.json();
    if (summaryData.summary) summaryInsight = summaryData.summary;
  } catch (e) {
    console.error("Summary error:", e);
  }

  if (reportSession !== sessionId) return;
  const reportSaved = await saveSessionProgress(summaryInsight);
  if (reportSession !== sessionId) return;
  
  // Update local session cache count
  if (currentUser && currentUser.id) {
    try {
      const sessRes = await fetch(`/api/users/${currentUser.id}/sessions`);
      const sessData = await sessRes.json();
      if (reportSession !== sessionId) return;
      userSessions = sessData.sessions || [];
      const countEl = document.getElementById('nav-session-count');
      if (countEl) countEl.textContent = userSessions.length;
    } catch (e) {}
  }

  finishingCheckIn = false;
  updateChatActivity();
  finishBtn.innerHTML = finishLabel;
  updateReportSaveStatus(reportSaved);
  renderReportScreen(summaryInsight);
  document.getElementById('wearable-bar').hidden = true;
  showScreen('screen-results');
}

let safetyReview = null;   // Samvya's own safety flags for the report (services/risk_flags.py)

function renderSafetyFlag() {
  const box = document.getElementById('safety-flag');
  const review = safetyReview;
  if (!review || !review.flagged) {
    box.hidden = true;
    box.replaceChildren();
    return;
  }
  const lead = document.createElement('p');
  const title = document.createElement('strong');
  title.textContent = review.urgent ? 'Urgent: safety concern flagged' : 'Concern flagged';
  const count = review.messages === 1 ? '1 message' : review.messages + ' messages';
  lead.append(title, ' Samvya flagged ' + count + ' in this check-in for a counsellor to follow up. ' +
    (review.support_shown ? 'India support contacts (Tele MANAS 14416, emergency 112) were shared.' : ''));
  const parts = [lead];
  review.categories.forEach(category => {
    const label = document.createElement('p');
    label.className = 'safety-label';
    label.textContent = category.label;
    parts.push(label);
    category.statements.forEach(statement => {
      const quote = document.createElement('blockquote');
      quote.textContent = '“' + statement + '”';
      parts.push(quote);
    });
  });
  box.classList.toggle('urgent', review.urgent);
  box.replaceChildren(...parts);
  box.hidden = false;
}

function renderReportScreen(summaryInsight) {
  renderSafetyFlag();
  const hasScan = latestScore !== null;
  document.getElementById('ring-wrapper').hidden = !hasScan;
  document.getElementById('score-placeholder').hidden = hasScan;
  document.getElementById('ring-score').textContent = hasScan ? Math.round(latestScore) : '';
  const ringFill = document.getElementById('ring-fill');
  const circumference = 2 * Math.PI * 80;
  ringFill.style.strokeDasharray = String(circumference);
  ringFill.style.strokeDashoffset = String(circumference * (1 - (latestScore ?? 0) / 100));
  const cue = lockedEmotion && latestConfidence !== null
    ? lockedEmotion + ' · ' + Math.round(latestConfidence) + '% model confidence'
    : cameraStatus === 'unclear'
      ? 'Unclear · the camera view was not clear enough for an expression reading'
      : 'No camera cue recorded';
  document.getElementById('res-emotion-badge').textContent = cue;
  document.getElementById('res-message').textContent = summaryInsight;

  const breakdown = document.getElementById('breakdown');
  const heading = document.createElement('p');
  heading.className = 'breakdown-title';
  heading.textContent = 'Session details';
  breakdown.replaceChildren(heading);
  const rows = [
    ['Camera risk estimate', latestRisk || (cameraStatus === 'unclear' ? 'Not scored (unclear reading)' : 'Not captured')],
    ['Your messages', String(chatHistory.filter(message => message.role === 'user').length)]
  ];
  if (wearableSummary) {
    const summary = wearableSummary;
    const trends = summary.trends || {};
    const minutes = Math.floor(summary.sample_count / 60);
    rows.push(['Wearable data recorded', (minutes ? minutes + ' min ' : '') + (summary.sample_count % 60) + ' s']);
    rows.push(['Average pulse', Number.isFinite(summary.average_hr)
      ? summary.average_hr.toFixed(1) + ' BPM (range ' + summary.min_hr.toFixed(0) + '–' + summary.max_hr.toFixed(0) +
        ', measured ' + summary.pulse_coverage + '% of the time)'
      : 'Not measured: the hand was not still for 10 s']);
    rows.push(['Average SpO₂ (estimate)', sensorValue(summary.average_spo2, '%', 1, 'Not measured at this sensor position')]);
    if (trends.temperature) {
      rows.push(['Skin temperature', trends.temperature.current.toFixed(2) + ' °C (' + signed(trends.temperature.change, 2, ' °C') + ' vs start)']);
    } else {
      rows.push(['Skin temperature', sensorValue(summary.temperature_c, '°C', 2)]);
    }
    if (trends.gsr && trends.gsr.change_percent !== null) {
      rows.push(['GSR', trends.gsr.current + ' mV (' + signed(trends.gsr.change_percent, 1, '%') + ' vs start; skin conductance ' +
        conductanceLabel(trends.gsr.skin_conductance_vs_start) + ')']);
    } else {
      rows.push(['GSR', sensorValue(summary.gsr_mv, 'mV')]);
    }
    if (trends.movement) rows.push(['Hand still', trends.movement.still_percent + '% of the check-in']);
    rows.push(
      ['Last movement', sensorValue(summary.motion, 'm/s²', 2)],
      ['Last accelerometer X / Y / Z', sensorVector(summary.accel, 'm/s²')],
      ['Last gyroscope X / Y / Z', sensorVector(summary.gyro, 'rad/s')]
    );
  } else {
    rows.push(['USB wearable', 'No readings captured']);
  }
  rows.forEach(([label, value]) => {
    const row = document.createElement('p');
    row.className = 'breakdown-row';
    const name = document.createElement('span');
    name.textContent = label;
    const detail = document.createElement('strong');
    detail.textContent = value;
    row.append(name, detail);
    breakdown.appendChild(row);
  });
}

function saveSessionProgress(summary = null) {
  const checkIn = sessionId;
  const payload = JSON.stringify({
    session_id: sessionId,
    user_id: currentUser?.id ?? null,
    locked_emotion: lockedEmotion,
    confidence: latestConfidence,
    mental_health_score: latestScore,
    risk_level: latestRisk,
    primary_concern: currentUser?.notes ?? '',
    transcript: chatHistory,
    summary_insight: summary
  });
  saveQueue = saveQueue.then(async () => {
    try {
      const response = await fetch('/api/sessions/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
        signal: AbortSignal.timeout(6500)
      });
      if (!response.ok) return false;
      const result = await response.json();
      if (result.wearable_summary && checkIn === sessionId) wearableSummary = result.wearable_summary;
      if (result.safety_review && checkIn === sessionId) safetyReview = result.safety_review;
      return result.success;
    } catch (error) {
      console.error('Failed to save session progress:', error);
      return false;
    }
  });
  return saveQueue;
}

function updateReportSaveStatus(saved) {
  document.getElementById('share-report').disabled = !saved;
  document.getElementById('report-save-status').hidden = saved;
}

async function retryReportSave() {
  const retrySession = sessionId;
  const retry = document.getElementById('retry-report-save');
  retry.disabled = true;
  try {
    const saved = await saveSessionProgress(document.getElementById('res-message').textContent);
    if (retrySession === sessionId) {
      updateReportSaveStatus(saved);
      if (saved) document.getElementById('share-report').focus();
    }
  } finally {
    retry.disabled = false;
  }
}

function openShareModal() {
  const url = new URL('/session/' + encodeURIComponent(sessionId), window.location.origin);
  document.getElementById('share-link-input').value = url.href;
  document.getElementById('share-open-link').href = url.href;
  document.getElementById('copy-share-btn-text').textContent = 'Copy link';
  document.getElementById('share-modal-overlay').showModal();
}

function closeShareModal() {
  document.getElementById('share-modal-overlay').close();
}

async function copyDirectShareLink() {
  const input = document.getElementById('share-link-input');
  const label = document.getElementById('copy-share-btn-text');
  try {
    await navigator.clipboard.writeText(input.value);
    label.textContent = 'Link copied';
  } catch {
    input.select();
    label.textContent = 'Link selected';
  }
}

function formatSessionDate(value) {
  if (!value) return 'Recent check-in';
  const timestamp = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(value)
    ? value.replace(' ', 'T') + 'Z'
    : value;
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat('en-IN', {
    dateStyle: 'medium', timeStyle: 'short', timeZone: 'Asia/Kolkata'
  }).format(date);
}

async function openHistoryDrawer() {
  const drawer = document.getElementById('history-drawer-overlay');
  const body = document.getElementById('history-drawer-body');
  if (!drawer.open) drawer.showModal();
  if (!currentUser?.id) {
    body.innerHTML = '<div class="history-empty">' + iconMarkup('user') + '<p>Choose a saved profile to see your past check-ins.</p></div>';
    return;
  }

  const profileId = currentUser.id;
  body.innerHTML = '<div class="history-empty" role="status">' + iconMarkup('loader', 'spinning') + '<p>Loading your check-ins…</p></div>';
  try {
    const response = await fetch('/api/users/' + encodeURIComponent(profileId) + '/sessions');
    if (!response.ok) throw new Error('Unable to load check-ins');
    const data = await response.json();
    if (currentUser?.id !== profileId) return;
    userSessions = data.sessions || [];
    document.getElementById('nav-session-count').textContent = userSessions.length;

    if (!userSessions.length) {
      body.innerHTML = '<div class="history-empty">' + iconMarkup('calendar-x') + '<p>No check-ins yet. Your saved reflections will appear here.</p></div>';
      return;
    }

    let html = '<p class="history-intro">' + userSessions.length + ' saved check-in' + (userSessions.length === 1 ? '' : 's') + '</p>';
    for (const session of userSessions) {
      const concern = session.primary_concern || session.summary_insight || 'Wellbeing check-in';
      const score = session.mental_health_score == null ? 'No camera score' : 'Camera score: ' + session.mental_health_score + '/100';
      const reportPath = '/session/' + encodeURIComponent(session.session_id);
      html += `
        <article class="history-card">
          <header class="history-card-header">
            <span class="history-date">${escapeHTML(formatSessionDate(session.created_at))}</span>
            <span class="history-pill">${escapeHTML(score)}</span>
          </header>
          <p class="history-topic">${escapeHTML(concern)}</p>
          <div class="history-actions">
            <a href="${reportPath}" target="_blank" rel="noopener" class="button secondary compact">${iconMarkup('file-text')} View report</a>
            <button type="button" class="button quiet compact" data-report-path="${reportPath}">${iconMarkup('share-2')} <span>Copy link</span></button>
          </div>
        </article>`;
    }
    body.innerHTML = html;
    body.querySelectorAll('[data-report-path]').forEach(button => {
      button.addEventListener('click', () => copySessionLink(button));
    });
  } catch (error) {
    if (currentUser?.id !== profileId) return;
    body.innerHTML = '<p class="status-error" role="status">Past sessions could not be loaded.</p><button type="button" class="button secondary compact" onclick="openHistoryDrawer()">Try again</button>';
    console.error('History error:', error);
  }
}

function closeHistoryDrawer() {
  document.getElementById('history-drawer-overlay').close();
}

async function copySessionLink(button) {
  const url = new URL(button.dataset.reportPath, window.location.origin);
  const label = button.querySelector('span');
  try {
    await navigator.clipboard.writeText(url.href);
    label.textContent = 'Link copied';
  } catch {
    const copyField = document.createElement('input');
    copyField.className = 'history-copy-field';
    copyField.value = url.href;
    copyField.readOnly = true;
    copyField.setAttribute('aria-label', 'Report link');
    button.replaceWith(copyField);
    copyField.select();
  }
}

function restart() {
  stopScanning();
  stopCamera();
  stopWearablePolling();
  sessionId = newSessionId();
  finishingCheckIn = false;
  stopChatActivity();
  saveQueue = Promise.resolve();
  lockedEmotion = null;
  scanStartTime = null;
  lastCheckIn = null;
  cameraStatus = 'off';
  isLocked = false;
  latestScore = latestRisk = latestConfidence = null;
  chatHistory = [];
  wearableSummary = null;
  safetyReview = null;
  document.getElementById('finish-session').innerHTML = iconMarkup('check-circle') + '<span>Finish check-in</span>';
  document.getElementById('chat-messages').replaceChildren();
  document.getElementById('conf-fill').style.width = '0%';
  document.getElementById('e-label').textContent = 'Scanning…';
  document.getElementById('lock-row').style.display = 'none';
  document.getElementById('timer-fill').style.width = '0%';
  document.getElementById('start-camera').disabled = false;
  document.getElementById('camera-button-text').textContent = 'Start camera';
  setStatus('Your camera is off.');
  document.getElementById('app-container').classList.remove('focus-mode');
  showScreen('screen-idle');
  startWearablePolling();
}

// ── Heatmap toggle ───────────────────────────────────────────────────
function toggleHeatmap() {
  heatmapEnabled = !heatmapEnabled;
  const btn = document.getElementById('heatmap-toggle');
  const tag = document.getElementById('heatmap-tag');
  btn.classList.toggle('active', heatmapEnabled);
  tag.textContent = heatmapEnabled ? 'On' : 'Off';
  btn.setAttribute('aria-pressed', String(heatmapEnabled));
}

// ── Developer Debug Panel ───────────────────────────────────────────────────
function toggleDebugSection(header) {
  const content = header.nextElementSibling;
  const expanded = header.getAttribute('aria-expanded') === 'true';
  content.style.display = expanded ? 'none' : 'block';
  header.setAttribute('aria-expanded', String(!expanded));
}

function clearTimeline() {
  document.getElementById('dbg-timeline-content').innerHTML = '';
}

function renderWearableDebug(wearable) {
  const label = document.getElementById('debug-wearable-label');
  const details = document.getElementById('dbg-wearable-content');
  const status = document.getElementById('dbg-st-wearable');
  label.style.display = wearable ? 'inline-block' : 'none';
  status.textContent = wearable ? 'ACTIVE' : 'WAITING';
  status.className = wearable ? 'dbg-status dbg-active' : 'dbg-status dbg-waiting';
  if (!wearable) {
    details.textContent = 'Waiting for a wearable reading…';
  } else if (wearable.source === 'USB') {
    label.textContent = 'LIVE USB';
    details.textContent = [
      'Source: USB wearable',
      'Pulse (waveform estimate): ' + sensorValue(wearable.vitals?.pulse?.value, 'BPM', 1, 'not measured') +
        ' (' + (wearable.vitals?.pulse?.state || 'waiting') + ')',
      'SpO2 estimate: ' + sensorValue(wearable.vitals?.spo2?.value, '%', 1, wearable.vitals?.spo2?.state || 'not measured'),
      'Firmware pulse status: ' + wearable.pulse_status + ', firmware pulse: ' + sensorValue(wearable.hr, 'BPM', 1, 'none'),
      'Sensor temperature: ' + sensorValue(wearable.temperature_c, '°C', 2) +
        (wearable.temperature_sensor ? ' (sensor ' + wearable.temperature_sensor + ')' : ''),
      'GSR: ' + sensorValue(wearable.gsr_adc, 'ADC') + ' / ' + sensorValue(wearable.gsr_mv, 'mV'),
      'Optical IR: ' + sensorValue(wearable.ir, 'counts'),
      'Movement: ' + sensorValue(wearable.motion, 'm/s²', 2),
      'Accelerometer: ' + sensorVector(wearable.accel, 'm/s²'),
      'Gyroscope: ' + sensorVector(wearable.gyro, 'rad/s')
    ].join('\n');
  } else {
    label.textContent = 'SIMULATED WEARABLE';
    details.textContent = [
      'Source: SIMULATED',
      'Scenario: ' + wearable.scenario,
      'Pulse: ' + sensorValue(wearable.hr, 'BPM', 1),
      'Temperature: ' + sensorValue(wearable.temp, '°C', 1),
      'GSR: ' + sensorValue(wearable.gsr_adc, 'ADC'),
      'Optical IR: ' + sensorValue(wearable.ir, 'counts'),
      'BVP mean: ' + (wearable.bvp_mean ?? 'Not recorded')
    ].join('\n');
  }
}

function renderDebugState(state) {
  if (!state) return;

  // Pipeline Status
  const sCam = state.camera?.emotion !== "Neutral" || state.camera?.confidence > 0 ? "ACTIVE" : "WAITING";
  const sCalib = state.calibration?.status || "WAITING";
  const sBilstm = state.css?.s_t !== null ? "READY" : "WAITING";
  const sCss = state.css?.s_t !== null ? "UPDATED" : "WAITING";
  const sInterv = state.intervention ? "SELECTED" : "WAITING";
  const sGroq = state.groq_context ? "GENERATED" : "WAITING";

  const updateStatus = (id, status) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = status;
    el.className = 'dbg-status ' + (
      status === 'WAITING' ? 'dbg-waiting' :
      status === 'ACTIVE' || status === 'UPDATED' || status === 'GENERATED' || status === 'READY' || status === 'SELECTED' ? 'dbg-active' :
      status === 'COMPLETE' ? 'dbg-success' :
      status === 'IN_PROGRESS' ? 'dbg-calibrating' : ''
    );
  };
  
  updateStatus('dbg-st-camera', sCam);
  updateStatus('dbg-st-calibration', sCalib);
  updateStatus('dbg-st-bilstm', sBilstm);
  updateStatus('dbg-st-css', sCss);
  updateStatus('dbg-st-intervention', sInterv);
  updateStatus('dbg-st-groq', sGroq);

  // Camera
  if (state.camera) {
    let camHtml = `Locked Emotion: ${state.camera.emotion}\nConfidence: ${state.camera.confidence}%\n\nProbabilities:\n`;
    for (let k in state.camera.probs) {
       camHtml += `${k}: ${(state.camera.probs[k]*100).toFixed(1)}%\n`;
    }
    document.getElementById('dbg-camera-content').textContent = camHtml;
  }

  // MHS
  document.getElementById('dbg-ms-content').textContent = `Mental Health Score (MS_t): ${state.mental_health_score.toFixed(1)}\nStatus: ${state.mental_health_score >= 55 ? 'High Risk' : 'Normal'}`;

  renderWearableDebug(state.wearable);

  // Calibration
  if (state.calibration) {
    let calHtml = `Calibration: ${state.calibration.status}\n`;
    if (state.calibration.status === "IN_PROGRESS") {
      calHtml += `Progress: ${state.calibration.progress} / 300 seconds\n\nS_t: NULL\nReason: Baseline calibration not complete`;
    } else if (state.calibration.status === "COMPLETE") {
      calHtml += `Baseline established for this session.\nSession ID: ${state.session_id.substring(0,8)}...`;
    }
    document.getElementById('dbg-calibration-content').textContent = calHtml;
  }

  // BiLSTM
  if (state.css && state.css.s_t !== null) {
    document.getElementById('dbg-bilstm-content').textContent = `Input shape: [12, 14]\nModel: BiLSTM (Physiological_V2)\n\nStress probability:\nS_t = ${state.css.s_t.toFixed(4)}\n\nPredicted class: ${state.css.s_t > 0.5 ? 'Stress' : 'Baseline'}\n\nSynthetic input. Model output does not represent a physiological measurement.`;
  }

  // CSS
  if (state.css) {
    document.getElementById('dbg-css-content').textContent = `Session ID: ${state.session_id.substring(0,8)}...\nLast updated: ${state.css.last_updated}\n\nS_t: ${state.css.s_t !== null ? state.css.s_t.toFixed(4) : 'NULL'}\nΔS_t: ${state.css.delta_s_t !== null ? (state.css.delta_s_t > 0 ? '+'+state.css.delta_s_t.toFixed(4) : state.css.delta_s_t.toFixed(4)) : 'NULL'}\nH_t: [${(state.css.h_t || []).map(v => v.toFixed(2)).join(', ')}]\n\nMS_t: ${state.css.ms_t.toFixed(1)}\nEmotion: ${state.css.emotion}\nIntent: ${state.css.intent}`;
    
    // NLP Analysis
    let nlpConf = state.css.intent_confidence !== undefined ? state.css.intent_confidence.toFixed(2) : '0.00';
    let sentConf = state.css.sentiment_confidence !== undefined ? state.css.sentiment_confidence.toFixed(2) : '0.00';
    let secIntents = state.css.secondary_intents && state.css.secondary_intents.length ? state.css.secondary_intents.join(', ') : 'None';
    let iRules = state.css.matched_intent_rules && state.css.matched_intent_rules.length ? state.css.matched_intent_rules.join(', ') : 'None';
    let sRules = state.css.matched_sentiment_rules && state.css.matched_sentiment_rules.length ? state.css.matched_sentiment_rules.join(', ') : 'None';
    let fallback = state.css.multimodal_fallback_used ? 'USED (Student Language + Multimodal Support)' : 'NOT USED (Student Language)';
    
    let nlpText = `Primary Intent: ${state.css.intent}\nSecondary Intents: [${secIntents}]\nIntent Confidence: ${nlpConf}\n\nSentiment: ${state.css.sentiment}\nSentiment Confidence: ${sentConf}\n\nMatched Intent Rules: [${iRules}]\nMatched Sentiment Rules: [${sRules}]\n\nMultimodal Fallback: ${fallback}`;
    const dbgNlp = document.getElementById('dbg-nlp-content');
    if (dbgNlp) {
      dbgNlp.textContent = nlpText;
    }
  }

  // Intervention
  if (state.intervention) {
    document.getElementById('dbg-intervention-content').textContent = `Selected intervention:\n${state.intervention.intervention}\n\nPriority: ${state.intervention.priority}\nSafety required: ${state.intervention.safety_required}\nSelector version: ${state.intervention.selector_version}\nReason: ${state.intervention.reason}\n\n(Intervention selected BEFORE LLM generation)`;
  }

  // Guardrail
  if (state.guardrail) {
    let guardrailText = "";
    if (state.guardrail.input) {
        guardrailText += `[INPUT GUARDRAIL]\nAction: ${state.guardrail.input.action.toUpperCase()}\nCategory: ${state.guardrail.input.category}\nConfidence: ${state.guardrail.input.confidence}\nMatched Rules: [${state.guardrail.input.matched_rules.join(", ")}]\n\n`;
    } else {
        guardrailText += `[INPUT GUARDRAIL]\nWaiting...\n\n`;
    }
    
    if (state.guardrail.output) {
        guardrailText += `[OUTPUT GUARDRAIL]\nAction: ${state.guardrail.output.action.toUpperCase()}\nCategory: ${state.guardrail.output.category}\n`;
    } else {
        guardrailText += `[OUTPUT GUARDRAIL]\nWaiting...\n`;
    }
    
    document.getElementById('dbg-guardrail-content').textContent = guardrailText;
  }

  // Groq
  if (state.groq_context) {
    document.getElementById('dbg-groq-content').textContent = `Model: ${state.groq_context.model}\nUser message: "${state.groq_context.user_message}"\nSafety instructions: ${state.groq_context.safety_instructions}\n\nSystem Prompt (Safe):\n${state.groq_context.system_prompt_safe}`;
  }

  // Timeline
  if (state.timeline && state.timeline.length > 0) {
    const tlContainer = document.getElementById('dbg-timeline-content');
    tlContainer.innerHTML = '';
    [...state.timeline].reverse().forEach(ev => {
      const d = document.createElement('div');
      d.style.marginBottom = '8px';
      d.innerHTML = `<span style="color:#71717a">${escapeHTML(ev.time)}</span> <br><strong style="color:var(--accent)">${escapeHTML(ev.category)}</strong><br>${escapeHTML(ev.message)}`;
      tlContainer.appendChild(d);
    });
  }
  
  // Plot Graph
  if (state.css && state.css.h_t) {
    drawStressGraph(state.css.h_t);
  }
}

function drawStressGraph(historyArray) {
  const canvas = document.getElementById('dbg-stress-graph');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  
  ctx.clearRect(0, 0, w, h);
  
  // Draw grid/guides
  ctx.strokeStyle = 'rgba(255,255,255,0.1)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h/2); ctx.lineTo(w, h/2);
  ctx.stroke();
  
  if (!historyArray || historyArray.length === 0) {
    ctx.fillStyle = '#52635a';
    ctx.font = '12px monospace';
    ctx.fillText('No data yet', 10, 20);
    return;
  }
  
  const maxPts = 30; // Show last 30 points
  const pts = historyArray.slice(-maxPts);
  const step = w / Math.max(pts.length - 1, 1);
  
  ctx.beginPath();
  ctx.strokeStyle = '#34745f';
  ctx.lineWidth = 2;
  
  pts.forEach((val, i) => {
    const x = i * step;
    const y = h - (val * h); // val is 0.0-1.0
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  
  // Draw points
  ctx.fillStyle = '#173d34';
  pts.forEach((val, i) => {
    const x = i * step;
    const y = h - (val * h);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2*Math.PI);
    ctx.fill();
  });
}
