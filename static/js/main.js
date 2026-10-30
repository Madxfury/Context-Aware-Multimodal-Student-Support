// Escape stored profile/session text before inserting it into HTML templates.
function escapeHTML(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[char]));
}

// ═══════════════════════════════════════════════════════════
//  Samvya — main.js
// ═══════════════════════════════════════════════════════════

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
let latestScore = 0;
let latestRisk = "Normal";
let latestConfidence = 0;

let lockedEmotion = null;
let isLocked = false;
let scanStartTime = null;
let emotionCounts = {};
let isScanning = false;
let inflight = false;
let heatmapEnabled = false;
let video, captureCanvas, captureCtx, overlayCanvas, overlayCtx;

// Audio recording state
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// Chat history state
let chatHistory = [];
let currentVitals = {};

// ── Initialise ────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', initApp);

async function initApp() {
  video = document.getElementById('video');
  captureCanvas = document.createElement('canvas');
  captureCtx = captureCanvas.getContext('2d');
  overlayCanvas = document.getElementById('overlay-canvas');
  overlayCtx = overlayCanvas.getContext('2d');

  initChatUI();
  
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
  setStatus('Awaiting user profile confirmation…');
}

// ── User Management & Profile Gating ───────────────────────────
async function loadExistingUsers() {
  try {
    const res = await fetch('/api/users');
    const data = await res.json();
    const users = data.users || [];
    const container = document.getElementById('user-chips-list');
    const box = document.getElementById('returning-users-box');
    
    if (users.length > 0 && container && box) {
      box.style.display = 'block';
      container.innerHTML = '';
      users.forEach(u => {
        const chip = document.createElement('div');
        chip.className = 'user-chip';
        chip.innerHTML = `<i data-lucide="user" style="width:12px; height:12px;"></i> <span>${escapeHTML(u.name)}</span> <span style="opacity:0.6; font-size:0.75rem;">(${escapeHTML(u.role || 'Student')})</span>`;
        chip.onclick = () => selectUserChip(u);
        container.appendChild(chip);
      });
      lucide.createIcons();
    }
  } catch (err) {
    console.error("Failed to load existing users:", err);
  }
}

function selectUserChip(user) {
  currentUser = user;
  document.getElementById('input-user-name').value = user.name;
  if (user.role) document.getElementById('input-user-role').value = user.role;
  if (user.age) document.getElementById('input-user-age').value = user.age;
  if (user.notes) document.getElementById('input-user-concern').value = user.notes;
  
  activateUserProfile(user);
}

async function handleProfileSubmit(event) {
  if (event) event.preventDefault();
  const name = document.getElementById('input-user-name').value.trim();
  if (!name) return;
  
  const role = document.getElementById('input-user-role').value.trim() || 'Student';
  const age = document.getElementById('input-user-age').value.trim();
  const notes = document.getElementById('input-user-concern').value.trim();

  try {
    const res = await fetch('/api/users', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, role, age, notes })
    });
    const data = await res.json();
    if (data.user) {
      currentUser = data.user;
      userSessions = data.recent_sessions || [];
      activateUserProfile(currentUser);
    }
  } catch (err) {
    console.error("Profile submit error:", err);
    currentUser = { id: null, name, role, age, notes };
    activateUserProfile(currentUser);
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
  
  // Close profile modal
  const modal = document.getElementById('profile-modal');
  if (modal) modal.style.display = 'none';

  // Now start camera and scanning!
  startCameraAndScan();
}

function openProfileModal() {
  const modal = document.getElementById('profile-modal');
  if (modal) modal.style.display = 'flex';
  loadExistingUsers();
  lucide.createIcons();
}

async function startCameraAndScan() {
  if (video && video.srcObject) {
    setStatus(`Looking for ${currentUser ? currentUser.name : ''}'s face…`);
    startScanning();
    return;
  }
  
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      captureCanvas.width = video.videoWidth;
      captureCanvas.height = video.videoHeight;
      overlayCanvas.width = video.videoWidth;
      overlayCanvas.height = video.videoHeight;
      setStatus(`Looking for ${currentUser ? currentUser.name : ''}'s face…`);
      startScanning();
    };
  } catch (err) {
    setStatus('Camera access denied. Please allow camera access and refresh.');
    console.error(err);
  }
}

// ── Scanning loop ─────────────────────────────────────────────
// Mirrors final.py's continuous while-loop: fires the next request
// immediately after the previous one returns (no fixed interval).
function startScanning() {
  isScanning = true;
  scanLoop();
}

function stopScanning() {
  isScanning = false;
}

async function scanLoop() {
  if (!isScanning) return;
  await captureAndDetect();
  // Chain next iteration immediately — matches final.py's continuous loop
  if (isScanning) setTimeout(scanLoop, 0);
}

async function captureAndDetect() {
  if (!video.videoWidth) return;
  captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  const b64 = captureCanvas.toDataURL('image/jpeg', 0.75);

  try {
    const res = await fetch('/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: b64, heatmap: heatmapEnabled, session_id: sessionId })
    });
    const data = await res.json();
    handleDetection(data);
  } catch (e) {
    console.error('Detect error:', e);
  }
}

async function fetchDebugState() {
  if (!document.getElementById('app-container').classList.contains('debug-active')) return;
  try {
    const res = await fetch(`/api/debug/state?session_id=${sessionId}`);
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
    setEmotionBadge('—', 0);
    document.getElementById('lock-row').style.display = 'none';
    scanStartTime = null;
    return;
  }

  const { emotion, confidence, face_box } = data;
  if (data.final_ms !== undefined) latestScore = data.final_ms;
  if (data.risk) latestRisk = data.risk;
  if (data.confidence) latestConfidence = data.confidence;

  // Draw face box on overlay
  if (face_box) {
    const [x, y, w, h] = face_box;
    const scaleX = overlayCanvas.width / captureCanvas.width;
    const scaleY = overlayCanvas.height / captureCanvas.height;

    // If heatmap returned, draw blended face+heatmap image first
    if (data.heatmap_b64) {
      const img = new Image();
      img.onload = () => {
        overlayCtx.globalAlpha = 1;
        overlayCtx.drawImage(img, x * scaleX, y * scaleY, w * scaleX, h * scaleY);
        // Redraw face rect on top
        overlayCtx.strokeStyle = '#dddddd';
        overlayCtx.lineWidth = 2;
        overlayCtx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);

        // Draw stats text above the bounding box
        overlayCtx.save();
        overlayCtx.scale(-1, 1);
        overlayCtx.font = 'bold 16px monospace';
        overlayCtx.fillStyle = '#ffffff';
        overlayCtx.strokeStyle = '#000000';
        overlayCtx.lineWidth = 3;
        
        const labelText = `${data.emotion} ${data.confidence.toFixed(1)}% | MS:${data.final_ms.toFixed(1)} | ${data.risk}`;
        const textX = -((x + w) * scaleX);
        const textY = (y * scaleY) - 10;
        
        overlayCtx.strokeText(labelText, textX, textY);
        overlayCtx.fillText(labelText, textX, textY);
        overlayCtx.restore();
      };
      img.src = 'data:image/jpeg;base64,' + data.heatmap_b64;
    } else {
      overlayCtx.strokeStyle = getComputedStyle(document.body).getPropertyValue('--accent').trim() || '#dddddd';
      overlayCtx.lineWidth = 2;
      overlayCtx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);
    }
  }

  setEmotionBadge(emotion, confidence);

  // Lock logic: 10-second scanning window
  if (!isLocked) {
    if (!scanStartTime) {
      scanStartTime = Date.now();
      emotionCounts = {};
      document.getElementById('lock-row').style.display = 'flex';
    }

    if (confidence >= 50) {
      emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1;
    }

    const elapsed = Date.now() - scanStartTime;
    const remaining = Math.max(0, 10000 - elapsed);
    const progress = Math.min(100, (elapsed / 10000) * 100);

    document.getElementById('timer-fill').style.width = `${progress}%`;
    document.getElementById('lock-text').textContent = `Analyzing... ${Math.ceil(remaining/1000)}s`;

    if (elapsed >= 10000) {
      isLocked = true;
      let prominentEmotion = 'Neutral';
      let maxCount = -1;
      for (const [emo, count] of Object.entries(emotionCounts)) {
        if (count > maxCount) {
          maxCount = count;
          prominentEmotion = emo;
        }
      }
      lockEmotion(prominentEmotion);
      return;
    }
  }

  setStatus(`Confidence: ${confidence.toFixed(1)}% — analyzing…`);
}

function setEmotionBadge(emotion, confidence) {
  const cfg = EMOTION_CFG[emotion];
  document.getElementById('e-label').textContent = cfg
    ? `${cfg.label} · ${confidence.toFixed(0)}%`
    : emotion || 'Scanning…';
  document.getElementById('conf-fill').style.width = `${Math.min(confidence, 100)}%`;
}

function setStatus(msg) {
  document.getElementById('status-text').textContent = msg;
}

// ── Lock emotion & transition ─────────────────────────────────
function lockEmotion(emotion) {
  lockedEmotion = emotion;
  isLocked = true;
  stopScanning();
  stopCamera();
  document.body.setAttribute('data-emotion', emotion);

  setStatus(`Prominent emotion locked: ${emotion}`);
  saveSessionProgress();
  setTimeout(() => showScreen('screen-chat', emotion), 600);
}

function stopCamera() {
  if (video && video.srcObject) {
    const tracks = video.srcObject.getTracks();
    tracks.forEach(track => track.stop());
    video.srcObject = null;
  }
}

// ── SCREEN 2: CHAT ───────────────────────────────────────
function initChatUI() {
  const btnSend = document.getElementById('btn-send');
  const chatInput = document.getElementById('chat-input');
  const btnVoice = document.getElementById('btn-voice');

  btnSend.onclick = () => {
    const text = chatInput.value.trim();
    if (text) {
      appendMessage('user', text);
      chatInput.value = '';
      sendChatMessage(text);
    }
  };

  chatInput.onkeypress = (e) => {
    if (e.key === 'Enter') btnSend.click();
  };

  // Setup Voice Recording
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };
      mediaRecorder.onstop = () => {
        if (audioChunks.length === 0) {
          console.warn("No audio chunks to send.");
          return;
        }
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        audioChunks = [];
        sendVoiceMessage(audioBlob);
      };
    }).catch(err => console.error("Mic access denied:", err));
  }

  btnVoice.onclick = toggleRecording;
}

function toggleRecording(e) {
  if (e) e.preventDefault();
  if (!mediaRecorder) {
    appendMessage('bot', "Microphone access is not available. Please check your browser permissions.");
    return;
  }
  
  const btn = document.getElementById('btn-voice');
  
  if (mediaRecorder.state === 'inactive') {
    // Start recording
    isRecording = true;
    audioChunks = [];
    recordingStartTime = Date.now();
    mediaRecorder.start(100); // Send chunks every 100ms
    
    btn.classList.add('recording');
    btn.style.backgroundColor = "rgba(239, 68, 68, 0.2)"; // Red tint
    btn.style.borderColor = "#ef4444";
    btn.style.color = "#ef4444";
    btn.title = "Tap to stop recording";
    
  } else if (mediaRecorder.state === 'recording') {
    // Stop recording
    const duration = Date.now() - recordingStartTime;
    isRecording = false;
    mediaRecorder.stop();
    
    btn.classList.remove('recording');
    btn.style.backgroundColor = "rgba(255,255,255,0.05)";
    btn.style.borderColor = "var(--accent)";
    btn.style.color = "var(--text)";
    btn.title = "Tap to record voice";
    
    if (duration < 500) {
      // Too short
      audioChunks = [];
      appendMessage('bot', "Recording was too short to process.");
    }
  }
}

function appendMessage(sender, text) {
  const container = document.getElementById('chat-messages');
  const msgDiv = document.createElement('div');
  msgDiv.className = `chat-bubble ${sender}`;
  msgDiv.textContent = text;
  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
  return msgDiv;
}

function typeMessage(sender, text, speed = 15) {
  const container = document.getElementById('chat-messages');
  const msgDiv = document.createElement('div');
  msgDiv.className = `chat-bubble ${sender}`;
  container.appendChild(msgDiv);
  
  let i = 0;
  function type() {
    if (i < text.length) {
      msgDiv.textContent += text.charAt(i);
      i++;
      container.scrollTop = container.scrollHeight;
      setTimeout(type, speed);
    }
  }
  type();
}

async function sendChatMessage(text) {
  chatHistory.push({ role: 'user', content: text });
  
  const typingIndicator = appendMessage('bot', 'AI is typing...');
  typingIndicator.style.opacity = '0.5';
  typingIndicator.style.fontStyle = 'italic';
  
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        message: text, 
        emotion: lockedEmotion, 
        history: chatHistory, 
        sensors: currentVitals, 
        session_id: sessionId,
        user_id: currentUser ? currentUser.id : null 
      })
    });
    const data = await res.json();
    
    typingIndicator.remove();
    
    if (data.debug_state) renderDebugState(data.debug_state);
    
    if (data.response) {
      chatHistory.push({ role: 'assistant', content: data.response });
      typeMessage('bot', data.response);
      saveSessionProgress();
      
      if (data.audio_base64) {
        try {
          const audio = new Audio("data:audio/mp3;base64," + data.audio_base64);
          audio.play();
        } catch (e) {
          console.error("Audio playback error:", e);
        }
      }
    }
  } catch (err) {
    console.error("Chat error:", err);
    typingIndicator.remove();
    appendMessage('bot', "Sorry, I'm having trouble connecting to the server.");
  }
}

async function sendVoiceMessage(blob) {
  appendMessage('user', '🎤 Voice note sent...');
  const formData = new FormData();
  formData.append('audio', blob, 'voice.webm');
  
  try {
    const res = await fetch('/transcribe', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (data.transcription) {
      // Replace the loading message or just append the transcription
      appendMessage('user', `(Transcription) ${data.transcription}`);
      sendChatMessage(data.transcription);
    }
  } catch (err) {
    console.error("Voice error:", err);
    appendMessage('bot', "Sorry, I couldn't process your voice note.");
  }
}

let simulatorInterval;
let sessionId = "session_" + Math.random().toString(36).substring(7);

async function fetchSimulatedVitals(emotion) {
  try {
    const res = await fetch(`/api/simulator/stream?session_id=${sessionId}&emotion=${emotion}`);
    const data = await res.json();
    currentVitals = {
      hr: data.hr,
      temp: data.temp,
      v2_sequence: data.v2_sequence,
      gsr: data.gsr_adc
    };
    
    if (data.debug_state) renderDebugState(data.debug_state);
  } catch (err) {
    console.error("Simulator stream error:", err);
  }
}

function showChat(emotion) {
  document.getElementById('chat-messages').innerHTML = ''; // Clear chat
  chatHistory = [];
  
  document.getElementById('app-container').classList.add('focus-mode');
  
  // Start simulator stream
  currentVitals = {};
  fetchSimulatedVitals(emotion);
  if (simulatorInterval) clearInterval(simulatorInterval);
  simulatorInterval = setInterval(() => fetchSimulatedVitals(emotion), 3000);
  
  const userName = currentUser ? currentUser.name : 'there';
  const isReturning = userSessions && userSessions.length > 0;
  const initMsg = isReturning
    ? `Welcome back, ${userName}. How are you feeling today?`
    : `Hello ${userName}, I'm Samvya, your wellness counselor. How are you feeling today?`;

  chatHistory.push({ role: 'assistant', content: initMsg });
  typeMessage('bot', initMsg);
  
  setTimeout(() => { lucide.createIcons(); }, 10);
}

// ── Helpers ───────────────────────────────────────────────────
function showScreen(id, emotion = null) {
  document.querySelectorAll('.right-panel .screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  if (id === 'screen-chat') showChat(emotion);
}

// ── Finalize & Report Workflow ─────────────────────────────────
async function finishSessionAndShowReport() {
  if (simulatorInterval) clearInterval(simulatorInterval);
  
  const finishBtn = document.querySelector('.chat-header button');
  if (finishBtn) finishBtn.innerHTML = `<i data-lucide="loader" class="icon-small" style="animation:spin 1s linear infinite;"></i> Preparing...`;
  
  let summaryInsight = "Client completed a mental wellness check-in.";
  try {
    const summaryRes = await fetch('/api/sessions/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        transcript: chatHistory,
        emotion: lockedEmotion || 'Neutral',
        mental_health_score: latestScore,
        user_name: currentUser ? currentUser.name : 'Student'
      })
    });
    const summaryData = await summaryRes.json();
    if (summaryData.summary) summaryInsight = summaryData.summary;
  } catch (e) {
    console.error("Summary error:", e);
  }

  // Save finalized session to DB
  await saveSessionProgress(summaryInsight);
  
  // Update local session cache count
  if (currentUser && currentUser.id) {
    try {
      const sessRes = await fetch(`/api/users/${currentUser.id}/sessions`);
      const sessData = await sessRes.json();
      userSessions = sessData.sessions || [];
      const countEl = document.getElementById('nav-session-count');
      if (countEl) countEl.textContent = userSessions.length;
    } catch (e) {}
  }

  renderReportScreen(summaryInsight);
  showScreen('screen-results');
  lucide.createIcons();
}

function renderReportScreen(summaryInsight) {
  // Score Ring
  const scoreEl = document.getElementById('ring-score');
  if (scoreEl) scoreEl.textContent = Math.round(latestScore);
  
  const ringFill = document.getElementById('ring-fill');
  if (ringFill) {
    const circumference = 2 * Math.PI * 80;
    const offset = circumference - (latestScore / 100) * circumference;
    ringFill.style.strokeDasharray = `${circumference}`;
    ringFill.style.strokeDashoffset = `${offset}`;
  }

  // Emotion badge
  const emoBadge = document.getElementById('res-emotion-badge');
  if (emoBadge) {
    const cfg = EMOTION_CFG[lockedEmotion] || { label: lockedEmotion || 'Neutral' };
    emoBadge.textContent = `${cfg.label} (${Math.round(latestConfidence)}% confidence)`;
  }

  // Insight message
  const msgEl = document.getElementById('res-message');
  if (msgEl) {
    msgEl.textContent = summaryInsight;
  }

  // Breakdown bars
  const breakdown = document.getElementById('breakdown');
  if (breakdown) {
    breakdown.innerHTML = '<div class="breakdown-title">Session Summary & Vitals</div>';
    const items = [
      { label: 'Wellbeing Risk Level', val: latestRisk },
      { label: 'Facial Emotion Affect', val: lockedEmotion || 'Neutral' },
      { label: 'Observed Heart Rate', val: currentVitals.hr ? `${Math.round(currentVitals.hr)} BPM` : 'Standard' },
      { label: 'Dialogue Turns Completed', val: `${chatHistory.filter(m => m.role === 'user').length} turns` }
    ];
    items.forEach(item => {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.05); font-size:0.9rem;';
      row.innerHTML = `<span style="color:var(--text-muted);">${item.label}</span><span style="font-weight:600; color:var(--text);">${item.val}</span>`;
      breakdown.appendChild(row);
    });
  }
}

async function saveSessionProgress(summary = null) {
  try {
    const payload = {
      session_id: sessionId,
      user_id: currentUser ? currentUser.id : null,
      locked_emotion: lockedEmotion,
      confidence: latestConfidence,
      mental_health_score: latestScore,
      risk_level: latestRisk,
      primary_concern: currentUser ? currentUser.notes : "",
      average_hr: currentVitals.hr || null,
      transcript: chatHistory,
      summary_insight: summary
    };
    await fetch('/api/sessions/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  } catch (err) {
    console.error("Failed to save session progress:", err);
  }
}

function openShareModal() {
  const shareInput = document.getElementById('share-link-input');
  const shareOpenLink = document.getElementById('share-open-link');
  const shareModal = document.getElementById('share-modal-overlay');
  
  const shareUrl = window.location.origin + '/session/' + sessionId;
  if (shareInput) shareInput.value = shareUrl;
  if (shareOpenLink) shareOpenLink.href = '/session/' + sessionId;
  if (shareModal) shareModal.style.display = 'flex';
  lucide.createIcons();
}

function closeShareModal() {
  const shareModal = document.getElementById('share-modal-overlay');
  if (shareModal) shareModal.style.display = 'none';
}

function copyDirectShareLink() {
  const shareInput = document.getElementById('share-link-input');
  if (!shareInput) return;
  navigator.clipboard.writeText(shareInput.value).then(() => {
    const btnText = document.getElementById('copy-share-btn-text');
    if (btnText) {
      btnText.textContent = 'Copied!';
      setTimeout(() => { btnText.textContent = 'Copy'; }, 2000);
    }
  }).catch(err => {
    alert("Share link: " + shareInput.value);
  });
}

async function openHistoryDrawer() {
  const drawer = document.getElementById('history-drawer-overlay');
  if (!drawer) return;
  drawer.style.display = 'flex';
  
  const body = document.getElementById('history-drawer-body');
  if (!body) return;
  
  if (!currentUser) {
    body.innerHTML = `
      <div style="text-align:center; color:var(--text-muted); padding:40px 10px;">
        <i data-lucide="user-x" style="width:36px; height:36px; margin-bottom:10px; opacity:0.5;"></i>
        <p>No active user profile selected.</p>
        <button class="badge-nav-btn" style="margin-top:15px;" onclick="closeHistoryDrawer(); openProfileModal();">Select Profile</button>
      </div>`;
    lucide.createIcons();
    return;
  }

  body.innerHTML = `
    <div style="text-align:center; color:var(--text-muted); padding:30px 10px;">
      <i data-lucide="loader" style="width:32px; height:32px; margin-bottom:10px; animation: spin 1s linear infinite;"></i>
      <p>Loading sessions for ${escapeHTML(currentUser.name)}...</p>
    </div>`;
  lucide.createIcons();

  try {
    const res = await fetch(`/api/users/${currentUser.id}/sessions`);
    const data = await res.json();
    userSessions = data.sessions || [];
    
    const countEl = document.getElementById('nav-session-count');
    if (countEl) countEl.textContent = userSessions.length;

    if (userSessions.length === 0) {
      body.innerHTML = `
        <div style="text-align:center; color:var(--text-muted); padding:40px 10px;">
          <i data-lucide="calendar-x" style="width:36px; height:36px; margin-bottom:10px; opacity:0.5;"></i>
          <p>No previous sessions recorded for ${escapeHTML(currentUser.name)} yet.</p>
          <p style="font-size:0.8rem; margin-top:5px;">Complete a check-in to save your first wellness report!</p>
        </div>`;
      lucide.createIcons();
      return;
    }

    let html = `<div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:5px;">${userSessions.length} total check-in session${userSessions.length > 1 ? 's' : ''}</div>`;
    
    userSessions.forEach(sess => {
      const emo = sess.locked_emotion || 'Neutral';
      const score = sess.mental_health_score !== null ? sess.mental_health_score : '--';
      const risk = sess.risk_level || 'Normal';
      const dateStr = sess.created_at ? new Date(sess.created_at).toLocaleString() : 'Recent';
      const concern = sess.primary_concern || sess.summary_insight || 'General mental health check-in';
      
      const badgeClass = (risk === 'high' || risk.toLowerCase().includes('high')) ? 'badge-high' : 
                         (risk.toLowerCase().includes('moderate')) ? 'badge-moderate' : 'badge-normal';

      html += `
        <div class="history-card">
          <div class="history-card-header">
            <span class="history-date">${escapeHTML(dateStr)}</span>
            <span class="history-pill ${badgeClass}">MS: ${escapeHTML(score)}/100</span>
          </div>
          <div class="history-stats-row">
            <span style="font-weight:600; color:var(--text); font-size:0.9rem;">Emotion: ${escapeHTML(emo)}</span>
            ${sess.average_hr ? `<span style="font-size:0.8rem; color:var(--text-muted);">&bull; HR: ${Math.round(sess.average_hr)} BPM</span>` : ''}
          </div>
          <div class="history-topic">"${escapeHTML(concern)}"</div>
          <div class="history-actions">
            <a href="/session/${encodeURIComponent(sess.session_id)}" target="_blank" class="btn-history-action">
              <i data-lucide="file-text" style="width:13px; height:13px;"></i> View Full Report
            </a>
            <button class="btn-history-action" onclick="copySessionLink(decodeURIComponent('${escapeHTML(encodeURIComponent(sess.session_id)).replace(/&#39;/g, '%27')}'))">
              <i data-lucide="share-2" style="width:13px; height:13px;"></i> Copy Link
            </button>
          </div>
        </div>
      `;
    });
    
    body.innerHTML = html;
    lucide.createIcons();
  } catch (err) {
    console.error("Error loading sessions:", err);
    body.innerHTML = `<p style="color:#ef4444;">Failed to load session history.</p>`;
  }
}

function closeHistoryDrawer() {
  const drawer = document.getElementById('history-drawer-overlay');
  if (drawer) drawer.style.display = 'none';
}

function copySessionLink(sid) {
  const link = window.location.origin + '/session/' + sid;
  navigator.clipboard.writeText(link).then(() => {
    alert("Session link copied to clipboard!");
  }).catch(() => {
    prompt("Copy session link:", link);
  });
}

function restart() {
  lockedEmotion = null; scanStartTime = null; emotionCounts = {}; isLocked = false;
  sessionId = "session_" + Math.random().toString(36).substring(7);
  document.body.removeAttribute('data-emotion');
  document.getElementById('chat-messages').innerHTML = '';
  document.getElementById('conf-fill').style.width = '0%';
  document.getElementById('e-label').textContent = 'Scanning…';
  document.getElementById('lock-row').style.display = 'none';
  document.getElementById('timer-fill').style.width = '0%';
  
  document.getElementById('app-container').classList.remove('focus-mode');
  
  if (simulatorInterval) clearInterval(simulatorInterval);
  
  showScreen('screen-idle');
  startCameraAndScan();
}

// ── Heatmap toggle ───────────────────────────────────────────────────
function toggleHeatmap() {
  heatmapEnabled = !heatmapEnabled;
  const btn = document.getElementById('heatmap-toggle');
  const tag = document.getElementById('heatmap-tag');
  btn.classList.toggle('active', heatmapEnabled);
  tag.textContent = heatmapEnabled ? 'ON' : 'OFF';
}

// ── Developer Debug Panel ───────────────────────────────────────────────────
function toggleDebugSection(header) {
  const content = header.nextElementSibling;
  const section = header.parentElement;
  if (content.style.display === 'none') {
    content.style.display = 'block';
    section.classList.add('open');
  } else {
    content.style.display = 'none';
    section.classList.remove('open');
  }
}

function clearTimeline() {
  document.getElementById('dbg-timeline-content').innerHTML = '';
}

function renderDebugState(state) {
  if (!state) return;

  // Pipeline Status
  const sCam = state.camera?.emotion !== "Neutral" || state.camera?.confidence > 0 ? "ACTIVE" : "WAITING";
  const sWear = state.wearable ? "ACTIVE" : "WAITING";
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
  updateStatus('dbg-st-wearable', sWear);
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
  document.getElementById('dbg-ms-content').textContent = `Mental Health Score (MS_t): ${state.mental_health_score.toFixed(1)}\nStatus: ${state.mental_health_score >= 45 ? 'High Risk' : 'Normal'}`;

  // Wearable
  if (state.wearable) {
    document.getElementById('debug-wearable-label').style.display = 'inline-block';
    let wHtml = `Source: SIMULATED\nScenario: ${state.wearable.scenario}\n\n[Hardware Telemetry]\nHR: ${state.wearable.hr}\nTemp: ${state.wearable.temp}\nGSR_ADC: ${state.wearable.gsr_adc}\nIR: ${state.wearable.ir}\nRED: ${state.wearable.red}\n\n[WESAD-Compatible Layer]\nBVP_Mean: ${state.wearable.bvp_mean}\n... (Sent to BiLSTM)`;
    document.getElementById('dbg-wearable-content').textContent = wHtml;
  }

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
    document.getElementById('dbg-bilstm-content').textContent = `Input shape: [12, 14]\nModel: BiLSTM (Physiological_V2)\n\nStress probability:\nS_t = ${state.css.s_t.toFixed(4)}\n\nPredicted class: ${state.css.s_t > 0.5 ? 'Stress' : 'Baseline'}\n\n⚠️ Synthetic input — model output has no physiological/clinical meaning.`;
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
    ctx.fillStyle = '#a1a1aa';
    ctx.font = '12px monospace';
    ctx.fillText('No data yet', 10, 20);
    return;
  }
  
  const maxPts = 30; // Show last 30 points
  const pts = historyArray.slice(-maxPts);
  const step = w / Math.max(pts.length - 1, 1);
  
  ctx.beginPath();
  ctx.strokeStyle = '#f43f5e'; // Red-ish for stress
  ctx.lineWidth = 2;
  
  pts.forEach((val, i) => {
    const x = i * step;
    const y = h - (val * h); // val is 0.0-1.0
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  
  // Draw points
  ctx.fillStyle = '#fff';
  pts.forEach((val, i) => {
    const x = i * step;
    const y = h - (val * h);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2*Math.PI);
    ctx.fill();
  });
}
