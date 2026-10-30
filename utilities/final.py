# improved_realtime_emotion.py
# ===================================
# REAL-TIME EMOTION (Swin) + GRAD-CAM HEATMAP + IMPROVED MS + MOTION + BLINK + DRIVE UPLOAD
# Includes robust handling for missing MPS ops (aten::roll) by using PYTORCH_ENABLE_MPS_FALLBACK
# and explicit CPU retry for failing model calls.
# ===================================

import os
import platform

# IMPORTANT: Set this BEFORE importing torch so PyTorch will allow CPU fallback for missing MPS ops.
if platform.system() == "Darwin":
    # ensure fallback to CPU for MPS-unimplemented ops (as suggested by PyTorch)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import math
import time
import torch
import timm
import numpy as np
import pandas as pd
import torch.nn as nn
from collections import deque
from PIL import Image
from datetime import datetime
from torchvision import transforms
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# ---------------- CONFIG -------------------
MODEL_PATH = "/Users/sanskarparab/CC Emotion Detection /Facial-Expression-Recognition-FER-for-Mental-Health-Detection-/Models/Swin_FANE_Best_Model.pth"
OUTPUT_DIR = "./utilities/realtime_results"
SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, "snapshots")
CSV_LOG = os.path.join(OUTPUT_DIR, "emotion_log.csv")

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTION_WEIGHTS = {'Angry': 58, 'Disgust': 53, 'Fear': 53, 'Surprise': 46, 'Happy': 43, 'Sad': 53, 'Neutral': 30}

ROLL_EMOTION = 20
ROLL_BLINK = 20
ROLL_MOTION = 10

SAVE_CONF_THRESHOLD = 60.0  # percent
MOTION_RESTLESS_THRESHOLD = 1.0
BLINK_CLOSED_FRAMES_THRESHOLD = 3

# ---------------- DEVICE -------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available()
                      else 'mps' if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                      else 'cpu')
print(f"Using device: {DEVICE}")

# ---------------- MODEL DEFINITION -------------
class CustomSwinTransformer(nn.Module):
    def __init__(self, pretrained=False, num_classes=7):
        super(CustomSwinTransformer, self).__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

# ---------------- LOAD MODEL -------------------
model = CustomSwinTransformer(pretrained=False, num_classes=len(EMOTIONS))

# load saved state (support wrapped dicts)
state = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(state, dict) and 'model_state_dict' in state:
    state = state['model_state_dict']
try:
    model.load_state_dict(state, strict=False)
except Exception as e:
    print("Warning: strict load failed, trying loose load. Error:", e)
    try:
        model.load_state_dict(state, strict=False)
    except Exception as e2:
        print("Second attempt failed:", e2)

model.to(DEVICE)
model.eval()
print("Model loaded.")

# ---------------- TRANSFORMS -------------------
transform = transforms.Compose([
    lambda img: img.convert("RGB"),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- FACE + EYE DETECTORS -----------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# ---------------- Grad-CAM Implementation ----------------
class SwinGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            # detach so we don't keep computation graph
            try:
                self.activations = out.detach()
            except Exception:
                self.activations = out
        def backward_hook(module, grad_in, grad_out):
            try:
                self.gradients = grad_out[0].detach()
            except Exception:
                self.gradients = grad_out[0]
        # register hooks
        self.target_layer.register_forward_hook(forward_hook)
        try:
            self.target_layer.register_full_backward_hook(backward_hook)
        except Exception:
            self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):
        """
        Generate CAM. This function will attempt to run on the model's current device.
        If a NotImplementedError occurs (e.g., MPS missing op), it will retry with the model moved to CPU.
        """
        # clear previous stored tensors
        self.activations = None
        self.gradients = None

        # normal path: run model and compute gradients (require grad)
        try:
            # ensure gradients enabled for backward
            out = self.model(input_tensor)  # may raise NotImplementedError on MPS
        except NotImplementedError as e:
            # fallback: run on CPU
            print("Grad-CAM model forward raised NotImplementedError (likely MPS missing op). Retrying on CPU.")
            # move model to CPU and run
            orig_device = next(self.model.parameters()).device if any(p.requires_grad for p in self.model.parameters()) else DEVICE
            self.model.to('cpu')
            input_cpu = input_tensor.to('cpu')
            out = self.model(input_cpu)
            # keep model on CPU for backward (grad capture) — then will move back below
            ran_on_cpu = True
        else:
            ran_on_cpu = False

        if class_idx is None:
            class_idx = torch.argmax(out, dim=1).item()
        target = out[0, class_idx]

        # backward to collect gradients
        try:
            target.backward(retain_graph=True)
        except RuntimeError as e:
            # If backward fails on the current device (rare), try CPU fallback
            if 'not currently implemented for the mps device' in str(e).lower() or isinstance(e, NotImplementedError):
                print("Grad-CAM backward failed on device; retrying forward+backward on CPU.")
                self.model.to('cpu')
                input_cpu = input_tensor.to('cpu')
                out = self.model(input_cpu)
                class_idx = torch.argmax(out, dim=1).item()
                target = out[0, class_idx]
                target.backward(retain_graph=True)
                ran_on_cpu = True
            else:
                raise

        acts = self.activations
        grads = self.gradients

        # if we ran on cpu, ensure acts/grads are CPU tensors
        if acts is None or grads is None:
            raise RuntimeError("Activations or gradients not captured by hooks.")

        # support acts shape [B, N, C] from Swin token outputs
        if acts.ndim == 3:  # [B, N, C]
            weights = grads.mean(dim=2, keepdim=True)  # [B, N, 1]
            cam = (weights * acts).sum(dim=2).squeeze(0)  # [N]
            size = int(math.sqrt(cam.shape[0]))
            cam_map = cam.reshape(size, size).cpu().numpy()
        elif acts.ndim == 4:  # [B, C, H, W]
            weights = grads.mean(dim=1, keepdim=True)
            cam = (weights * acts).sum(dim=1).squeeze(0)
            cam_map = cam.cpu().numpy()
        else:
            raise RuntimeError(f"Unsupported activation shape: {acts.shape}")

        # optional: move model back to original device if needed
        if ran_on_cpu and DEVICE.type != 'cpu':
            try:
                self.model.to(DEVICE)
            except Exception:
                pass

        cam_map = np.maximum(cam_map, 0)
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
        return cam_map

# choose a target layer from backbone (deep layer)
# If backbone structure differs in versions, this may need small adjustment
try:
    target_layer = model.backbone.layers[-1].blocks[-1].norm2
except Exception:
    # fallback: try another deep module
    target_layer = list(model.backbone.children())[-1]
gradcam = SwinGradCAM(model, target_layer)

# ---------------- CSV INIT ----------------
if not os.path.exists(CSV_LOG):
    pd.DataFrame(columns=["Timestamp", "PersonID", "Snapshot", "Heatmap", "PredictedEmotion",
                          "Confidence", "MentalScore", "Risk", "MotionScore", "BlinkScore"]).to_csv(CSV_LOG, index=False)

# ---------------- Buffers for smoothing ----------------
emotion_buffer = deque(maxlen=ROLL_EMOTION)
motion_buffer = deque(maxlen=ROLL_MOTION)
blink_buffer = deque(maxlen=ROLL_BLINK)

person_counter = 0
frame_idx = 0
prev_gray_face = None

# ---------------- Utility functions ----------------
def compute_weighted_emotion_score(prob_array):
    weighted = 0.0
    for i, p in enumerate(prob_array):
        weighted += float(p) * EMOTION_WEIGHTS[EMOTIONS[i]]
    max_possible = max(EMOTION_WEIGHTS.values()) * 1.0
    norm = (weighted / (max_possible * 1.0)) * 100.0
    return float(np.clip(norm, 0.0, 100.0))

def compute_motion_score(flow_mag_mean):
    ms = min(flow_mag_mean * 40.0, 100.0)
    return float(ms)

def compute_blink_score(eye_open_ratio):
    fatigue = (1.0 - eye_open_ratio) * 100.0
    return float(np.clip(fatigue, 0.0, 100.0))

def fusion_mental_score(emotion_component, motion_component, blink_component,
                        w_e=0.6, w_m=0.25, w_b=0.15):
    ms = w_e * emotion_component + w_m * motion_component + w_b * blink_component
    return float(np.clip(ms, 0.0, 100.0))

def ms_to_risk(ms):
    if ms >= 70:
        return "Very High Risk"
    elif ms >= 55:
        return "High Risk"
    elif ms >= 45:
        return "Moderate Risk"
    else:
        return "Low Risk"

# ---------------- Google Drive Upload ----------------
def upload_to_drive(local_folder=SNAPSHOT_DIR, csv_file=CSV_LOG, folder_title="EmotionMonitoringCloud"):
    try:
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)
    except Exception as e:
        print("Drive auth failed:", e)
        return False

    folder = drive.CreateFile({'title': folder_title, 'mimeType': 'application/vnd.google-apps.folder'})
    folder.Upload()
    folder_id = folder['id']

    try:
        file = drive.CreateFile({'parents':[{'id': folder_id}], 'title': os.path.basename(csv_file)})
        file.SetContentFile(csv_file)
        file.Upload()
    except Exception as e:
        print("Failed to upload CSV:", e)

    for fname in os.listdir(local_folder):
        path = os.path.join(local_folder, fname)
        try:
            file = drive.CreateFile({'parents':[{'id': folder_id}], 'title': fname})
            file.SetContentFile(path)
            file.Upload()
        except Exception as e:
            print("Failed to upload:", fname, e)
    return True

# ---------------- Safe model call helpers ----------------
def safe_forward_no_grad(input_t):
    """
    Run a no-grad forward pass safely with CPU fallback on NotImplementedError (MPS).
    Returns model outputs on CPU (detached) or on DEVICE as returned.
    """
    try:
        with torch.no_grad():
            return model(input_t)
    except NotImplementedError as e:
        print("Model forward hit NotImplementedError on device; retrying on CPU.")
        # move model to CPU, run there
        model_cpu = model.to('cpu')
        input_cpu = input_t.to('cpu')
        with torch.no_grad():
            out = model_cpu(input_cpu)
        # move model back if necessary
        if DEVICE.type != 'cpu':
            try:
                model.to(DEVICE)
            except Exception:
                pass
        return out

# ---------------- Main loop ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Ensure camera is available and permitted.")

print("▶ Running improved Real-Time Emotion Monitor. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        display_frame = frame.copy()
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_full, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_roi_color = frame[y:y+h, x:x+w]
            face_roi_gray = gray_full[y:y+h, x:x+w]

            # --- prepare input and run model ---
            pil_face = Image.fromarray(cv2.cvtColor(face_roi_color, cv2.COLOR_BGR2RGB))
            input_t = transform(pil_face).unsqueeze(0).to(DEVICE)

            # inference (safe wrapper)
            outputs = safe_forward_no_grad(input_t)
            probs_tensor = torch.softmax(outputs, dim=1)[0].detach().cpu().numpy()

            # push into rolling buffer
            emotion_buffer.append(probs_tensor)
            avg_probs = np.mean(np.stack(list(emotion_buffer)), axis=0)
            emotion_component = compute_weighted_emotion_score(avg_probs)

            # --- blink estimation via Haar eye detection (approx) ---
            eyes = eye_cascade.detectMultiScale(cv2.cvtColor(face_roi_color, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=3, minSize=(10,10))
            eye_open = 1 if len(eyes) > 0 else 0
            blink_buffer.append(eye_open)
            open_ratio = float(np.mean(blink_buffer)) if len(blink_buffer) > 0 else 1.0
            blink_component = compute_blink_score(open_ratio)

            # --- motion / restlessness via optical flow on ROI ---
            motion_mean = 0.0
            if prev_gray_face is not None and prev_gray_face.shape == face_roi_gray.shape:
                flow = cv2.calcOpticalFlowFarneback(prev_gray_face, face_roi_gray,
                                                    None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                motion_mean = float(np.mean(mag))
            else:
                motion_mean = 0.0
            motion_buffer.append(motion_mean)
            avg_motion = float(np.mean(motion_buffer)) if len(motion_buffer) > 0 else 0.0
            motion_component = compute_motion_score(avg_motion)

            prev_gray_face = face_roi_gray.copy()

            # --- fuse into final MS ---
            final_ms = fusion_mental_score(emotion_component, motion_component, blink_component,
                                          w_e=0.6, w_m=0.25, w_b=0.15)
            risk = ms_to_risk(final_ms)

            # --- Grad-CAM heatmap generation (safe) ---
            heatmap_to_save = None
            try:
                # Grad-CAM expects gradients; ensure input on same device as model
                cam_input = transform(pil_face).unsqueeze(0).to(next(model.parameters()).device)
                cam_map = gradcam.generate_cam(cam_input, None)
                cam_map = cv2.resize(cam_map, (w, h), interpolation=cv2.INTER_LINEAR)
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
                # blend heatmap with face
                overlay = cv2.addWeighted(face_roi_color, 0.5, heatmap, 0.5, 0)
                overlay = cv2.copyMakeBorder(overlay, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0,255,0))
                overlay_crop = overlay[2:2+h, 2:2+w]
                display_frame[y:y+h, x:x+w] = overlay_crop
                heatmap_to_save = heatmap.copy()
            except Exception as e:
                # fallback: show original face
                display_frame[y:y+h, x:x+w] = face_roi_color
                heatmap_to_save = None
                # optionally log
                # print("Grad-CAM failed:", e)

            # --- annotate frame ---
            pred_idx = int(np.argmax(avg_probs))
            pred_label = EMOTIONS[pred_idx]
            confidence = float(avg_probs[pred_idx]) * 100.0

            label_text = f"{pred_label} {confidence:.1f}% | MS:{final_ms:.1f} | {risk}"
            cv2.putText(display_frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0,255,0), 2)

            # --- conditionally save snapshot + heatmap + csv row ---
            if confidence >= SAVE_CONF_THRESHOLD and (frame_idx % 10 == 0):
                person_counter += 1
                pid = f"Person{person_counter}"
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                snap_name = f"{pid}_{ts}.jpg"
                heat_name = f"{pid}_{ts}_heat.jpg"
                snap_path = os.path.join(SNAPSHOT_DIR, snap_name)
                heat_path = os.path.join(SNAPSHOT_DIR, heat_name)
                cv2.imwrite(snap_path, display_frame)
                if heatmap_to_save is not None:
                    # heatmap is BGR from applyColorMap: convert to RGB only if you need RGB
                    cv2.imwrite(heat_path, heatmap_to_save)
                else:
                    heat_path = ""
                row = {
                    "Timestamp": datetime.now().isoformat(),
                    "PersonID": pid,
                    "Snapshot": snap_path,
                    "Heatmap": heat_path,
                    "PredictedEmotion": pred_label,
                    "Confidence": round(confidence, 3),
                    "MentalScore": round(final_ms, 3),
                    "Risk": risk,
                    "MotionScore": round(motion_component,3),
                    "BlinkScore": round(blink_component,3)
                }
                pd.DataFrame([row]).to_csv(CSV_LOG, mode='a', header=False, index=False)
                print(f"Saved {snap_name} | Emotion={pred_label} | MS={final_ms:.1f}")

        # show frame
        cv2.imshow("Improved Emotion Monitor (press 'q' to quit)", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed. Uploading results to Google Drive...")

# Upload to Google Drive (attempt)
uploaded = upload_to_drive()
if uploaded:
    print("Upload completed.")
else:
    print("Upload failed or skipped. Check config.")
print("Done. CSV at:", CSV_LOG)
