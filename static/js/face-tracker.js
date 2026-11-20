// Real-time face tracking for the camera check-in, on this computer (MediaPipe Face Landmarker).
//
// Each video frame gives 478 face landmarks and 52 facial movement scores. Samvya uses them to
// keep only clear frames (facing the camera, close enough, lit, still, not mid-speech), to cut
// a level face crop framed like the training images, and to describe facial movements
// (smiling, brows lowered...) against the student's own check-in baseline.
import { FaceLandmarker, FilesetResolver } from '../vendor/mediapipe/vision_bundle.mjs';

const VENDOR = new URL('../vendor/mediapipe/', import.meta.url).href;

// Landmark indices in the MediaPipe face mesh.
const EYE_OUTER_RIGHT = 33;   // the person's right eye, on the left of an unmirrored image
const EYE_OUTER_LEFT = 263;
const NOSE_TIP = 1;
const FOREHEAD = 10;
const CHIN = 152;
const CHEEK_RIGHT = 234;
const CHEEK_LEFT = 454;

// How the FANE training crops frame a face, measured by running this tracker on 576 training
// images: the crop is as wide as the landmark box (median 1.00) and 1.12 times as tall, and its
// centre sits 6.4% of the box height above the box centre (more forehead than chin).
export const CROP = { width: 1.0, height: 1.12, offsetX: -0.02, offsetY: -0.064 };

// Quality limits. Calibrated on frontal FANE faces and a live webcam (see README).
export const LIMITS = {
  minFaceWidth: 0.16,     // face width / frame width: closer than about 1 m
  // Pixels across the face. On the author's webcam frames, accuracy fell from 76% at 290 px to
  // 54% at 145 px for Swin-FANE alone, which then read neutral faces as "Angry".
  minFacePixels: 200,
  maxYaw: 0.16,           // nose offset from the cheek midpoint, as a share of face width
  minPitch: 0.40, maxPitch: 0.72,  // nose height between forehead and chin
  maxRoll: 0.45,          // radians (about 25 degrees); smaller tilts are levelled in the crop
  minBrightness: 55, maxBrightness: 235,
  maxMove: 0.035,         // face-centre movement per frame, as a share of face width
  talkingJaw: 0.06,       // spread of jaw opening over the last second while talking
  eyesClosed: 0.6,
};

export async function createFaceLandmarker(runningMode = 'VIDEO') {
  const fileset = await FilesetResolver.forVisionTasks(VENDOR + 'wasm');
  const options = delegate => ({
    baseOptions: { modelAssetPath: VENDOR + 'face_landmarker.task', delegate },
    runningMode, numFaces: 1, outputFaceBlendshapes: true,
    minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  try {
    return await FaceLandmarker.createFromOptions(fileset, options('GPU'));
  } catch {
    return await FaceLandmarker.createFromOptions(fileset, options('CPU'));
  }
}

/** Head pose and framing from landmarks (normalised coordinates) in a frame of width x height. */
export function faceGeometry(landmarks, width, height) {
  const point = i => ({ x: landmarks[i].x * width, y: landmarks[i].y * height });
  const right = point(EYE_OUTER_RIGHT), left = point(EYE_OUTER_LEFT);
  const roll = Math.atan2(left.y - right.y, left.x - right.x);
  const centre = { x: (right.x + left.x) / 2, y: (right.y + left.y) / 2 };
  const cos = Math.cos(-roll), sin = Math.sin(-roll);
  const level = p => ({ x: centre.x + (p.x - centre.x) * cos - (p.y - centre.y) * sin,
                        y: centre.y + (p.x - centre.x) * sin + (p.y - centre.y) * cos });
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < landmarks.length; i++) {
    const p = level({ x: landmarks[i].x * width, y: landmarks[i].y * height });
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const nose = level(point(NOSE_TIP)), cheekR = level(point(CHEEK_RIGHT)), cheekL = level(point(CHEEK_LEFT));
  const top = level(point(FOREHEAD)), chin = level(point(CHIN));
  return {
    roll, centre,
    box: { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY },
    yaw: (nose.x - cheekR.x) / (cheekL.x - cheekR.x || 1) - 0.5,
    pitch: (nose.y - top.y) / (chin.y - top.y || 1),
  };
}

/** Draw a level face crop framed like the training images; returns the canvas. */
export function alignedCrop(source, geometry, canvas, maxSide = 288) {
  const { box, roll, centre } = geometry;
  const cropW = box.width * CROP.width, cropH = box.height * CROP.height;
  const cx = (box.minX + box.maxX) / 2 + CROP.offsetX * box.width;
  const cy = (box.minY + box.maxY) / 2 + CROP.offsetY * box.height;
  const scale = Math.min(1, maxSide / Math.max(cropW, cropH));
  canvas.width = Math.max(1, Math.round(cropW * scale));
  canvas.height = Math.max(1, Math.round(cropH * scale));
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.save();
  // canvas = scale * (level(image point) - crop corner), where level() rotates by -roll about
  // the eye midpoint, exactly as faceGeometry() does.
  ctx.scale(scale, scale);
  ctx.translate(cropW / 2 - cx, cropH / 2 - cy);
  ctx.translate(centre.x, centre.y);
  ctx.rotate(-roll);
  ctx.translate(-centre.x, -centre.y);
  ctx.drawImage(source, 0, 0);
  ctx.restore();
  return canvas;
}

/** Mean luminance (0-255) of a canvas. */
export function brightness(canvas) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  let sum = 0;
  for (let i = 0; i < data.length; i += 16) sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  return sum / (data.length / 16);
}

/** Facial movement scores (0-1), averaged over both sides of the face. */
export function movementCues(blendshapes) {
  const score = {};
  for (const category of blendshapes?.categories || []) score[category.categoryName] = category.score;
  const mean = (...names) => names.reduce((total, name) => total + (score[name] || 0), 0) / names.length;
  return {
    smile: mean('mouthSmileLeft', 'mouthSmileRight'),
    frown: mean('mouthFrownLeft', 'mouthFrownRight'),
    brow_down: mean('browDownLeft', 'browDownRight'),
    brow_raise: mean('browInnerUp', 'browOuterUpLeft', 'browOuterUpRight'),
    eye_openness: 1 - mean('eyeBlinkLeft', 'eyeBlinkRight'),
    jaw_open: score.jawOpen || 0,
  };
}

/**
 * Tracks one video stream: call `track(video, now)` for each frame. Returns what the frame
 * shows and why it is or is not clear enough to classify.
 */
export class FaceTracker {
  constructor(landmarker) {
    this.landmarker = landmarker;
    this.previous = null;
    this.jaw = [];
    this.lastTimestamp = -1;
  }

  track(video, now) {
    const timestamp = Math.max(this.lastTimestamp + 1, Math.round(now));
    this.lastTimestamp = timestamp;
    const result = this.landmarker.detectForVideo(video, timestamp);
    const landmarks = result.faceLandmarks?.[0];
    if (!landmarks) {
      this.previous = null;
      this.jaw = [];
      return { face: false, reason: 'no_face' };
    }
    const width = video.videoWidth, height = video.videoHeight;
    const geometry = faceGeometry(landmarks, width, height);
    const cues = movementCues(result.faceBlendshapes?.[0]);
    this.jaw.push({ t: now, value: cues.jaw_open });
    this.jaw = this.jaw.filter(entry => now - entry.t <= 1000);
    const jawValues = this.jaw.map(entry => entry.value);
    const jawSpread = jawValues.length > 3 ? Math.max(...jawValues) - Math.min(...jawValues) : 0;
    const move = this.previous
      ? Math.hypot(geometry.centre.x - this.previous.x, geometry.centre.y - this.previous.y) / geometry.box.width
      : 0;
    this.previous = geometry.centre;

    let reason = null;
    if (geometry.box.width / width < LIMITS.minFaceWidth || geometry.box.width < LIMITS.minFacePixels) reason = 'too_far';
    else if (Math.abs(geometry.yaw) > LIMITS.maxYaw || geometry.pitch < LIMITS.minPitch || geometry.pitch > LIMITS.maxPitch) reason = 'turned';
    else if (Math.abs(geometry.roll) > LIMITS.maxRoll) reason = 'tilted';
    else if (move > LIMITS.maxMove) reason = 'moving';
    else if (jawSpread > LIMITS.talkingJaw) reason = 'talking';
    else if (cues.eye_openness < 1 - LIMITS.eyesClosed) reason = 'eyes_closed';
    return { face: true, reason, geometry, cues, landmarks, jawSpread, move };
  }
}

export const QUALITY_HINTS = {
  no_face: 'Look at the camera so your face is in view.',
  too_far: 'Move a little closer to the camera.',
  turned: 'Face the camera directly.',
  tilted: 'Keep your head level.',
  moving: 'Hold still for a moment.',
  talking: 'Pause talking for a moment.',
  eyes_closed: 'Keep your eyes open for a moment.',
  dark: 'Add a little more light on your face.',
  bright: 'Too much light behind or on your face.',
};
