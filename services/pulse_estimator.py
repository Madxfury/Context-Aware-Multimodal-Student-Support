"""Pulse rate and SpO2 from the wearable's raw red/IR waveform, estimated on the computer.

Worn on the hand, the MAX30102 sees a pulse of only 0.02-0.1% of the light level, and
timing single beats is unreliable at that size. Recordings from the LY board showed:
- The IR channel carries the pulse. Its rhythm (78-88 BPM) matched the wearer's own
  pulse count (75-90 BPM, then 88 BPM) on 2026-11-19.
- The red channel was dominated by a steady 64-70 BPM rhythm with strong 2nd and 3rd
  harmonics that was not the heart. This is the signature of room-light flicker (LED or
  PWM lighting) aliased by the sensor's sampling. LED room light contains red but almost
  no infrared, so it shows mainly in the red channel and weakly in IR.

The pulse rate comes from the IR spectrum of the last 10 s:
- The strongest IR rhythm between 42 and 180 BPM is used, except that the red channel's
  dominant harmonic rhythm is ignored in IR when it is flicker: when IR's strongest
  rhythm differs from it, or when it moves the two channels in opposite directions (the
  flicker did, correlation -0.7 to -0.99). Blood absorbs both colours, so a real pulse
  seen by both channels (as on a fingertip) moves them together and is kept.
- The hand must be still for the whole window (accelerometer). Rhythmic hand movement
  makes periodic light changes that look like a pulse (44-61 BPM on the LY board).
- The rhythm must hold at least MIN_QUALITY of the IR signal power, with skin contact.
- Continuity: the first reading needs two agreeing windows, and later windows must stay
  within MAX_JUMP of the recent rate. A real change is accepted once RELOCK_WINDOWS
  consecutive windows agree on the new rate.
Accepted windows are combined with a short median so one window cannot jump the reading.

SpO2 uses the red/IR "ratio of ratios" by regression of the two channels filtered
around the pulse rate and its 2nd harmonic, with Maxim's reference curve. The method
assumes both channels carry the same pulse, moving together, as on a fingertip. On the
hand they did not, so SpO2 is only reported when the channels move together
(correlation >= MIN_SPO2_CORRELATION) in every recent window with consistent values.
The sensor is uncalibrated: the value is an estimate, not a medical measurement.
"""

from collections import deque
from dataclasses import dataclass
from math import isfinite

import numpy as np

WINDOW_SECONDS = 10.0
MIN_BPM, MAX_BPM = 42.0, 180.0
GRID_STEP_HZ = 0.005
MIN_QUALITY = 0.28            # share of band power at f0 and 2*f0 (0-1)
INTERFERENCE_SHARE = 0.25     # red power in its own rhythm (f and 2f) that marks room-light flicker
INTERFERENCE_MASK_BPM = 3.0   # half-width of the IR frequencies ignored around each flicker harmonic
SAME_PULSE_CORRELATION = 0.5  # a rhythm in both channels is the pulse only if they move together
STILL_MOTION = 0.15           # m/s^2 of gravity-free movement, 90th percentile over the window
MAX_MOTION = 0.25             # m/s^2, highest value in the window (still windows peaked at 0.18)
MIN_CONTACT_LEVEL = 8000.0    # IR counts; below this the sensor is not on skin (firmware threshold)
MIN_SPO2_CORRELATION = 0.9
MAX_SPO2_SPREAD = 3.0         # percentage points across the windows used
SMOOTHING = 3                 # median of the last accepted windows
MAX_JUMP = 0.12               # largest change from the recent rate for one window
CONFIRM_SPREAD = 0.06         # windows confirming a (new) rate agree within this
RELOCK_WINDOWS = 3
LOCK_SECONDS = 60.0           # the recent rate is the median of accepted windows this recent
MAX_GAP_SECONDS = 20.0        # accepted windows older than this are not combined
BAND = (0.6, 5.0)             # Hz, power used for normalisation and quality
PAD = 1 << 14


@dataclass
class PulseEstimate:
    bpm: float
    quality: float
    spo2: float | None


def spo2_from_ratio(ratio):
    value = -45.060 * ratio * ratio + 30.354 * ratio + 94.845
    return min(value, 100.0)


def _detrended(signal):
    x = np.asarray(signal, dtype=float)
    t = np.arange(len(x))
    return x - np.polyval(np.polyfit(t, x, 1), t)


class PulseEstimator:
    def __init__(self, sample_rate=100.0):
        self.sample_rate = sample_rate
        size = int(WINDOW_SECONDS * 120)  # room for sample rates up to 120 Hz
        self.red = deque(maxlen=size)
        self.ir = deque(maxlen=size)
        self.motion = deque()          # (sample count when received, movement)
        self.next_index = None
        self.samples = 0
        self.accepted = deque()        # (time s, bpm, spo2 or None, quality)
        self.lock_history = deque()    # (time s, bpm) of accepted windows, for continuity
        self.candidates = []           # consecutive windows that passed the gates but not continuity
        self.last = None               # diagnostics of the latest window

    def restart_window(self):
        """Drop the waveform (discontinuity); accepted readings are kept."""
        self.red.clear()
        self.ir.clear()
        self.motion.clear()

    def add_motion(self, movement):
        if movement is not None and isfinite(movement):
            self.motion.append((self.samples, movement))

    def add_samples(self, first_index, pairs, sample_rate=None):
        """Add consecutive (red, ir) samples. An index gap restarts the window."""
        if sample_rate and 80.0 < sample_rate < 120.0:
            self.sample_rate = sample_rate
        if self.next_index is not None and first_index != self.next_index:
            self.restart_window()
        for red, ir in pairs:
            self.red.append(float(red))
            self.ir.append(float(ir))
        self.samples += len(pairs)
        self.next_index = first_index + len(pairs)

    def window_ready(self):
        return len(self.ir) >= int(WINDOW_SECONDS * self.sample_rate)

    def _normalised_power(self, signal, freqs_needed):
        x = _detrended(signal) * np.hanning(len(signal))
        power = np.abs(np.fft.rfft(x, PAD)) ** 2
        freqs = np.fft.rfftfreq(PAD, 1.0 / self.sample_rate)
        band = (freqs >= BAND[0]) & (freqs <= BAND[1])
        total = power[band].sum()
        if total <= 0:
            return None
        return np.interp(freqs_needed, freqs, power / total)

    def estimate(self, now_seconds):
        """Evaluate the current window. Returns the smoothed PulseEstimate or None."""
        needed = int(WINDOW_SECONDS * self.sample_rate)
        if len(self.ir) < needed:
            self.last = None
            return self.current(now_seconds)
        red = list(self.red)[-needed:]
        ir = list(self.ir)[-needed:]
        start = self.samples - needed
        while self.motion and self.motion[0][0] < start:
            self.motion.popleft()
        movement = [value for _, value in self.motion]
        still = (bool(movement) and float(np.percentile(movement, 90)) <= STILL_MOTION
                 and max(movement) <= MAX_MOTION)
        contact = min(np.mean(ir), np.mean(red)) >= MIN_CONTACT_LEVEL

        grid = np.arange(MIN_BPM / 60.0, MAX_BPM / 60.0 + GRID_STEP_HZ / 2, GRID_STEP_HZ)
        ir_fundamental = self._normalised_power(ir, grid)
        ir_harmonic = self._normalised_power(ir, 2 * grid)
        red_fundamental = self._normalised_power(red, grid)
        red_harmonic = self._normalised_power(red, 2 * grid)
        if any(spectrum is None for spectrum in (ir_fundamental, ir_harmonic, red_fundamental, red_harmonic)):
            self.last = None
            return self.current(now_seconds)
        # Share of power within 0.08 Hz of f and 0.16 Hz of 2f. The spectra are normalised
        # per FFT bin, so grid sums are rescaled to bin density.
        to_share = GRID_STEP_HZ / (self.sample_rate / PAD)

        # The red channel's dominant harmonic rhythm. Blood absorbs both colours, so a pulse
        # moves the two channels together. A strong red rhythm that IR does not share, or
        # that moves the channels apart, is room-light flicker: ignore its harmonics in IR.
        red_rhythm = grid[int(np.argmax(red_fundamental * red_harmonic))]
        near_red = np.abs(grid - red_rhythm) <= 0.08
        red_share = float((red_fundamental[near_red].sum() + red_harmonic[near_red].sum()) * to_share)
        ir_strongest = grid[int(np.argmax(ir_fundamental))]
        interference = False
        if red_share >= INTERFERENCE_SHARE:
            if abs(ir_strongest - red_rhythm) * 60.0 > INTERFERENCE_MASK_BPM:
                interference = True
            else:
                _, together = self._ratio(red, ir, red_rhythm)
                interference = together is None or together < SAME_PULSE_CORRELATION
        # Only real spectral peaks are candidates. Each peak is about +/-12 BPM wide in a
        # 10 s window, so the slope beside an ignored flicker peak must not be picked.
        score = ir_fundamental
        peaks = [i for i in range(1, len(grid) - 1) if score[i] > score[i - 1] and score[i] >= score[i + 1]]
        if interference:
            peaks = [i for i in peaks if all(
                abs(grid[i] - multiple * red_rhythm) * 60.0 > INTERFERENCE_MASK_BPM
                for multiple in (0.5, 1.0, 1.5, 2.0, 3.0))]
        if not peaks:
            self.last = {"bpm": None, "quality": 0.0, "still": still, "contact": bool(contact), "at_edge": False,
                         "accepted": False, "gated": False, "red_ir_correlation": None,
                         "light_interference_bpm": round(60.0 * float(red_rhythm), 1) if interference else None}
            return self.current(now_seconds)
        k = max(peaks, key=lambda i: score[i])
        f0 = grid[k]
        a, b, c = score[k - 1:k + 2]
        curve = a - 2 * b + c
        if curve < 0:
            f0 += GRID_STEP_HZ * 0.5 * (a - c) / curve
        at_edge = k < 4 or k > len(grid) - 5
        near = np.abs(grid - f0) <= 0.08
        quality = float((ir_fundamental[near].sum() + ir_harmonic[near].sum()) * to_share)

        ratio, correlation = self._ratio(red, ir, f0)
        spo2 = None
        if (ratio is not None and correlation >= MIN_SPO2_CORRELATION and 0.3 <= ratio <= 1.2):
            spo2 = spo2_from_ratio(ratio)
            if spo2 < 70.0:
                spo2 = None
        bpm = 60.0 * f0
        gated = still and contact and not at_edge and quality >= MIN_QUALITY
        accepted = gated and self._continuous(now_seconds, bpm, spo2, quality)
        self.last = {
            "bpm": round(bpm, 1), "quality": round(quality, 3), "still": still, "contact": bool(contact),
            "at_edge": at_edge, "accepted": accepted,
            "light_interference_bpm": round(60.0 * float(red_rhythm), 1) if interference else None,
            "red_ir_correlation": None if correlation is None else round(correlation, 2),
            "gated": gated,
        }
        if accepted:
            self.accepted.append((now_seconds, bpm, spo2, quality))
            while len(self.accepted) > SMOOTHING:
                self.accepted.popleft()
        return self.current(now_seconds)

    def _continuous(self, now_seconds, bpm, spo2=None, quality=0.0):
        """Accept a window that passed the signal gates if it fits the recent rate."""
        while self.lock_history and now_seconds - self.lock_history[0][0] > LOCK_SECONDS:
            self.lock_history.popleft()
        if self.candidates and now_seconds - self.candidates[-1][0] > 3.0:
            self.candidates = []  # confirmations must be consecutive windows
        if self.lock_history:
            locked = float(np.median([value for _, value in self.lock_history]))
            if abs(bpm - locked) <= MAX_JUMP * locked:
                self.candidates = []
                self.lock_history.append((now_seconds, bpm))
                return True
            needed = RELOCK_WINDOWS
        else:
            needed = 2
        self.candidates.append((now_seconds, bpm, spo2, quality))
        recent = [entry[1] for entry in self.candidates[-needed:]]
        if len(recent) == needed and max(recent) - min(recent) <= CONFIRM_SPREAD * min(recent):
            # A new rate confirmed by consecutive windows: start tracking it.
            self.lock_history = deque((entry[0], entry[1]) for entry in self.candidates[-needed:])
            self.accepted = deque(self.candidates[-needed:-1])
            self.candidates = []
            return True
        return False

    def _ratio(self, red, ir, f0):
        """Regression slope of red on IR around f0 and 2*f0, scaled by the DC levels."""
        filtered, levels = [], []
        for signal in (red, ir):
            level = float(np.mean(signal))
            if level <= 0:
                return None, None
            x = _detrended(signal) * np.hanning(len(signal))
            spectrum = np.fft.rfft(x)
            freqs = np.fft.rfftfreq(len(x), 1.0 / self.sample_rate)
            keep = (np.abs(freqs - f0) <= 0.1) | (np.abs(freqs - 2 * f0) <= 0.2)
            filtered.append(np.fft.irfft(np.where(keep, spectrum, 0), len(x)))
            levels.append(level)
        (red_ac, ir_ac), (red_dc, ir_dc) = filtered, levels
        covariance = float(np.mean(red_ac * ir_ac))
        ir_power = float(np.mean(ir_ac * ir_ac))
        red_power = float(np.mean(red_ac * red_ac))
        if ir_power <= 0 or red_power <= 0:
            return None, None
        correlation = covariance / np.sqrt(ir_power * red_power)
        return covariance / ir_power * ir_dc / red_dc, float(correlation)

    def current(self, now_seconds):
        """Median of accepted windows from the last MAX_GAP_SECONDS, or None."""
        recent = [entry for entry in self.accepted if now_seconds - entry[0] <= MAX_GAP_SECONDS]
        if not recent:
            return None
        oxygen = [entry[2] for entry in recent]
        spo2 = None
        if (len(oxygen) == SMOOTHING and all(value is not None for value in oxygen)
                and max(oxygen) - min(oxygen) <= MAX_SPO2_SPREAD):
            spo2 = round(float(np.median(oxygen)), 1)
        return PulseEstimate(
            bpm=round(float(np.median([entry[1] for entry in recent])), 1),
            quality=round(float(np.median([entry[3] for entry in recent])), 3),
            spo2=spo2,
        )
