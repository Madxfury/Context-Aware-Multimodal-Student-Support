"""Read the LY ESP32-S3 over USB: telemetry, the raw pulse waveform, and check-in trends.

Pulse rate (and SpO2 where it can be measured) is estimated here from the waveform; see
pulse_estimator.py. Every reading keeps its last good value with its age, so the
dashboard shows the most recent measurement instead of a gap, marked live or held.
Trends compare each check-in with its own first minute; they describe the signals and
are not a diagnosis.
"""

from collections import OrderedDict, deque
from math import isfinite
import os
from statistics import mean, median, pstdev
from threading import Lock, Thread
from time import monotonic, sleep

import serial
from serial.tools import list_ports

from services.pulse_estimator import PulseEstimator


PULSE_STATES = {"NO_FINGER", "BAD_SIGNAL", "WAITING", "DETECTING", "VALID"}
TEMP_SENSOR_STATES = {"OK": "connected", "MISSING": "missing"}
SERIAL_RATE = 115200
FRESH_SECONDS = 3            # telemetry older than this is stale
LIVE_SECONDS = 3             # a reading this recent is shown as live, older ones as held
HOLD_SECONDS = 600           # a held reading older than this is no longer shown
HISTORY_SECONDS = 2 * 3600   # one value per second per signal, kept for check-in trends
WAVEFORM_TIMEOUT = 5         # without PPG lines (firmware v1) the firmware's own pulse is used
STILL, ACTIVE = 0.15, 1.0    # m/s^2 of gravity-free movement
BASELINE_SECONDS = 60
TREND_RECENT, TREND_EARLIER = 60, 180
SIGNALS = ("pulse", "spo2", "temperature", "gsr", "gsr_adc", "motion")
CONDUCTANCE_CHANGE = 0.03    # GSR voltage change (fraction) treated as a skin conductance change


def accel_offset(text):
    """Zero-g offset (m/s^2) subtracted from the accelerometer axes.

    The default is this project's MPU6050: still readings in five orientations (palm
    down and up, on its side, flat on the table) gave |g| of 8.9-11.0 m/s^2 because Z
    reads 1.19 m/s^2 low; corrected, |g| is 9.73-9.84 (local g 9.79). Set
    SAMVYA_ACCEL_OFFSET="x,y,z" for another board, or "0,0,0" to turn it off.
    """
    try:
        values = [float(part) for part in text.split(",")]
    except ValueError:
        values = []
    return values if len(values) == 3 and all(isfinite(v) and abs(v) < 5 for v in values) else [0.0, 0.0, -1.19]


ACCEL_OFFSET = accel_offset(os.environ.get("SAMVYA_ACCEL_OFFSET", "0,0,-1.19"))


def parse_telemetry(line):
    if not line.startswith("TIME="):
        return None
    fields = {}
    for part in line.split("|"):
        key, separator, value = part.strip().partition("=")
        if separator:
            fields[key] = value.strip()
        elif key.endswith(" mV"):
            fields["GSR_MV"] = key
    uptime = measurement(fields, "TIME", " ms", 0, 2**32 - 1)
    pulse_state = fields.get("STATUS")
    if uptime is None or pulse_state not in PULSE_STATES:
        return None

    accel = vector(fields.get("ACC", ""))
    gyro = vector(fields.get("GYRO", ""))
    if accel == [0.0, 0.0, 0.0] and gyro == [0.0, 0.0, 0.0]:
        # Firmware v1 leaves both vectors zero when the MPU6050 is missing.
        accel = gyro = None
    if accel is not None:
        accel = [round(value - offset, 2) for value, offset in zip(accel, ACCEL_OFFSET)]
    pulse_valid = pulse_state == "VALID"
    return {
        "device_ms": int(uptime),
        "pulse_status": pulse_state,
        "hr": measurement(fields, "HR", " BPM", 30, 240) if pulse_valid else None,
        # Firmware v2 only. An uncalibrated red/IR ratio estimate, not a medical reading.
        "spo2": measurement(fields, "SPO2", " %", 70, 100) if pulse_valid else None,
        "temperature_c": measurement(fields, "TEMP", " C", -55, 150),
        "temperature_sensor": TEMP_SENSOR_STATES.get(fields.get("TEMP_SENSOR")),
        "gsr_adc": measurement(fields, "GSR", " ADC", 0, 4095),
        "gsr_mv": measurement(fields, "GSR_MV", " mV", 0, 5000),
        "ir": measurement(fields, "IR", "", 0, 262143),
        "accel": accel,
        "gyro": gyro,
        # Firmware v2 only: acceleration with gravity removed, so 0 means not moving.
        "motion": measurement(fields, "MOTION", " m/s2", 0, 200),
        "led": led_current(fields.get("LED", "")),
    }


def led_current(text):
    try:
        value = int(text, 16)
    except ValueError:
        return None
    return value if 0 <= value <= 0xFF else None


def parse_waveform(line):
    """'PPG <rate> <first index> <red> <ir> ...' -> (rate, first index, [(red, ir), ...])."""
    parts = line.split()
    if len(parts) < 5 or parts[0] != "PPG" or (len(parts) - 3) % 2:
        return None
    try:
        rate = float(parts[1])
        first = int(parts[2])
        values = [int(value) for value in parts[3:]]
    except ValueError:
        return None
    if not 50.0 < rate < 200.0 or first < 0 or any(not 0 <= value <= 262143 for value in values):
        return None
    for channel in (values[0::2], values[1::2]):
        # Within one 0.1 s line the light level changes by well under 1% (LED steps start a
        # new index), so a value far from the rest means bytes were lost on the USB link,
        # as in a recorded line where an IR sample of ~85,000 arrived as "86".
        middle = median(channel)
        if any(abs(value - middle) > 0.25 * middle for value in channel):
            return None
    return rate, first, list(zip(values[0::2], values[1::2]))


def measurement(fields, key, unit, lower, upper):
    text = fields.get(key, "")
    if not text.endswith(unit):
        return None
    try:
        value = float(text.removesuffix(unit))
    except ValueError:
        return None
    return value if lower <= value <= upper else None


def vector(value):
    try:
        axes = [float(axis) for axis in value.split(",")]
    except ValueError:
        return None
    return axes if len(axes) == 3 and all(isfinite(axis) for axis in axes) else None




def _rounded(value, digits):
    return None if value is None else round(value, digits)


def _baseline(series):
    """Median of the first minute of a check-in (at least the first 10 values)."""
    if not series:
        return None
    first = series[0][0]
    values = [value for when, value in series if when - first <= BASELINE_SECONDS]
    if len(values) < 10:
        values = [value for _, value in series[:10]]
    return median(values)


def _trend(series, now, threshold):
    """Compare the last minute with the two minutes before it."""
    recent = [value for when, value in series if when >= now - TREND_RECENT]
    earlier = [value for when, value in series if now - TREND_EARLIER <= when < now - TREND_RECENT]
    if len(recent) < 10 or len(earlier) < 10:
        return None
    change = median(recent) - median(earlier)
    direction = "rising" if change >= threshold else "falling" if change <= -threshold else "steady"
    return {"direction": direction, "change": change}


class WearableReader:
    def __init__(self, port=None):
        self.port = port
        self.serial_number = None
        self.lock = Lock()
        self.thread = None
        self.sample = None
        self.sample_id = 0
        self.last_received = None
        self.last_sample = None
        self.status = "disabled" if port == "off" else "disconnected"
        self.estimator = PulseEstimator()
        self.samples_since_estimate = 0
        self.waveform_at = None
        self.stream_started = None   # first line of the current connection
        self.led = None
        self.pulse_detail = None
        self.history = {name: deque(maxlen=HISTORY_SECONDS) for name in SIGNALS}
        self.latest = {}
        self.sessions = OrderedDict()

    # ------------------------------------------------------------------ input
    def ingest(self, line, now=None):
        """Process one line from the board (telemetry or waveform)."""
        now = monotonic() if now is None else now
        if line.startswith("PPG "):
            self._ingest_waveform(line, now)
            return
        sample = parse_telemetry(line)
        if sample is None:
            return
        self.estimator.add_motion(sample["motion"])
        if sample["led"] is not None:
            if self.led is not None and sample["led"] != self.led:
                self.estimator.restart_window()  # the light level steps when the LED current changes
            self.led = sample["led"]
        with self.lock:
            self.sample_id += 1
            self.sample = self.last_sample = {"id": self.sample_id, **sample}
            self.last_received = now
            self.status = "live"
            if self.stream_started is None:
                self.stream_started = now
            # Until the first PPG line, wait WAVEFORM_TIMEOUT before trusting the firmware's own
            # pulse and SpO2: telemetry arrives first after connecting, and the firmware SpO2
            # was 89-96 % there while the waveform estimate and a hospital oximeter gave 98-100 %.
            reference = self.waveform_at if self.waveform_at is not None else self.stream_started
            if now - reference > WAVEFORM_TIMEOUT:
                # Firmware without the waveform stream: use its own pulse estimate.
                self._record("pulse", now, sample["hr"])
                self._record("spo2", now, sample["spo2"])
            self._record("temperature", now, sample["temperature_c"])
            self._record("gsr", now, sample["gsr_mv"])
            self._record("gsr_adc", now, sample["gsr_adc"])
            self._record("motion", now, sample["motion"])

    def _ingest_waveform(self, line, now):
        waveform = parse_waveform(line)
        if waveform is None:
            return
        rate, first, pairs = waveform
        self.estimator.add_samples(first, pairs, rate)
        self.samples_since_estimate += len(pairs)
        with self.lock:
            self.waveform_at = now
        if self.samples_since_estimate < self.estimator.sample_rate or not self.estimator.window_ready():
            return
        self.samples_since_estimate = 0
        estimate = self.estimator.estimate(now)
        detail = self.estimator.last
        with self.lock:
            self.pulse_detail = detail
            if detail and detail["accepted"] and estimate is not None:
                self._record("pulse", now, estimate.bpm)
                self._record("spo2", now, estimate.spo2)

    def _record(self, name, now, value):
        """Keep the latest value and one value per second for trends. Lock must be held."""
        if value is None or not isfinite(value):
            return
        self.latest[name] = (now, value)
        history = self.history[name]
        if not history or now - history[-1][0] >= 1.0:
            history.append((now, value))

    # ------------------------------------------------------------------ output
    def snapshot(self, session_id=None):
        with self.lock:
            if self.thread is None and self.port != "off":
                self.thread = Thread(target=self._read, name="samvya-usb-wearable", daemon=True)
                self.thread.start()
            now = monotonic()
            status = self.status
            sample = self.sample
            if status == "live" and now - self.last_received >= FRESH_SECONDS:
                status, sample = "stale", None
            result = {
                "status": status,
                "sample": sample,
                # The last reading received, shown (with its age) when the device pauses.
                "last_sample": None if sample is not None else self.last_sample,
                "last_sample_age_s": (
                    int(now - self.last_received) if sample is None and self.last_received is not None else None
                ),
                "vitals": self._vitals(now, status, sample),
                "pulse_detail": self.pulse_detail,  # latest window: rate, quality, stillness, channel agreement
                "insights": None,
                "summary": None,
            }
            if session_id:
                self.sessions.setdefault(session_id, now)
                self.sessions.move_to_end(session_id)
                if len(self.sessions) > 128:
                    self.sessions.popitem(last=False)
                result["insights"] = self._insights(session_id, now)
                result["summary"] = self._summary(session_id, now)
            return result

    def summary(self, session_id):
        with self.lock:
            if session_id not in self.sessions:
                return None
            return self._summary(session_id, monotonic())

    def _vital(self, name, now, status):
        entry = self.latest.get(name)
        if entry is None or now - entry[0] > HOLD_SECONDS:
            # Nothing measured yet, or the last value is too old to describe the wearer now.
            return {"value": None, "state": "waiting", "age_s": None}
        age = now - entry[0]
        live = status == "live" and age <= LIVE_SECONDS
        return {"value": entry[1], "state": "live" if live else "held", "age_s": int(age)}

    def _vitals(self, now, status, sample):
        vitals = {name: self._vital(name, now, status) for name in ("pulse", "spo2", "temperature", "gsr", "motion")}
        detail = self.pulse_detail or {}
        pulse = vitals["pulse"]
        if pulse["state"] != "live":
            if status != "live":
                pulse["reason"] = "device"
            elif (sample and sample["pulse_status"] == "NO_FINGER") or detail.get("contact") is False:
                pulse["reason"] = "contact"
            elif detail.get("still") is False or (sample and (sample["motion"] or 0) > STILL):
                pulse["reason"] = "movement"
            else:
                pulse["reason"] = "measuring"
        oxygen = vitals["spo2"]
        correlation = detail.get("red_ir_correlation")
        if oxygen["state"] == "waiting" and correlation is not None and correlation < 0.9:
            # The red and IR pulses do not move together here, so the ratio has no
            # oxygen meaning (see pulse_estimator.py). This happens away from a fingertip.
            oxygen["state"] = "unsupported"
        temperature = vitals["temperature"]
        if temperature["state"] == "waiting" and sample and sample["temperature_sensor"] == "missing":
            temperature["state"] = "sensor_missing"
        return vitals

    def _since(self, name, start):
        return [(when, value) for when, value in self.history[name] if when >= start]

    def _insights(self, session_id, now):
        start = self.sessions[session_id]
        insights = {"elapsed_s": int(now - start)}
        seconds = len(self._since("gsr", start))

        pulse = self._since("pulse", start)
        if pulse:
            values = [value for _, value in pulse]
            baseline = _baseline(pulse)
            trend = _trend(pulse, now, 4.0)
            insights["pulse"] = {
                "average": round(mean(values), 1), "min": round(min(values), 1), "max": round(max(values), 1),
                "baseline": round(baseline, 1), "change": round(values[-1] - baseline, 1),
                "trend": trend and {"direction": trend["direction"], "change": round(trend["change"], 1)},
                "coverage": round(100 * len(pulse) / max(seconds, len(pulse))),
            }
        temperature = self._since("temperature", start)
        if temperature:
            baseline = _baseline(temperature)
            trend = _trend(temperature, now, 0.2)
            insights["temperature"] = {
                "current": round(temperature[-1][1], 2), "baseline": round(baseline, 2),
                "change": round(temperature[-1][1] - baseline, 2),
                "trend": trend and {"direction": trend["direction"], "change": round(trend["change"], 2)},
            }
        gsr = self._since("gsr", start)
        if gsr:
            baseline = _baseline(gsr)
            recent = [value for when, value in gsr if when >= now - TREND_RECENT]
            trend = _trend(gsr, now, CONDUCTANCE_CHANGE * baseline)
            change = (gsr[-1][1] - baseline) / baseline if baseline else None
            # The voltage falls as skin conductance rises: probes off the skin read the
            # highest voltage (about 2140 mV on this board), sweat-gland activity lowers it.
            insights["gsr"] = {
                "current": round(gsr[-1][1]), "baseline": round(baseline),
                "change_percent": None if change is None else round(100 * change, 1),
                "variability": round(pstdev(recent), 1) if len(recent) >= 10 else None,
                "trend": trend and {"direction": trend["direction"], "change": round(trend["change"])},
                "skin_conductance_vs_start": None if change is None else (
                    "higher" if change <= -CONDUCTANCE_CHANGE else "lower" if change >= CONDUCTANCE_CHANGE else "similar"),
                "skin_conductance_trend": trend and {"rising": "falling", "falling": "rising"}.get(
                    trend["direction"], "steady"),
            }
        movement = self._since("motion", start)
        if movement:
            values = [value for _, value in movement]
            insights["movement"] = {
                "still_percent": round(100 * sum(value < STILL for value in values) / len(values)),
                "active_percent": round(100 * sum(value >= ACTIVE for value in values) / len(values)),
            }
        return insights

    def _summary(self, session_id, now):
        """Check-in summary saved with the report. None if no readings were received."""
        insights = self._insights(session_id, now)
        start = self.sessions[session_id]
        seconds = len(self._since("gsr", start))
        oxygen = [value for _, value in self._since("spo2", start)]
        pulse = insights.get("pulse")
        if not seconds and not pulse:
            return None
        summary = {
            "source": "usb",
            "sample_count": seconds,
            "average_hr": pulse["average"] if pulse else None,
            "min_hr": pulse["min"] if pulse else None,
            "max_hr": pulse["max"] if pulse else None,
            "pulse_coverage": pulse["coverage"] if pulse else 0,
            "average_spo2": round(mean(oxygen), 1) if oxygen else None,
            "trends": insights,
        }
        last = self.last_sample or {}
        for key in ("temperature_c", "gsr_adc", "gsr_mv", "ir", "accel", "gyro", "motion"):
            if last.get(key) is not None:
                summary[key] = last[key]
        if "temperature_c" not in summary and "temperature" in self.latest:
            summary["temperature_c"] = self.latest["temperature"][1]
        return summary

    # ------------------------------------------------------------------ serial port
    def _select_port(self):
        if self.port:
            return self.port
        ports = [port for port in list_ports.comports() if (port.vid, port.pid) == (0x303A, 0x1001)]
        if self.serial_number:
            ports = [port for port in ports if port.serial_number == self.serial_number]
        if len(ports) != 1:
            return None
        self.serial_number = ports[0].serial_number
        return ports[0].device

    def _clear_sample(self, status):
        with self.lock:
            self.status = status
            self.sample = None

    def _read(self):
        while True:
            try:
                path = self._select_port()
                if path and (os.name != "posix" or os.path.exists(path)):
                    self._read_port(path)
                else:
                    self._clear_sample("disconnected")
            except OSError:
                self._clear_sample("waiting")
            sleep(1)

    def _read_port(self, path):
        self._clear_sample("waiting")
        connection = serial.Serial(baudrate=SERIAL_RATE, timeout=0.5, exclusive=True if os.name == "posix" else None)
        connection.port = path
        connection.rts = False
        with connection:
            connection.reset_input_buffer()
            self.estimator.restart_window()
            with self.lock:
                # A new connection: learn again whether this firmware streams the waveform.
                self.waveform_at = self.stream_started = None
            pending = b""
            while True:
                # Also check the path after a timed-out read so unplugged ports reopen cleanly.
                chunk = connection.read_until(size=1024)
                if not chunk and os.name == "posix" and not os.path.exists(path):
                    self._clear_sample("disconnected")
                    return
                pending += chunk
                if not pending.endswith(b"\n"):
                    if len(pending) >= 1024:
                        pending = b""
                    continue
                self.ingest(pending.decode("ascii", errors="replace").strip())
                pending = b""
