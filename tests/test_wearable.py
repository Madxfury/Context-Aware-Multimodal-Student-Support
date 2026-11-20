from time import monotonic

import pytest
from types import SimpleNamespace
from unittest.mock import Mock

import app as app_module
from services.db_manager import DatabaseManager
from services.wearable import WearableReader, parse_telemetry


READING = (
    "TIME=50258 ms | HR=73.2 BPM | INST=74.0 | STATUS=VALID | "
    "IR=58230 | IR_MEAN=59000 | IR_SD=1024 | LED=0x1F | TEMP=32.16 C | "
    "GSR=2463 ADC | 2135 mV | ACC=-1.62,1.36,8.40 | GYRO=-0.01,0.00,0.01"
)


def firmware_only_reader():
    """A reader that has had telemetry without PPG lines for an hour (firmware without the
    waveform stream), so the firmware's own pulse and SpO2 are used."""
    reader = WearableReader(port="off")
    reader.stream_started = monotonic() - 3600
    return reader


def test_reads_the_firmware_protocol_and_hides_invalid_channels():
    sample = parse_telemetry(READING)
    assert sample["hr"] == 73.2
    assert sample["temperature_c"] == 32.16
    assert sample["gsr_mv"] == 2135
    assert sample["accel"] == [-1.62, 1.36, 9.59]  # Z offset of the LY board's MPU6050 removed
    assert sample["gyro"] == [-0.01, 0.0, 0.01]

    no_contact = parse_telemetry(READING.replace("STATUS=VALID", "STATUS=NO_FINGER").replace("32.16 C", "---"))
    assert no_contact["hr"] is None
    assert no_contact["temperature_c"] is None
    assert no_contact["gsr_adc"] == 2463
    assert parse_telemetry("Initializing MAX30102...") is None
    assert parse_telemetry("TIME=1000 ms | HR=72 BPM") is None


READING_V2 = (
    "TIME=91234 ms | HR=72.4 BPM | INST=71.9 | STATUS=VALID | PULSE_Q=0.91 | IR=104233 | IR_MEAN=104100 | "
    "IR_SD=310 | LED=0x2A | RED=98321 | SPO2=97.8 % | SPO2_RATIO=0.512 | SPO2_FIT=0.94 | "
    "TEMP=--- | TEMP_SENSOR=MISSING | "
    "GSR=2425 ADC | 2104 mV | ACC=-0.04,-1.37,8.59 | GYRO=0.00,-0.00,0.00 | MOTION=0.02 m/s2"
)


def test_reads_firmware_v2_fields_and_keeps_v1_compatible():
    sample = parse_telemetry(READING_V2)
    assert sample["hr"] == 72.4
    assert sample["spo2"] == 97.8
    assert sample["motion"] == 0.02
    assert sample["temperature_c"] is None
    assert sample["temperature_sensor"] == "missing"
    assert sample["ir"] == 104233
    assert sample["gsr_mv"] == 2104
    assert sample["gyro"] == [0.0, 0.0, 0.0]

    detecting = parse_telemetry(READING_V2.replace("STATUS=VALID", "STATUS=DETECTING"))
    assert detecting["hr"] is None and detecting["spo2"] is None
    assert parse_telemetry(READING_V2.replace("SPO2=97.8 %", "SPO2=---"))["spo2"] is None
    assert parse_telemetry(READING_V2.replace("SPO2=97.8 %", "SPO2=55.0 %"))["spo2"] is None
    assert parse_telemetry(READING_V2.replace("TEMP_SENSOR=MISSING", "TEMP_SENSOR=OK"))["temperature_sensor"] == "connected"
    no_imu = parse_telemetry(READING_V2.replace("ACC=-0.04,-1.37,8.59", "ACC=---").replace(
        "GYRO=0.00,-0.00,0.00", "GYRO=---").replace("MOTION=0.02 m/s2", "MOTION=---"))
    assert no_imu["accel"] is None and no_imu["gyro"] is None and no_imu["motion"] is None

    v1 = parse_telemetry(READING)
    assert v1["spo2"] is None and v1["motion"] is None and v1["temperature_sensor"] is None


def test_guardrail_allows_reported_readings_but_not_judgements():
    guardrail = app_module.llm_guardrail
    assert guardrail.check_output("Your heart rate is 72 BPM and the SpO2 estimate is 98%.").allowed
    assert guardrail.check_output("Your pulse is 64 BPM right now.").allowed
    assert not guardrail.check_output("Your heart rate is very high.").allowed
    assert not guardrail.check_output("Your pulse seems a bit fast today.").allowed
    assert not guardrail.check_output("Your oxygen level is dangerously low.").allowed
    # GSR may be described as skin conductance, not interpreted as a mental state.
    assert guardrail.check_output("GSR is 1221 mV, 6.7% lower than at the start, so skin conductance is higher.").allowed
    assert guardrail.check_output("Skin conductance is higher. Are you feeling stressed about exams?").allowed
    assert not guardrail.check_output("This lower voltage suggests your skin conductance is higher now – often a "
                                      "sign of increased physiological arousal or stress.").allowed
    assert not guardrail.check_output("Your GSR shows you are anxious.").allowed


def test_contact_filter_keeps_readings_written_in_words():
    from services.india_support import without_generated_contacts

    for reading in (
        "Your heart rate is 84.8 beats per minute.",
        "Oxygen is about 99 percent.",
        "The sensor reads 33.1 degrees Celsius.",
        "GSR is 1257 millivolts.",
        "Heart rate: 84.8 BPM, SpO2 99.1%.",
    ):
        assert without_generated_contacts(reading) == reading
    for contact in ("Call 98 765 43210 for help.", "Dial 14416 now.", "Call 14416 percent of the time."):
        assert without_generated_contacts(contact) == ""


def reading_line(hr=73.2, temp="32.16 C", gsr_mv=2135, motion="0.02 m/s2", status="VALID"):
    return (
        f"TIME=50258 ms | HR={hr} BPM | INST=74.0 | STATUS={status} | PULSE_Q=0.9 | IR=58230 | IR_MEAN=59000 | "
        f"IR_SD=1024 | LED=0x1F | RED=52000 | SPO2=--- | SPO2_RATIO=--- | SPO2_FIT=--- | TEMP={temp} | "
        f"TEMP_SENSOR=OK | GSR=2463 ADC | {gsr_mv} mV | ACC=-1.62,1.36,8.40 | GYRO=-0.01,0.00,0.01 | MOTION={motion}"
    )


def test_readings_are_held_with_their_age_when_the_device_pauses():
    reader = firmware_only_reader()
    reader.ingest(READING, now=monotonic() - 10)
    snapshot = reader.snapshot("session")
    assert snapshot["status"] == "stale" and snapshot["sample"] is None  # older than FRESH_SECONDS
    assert snapshot["last_sample"]["gsr_adc"] == 2463 and snapshot["last_sample_age_s"] >= 9
    pulse = snapshot["vitals"]["pulse"]
    assert pulse == {"value": 73.2, "state": "held", "age_s": pulse["age_s"], "reason": "device"}
    assert 9 <= pulse["age_s"] <= 11
    assert snapshot["vitals"]["temperature"]["value"] == 32.16
    assert reader.summary("new-session") is None


def test_held_readings_expire_after_ten_minutes():
    reader = firmware_only_reader()
    reader.ingest(reading_line(), now=monotonic() - 540)
    assert reader.snapshot()["vitals"]["pulse"]["state"] == "held"  # 9 minutes old: still shown with its age
    reader = firmware_only_reader()
    reader.ingest(reading_line(), now=monotonic() - 700)
    vitals = reader.snapshot()["vitals"]
    # An 11-minute-old value no longer describes the wearer (e.g. a reading from an earlier sitting).
    assert vitals["pulse"] == {"value": None, "state": "waiting", "age_s": None, "reason": "device"}
    assert vitals["temperature"]["value"] is None and vitals["gsr"]["value"] is None


def test_accelerometer_zero_g_offset_is_removed():
    from services.wearable import accel_offset
    assert accel_offset("0,0,-1.19") == [0.0, 0.0, -1.19]
    assert accel_offset("0.1, 0, 0") == [0.1, 0.0, 0.0]
    assert accel_offset("nonsense") == [0.0, 0.0, -1.19] and accel_offset("0,0,99") == [0.0, 0.0, -1.19]
    # Palm-down and palm-up readings recorded on the LY board: raw |g| 10.95 and 8.87 m/s^2.
    for raw in ("0.12,3.16,-10.49", "0.49,5.44,6.99"):
        sample = parse_telemetry(READING.replace("ACC=-1.62,1.36,8.40", "ACC=" + raw))
        assert sum(value * value for value in sample["accel"]) ** 0.5 == pytest.approx(9.79, abs=0.07)


def test_check_in_trends_compare_with_the_first_minute():
    reader = firmware_only_reader()
    start = monotonic() - 200
    reader.snapshot("trend")
    reader.sessions["trend"] = start
    for second in range(200):
        hr = 70.0 if second < 140 else 80.0
        temp = 33.0 + 0.004 * second
        motion = "0.03 m/s2" if second % 10 else "1.20 m/s2"
        reader.ingest(reading_line(hr=hr, temp=f"{temp:.2f} C", gsr_mv=1400 - second, motion=motion), now=start + second)
    insights = reader.snapshot("trend")["insights"]
    assert insights["pulse"]["baseline"] == 70.0 and insights["pulse"]["change"] == 10.0
    assert insights["pulse"]["trend"]["direction"] == "rising" and insights["pulse"]["coverage"] == 100
    assert insights["temperature"]["trend"]["direction"] == "rising"
    assert insights["temperature"]["change"] == pytest.approx(0.67, abs=0.05)
    assert insights["gsr"]["trend"]["direction"] == "falling" and insights["gsr"]["change_percent"] < -10
    # A falling GSR voltage means rising skin conductance (probes off the skin read the highest voltage).
    assert insights["gsr"]["skin_conductance_vs_start"] == "higher"
    assert insights["gsr"]["skin_conductance_trend"] == "rising"
    assert insights["movement"]["still_percent"] == 90 and insights["movement"]["active_percent"] == 10
    summary = reader.summary("trend")
    assert summary["average_hr"] == 73.0 and summary["sample_count"] == 200
    assert summary["min_hr"] == 70.0 and summary["max_hr"] == 80.0 and summary["gsr_mv"] == 1201


def test_waveform_pulse_replaces_the_firmware_pulse():
    from services.pulse_estimator import spo2_from_ratio
    from tests.test_pulse_estimator import RATE, waveform

    reader = WearableReader(port="off")
    red, ir = waveform(66, seconds=16, interference=False)
    start = monotonic() - 16
    index = 0
    for position in range(0, len(ir), 10):
        now = start + position / RATE
        values = " ".join(f"{int(r)} {int(i)}" for r, i in zip(red[position:position + 10], ir[position:position + 10]))
        reader.ingest(f"PPG {RATE:.2f} {index} {values}", now=now)
        index += 10
        if position % 10 == 0:
            reader.ingest(reading_line(hr=73.2, motion="0.03 m/s2"), now=now)
    vitals = reader.snapshot()["vitals"]
    assert vitals["pulse"]["state"] == "live"
    assert abs(vitals["pulse"]["value"] - 66) <= 1.0  # the firmware said 73.2
    assert vitals["spo2"]["value"] == pytest.approx(spo2_from_ratio(0.5), abs=1.0)  # in phase, as on a fingertip


def test_firmware_spo2_is_not_shown_while_the_waveform_stream_starts():
    # After connecting, telemetry can arrive before the first PPG line. The LY firmware's own
    # SpO2 read 89.4 % there while the waveform estimate and a hospital oximeter gave 98-100 %.
    start = monotonic() - 20
    line = reading_line(hr=85.3).replace("SPO2=---", "SPO2=89.4 %")
    reader = WearableReader(port="off")
    for second in range(20):
        reader.ingest(line, now=start + second)
        reader.ingest(f"PPG 99.71 {157348 + 2 * second} 66629 149794 66620 149785", now=start + second + 0.1)
    assert reader.latest.get("spo2") is None and reader.latest.get("pulse") is None
    # Firmware that never sends PPG lines: its own values are used after WAVEFORM_TIMEOUT.
    reader = WearableReader(port="off")
    for second in range(20):
        reader.ingest(line, now=start + second)
    assert reader.history["spo2"][0][0] == start + 6 and reader.latest["spo2"][1] == 89.4


def test_parses_the_waveform_stream():
    from services.wearable import parse_waveform

    assert parse_waveform("PPG 99.89 2190 28398 83822 28393 83846") == (99.89, 2190, [(28398, 83822), (28393, 83846)])
    assert parse_waveform("PPG 99.89 2190 28398") is None
    assert parse_waveform("PPG 99.89 2190 28398 999999") is None
    assert parse_waveform("TIME=1 ms") is None
    # A line recorded from the LY board with bytes lost on USB: one IR sample reads 86.
    garbled = ("PPG 99.73 154039 107210 85382 107193 85387 107207 85387 107180 86 107088 85230 107059 85223 "
               "107079 85224 107072 85206 107068 85212 107069 85210 107072 85213")
    assert parse_waveform(garbled) is None
    assert parse_waveform(garbled.replace(" 86 ", " 85386 ")) is not None


def test_chat_and_report_use_server_observed_values(tmp_path, monkeypatch):
    reader = firmware_only_reader()
    monkeypatch.setattr(app_module, "wearable_reader", reader)
    monkeypatch.setattr(app_module, "db_manager", DatabaseManager(str(tmp_path / "wearable.db")))
    monkeypatch.setenv("GROQ_API_KEY", "test")
    provider = Mock()
    reply = "The wearable reports 73.2 BPM, 32.16 °C, and 2463 ADC counts."
    provider.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
    )
    monkeypatch.setattr(app_module, "Groq", lambda **kwargs: provider)
    client = app_module.app.test_client()

    client.get("/api/wearable?session_id=wearable_test")
    reader.ingest(READING)
    sample = client.get("/api/wearable?session_id=wearable_test").json["sample"]
    assert sample["gsr_adc"] == 2463
    response = client.post("/chat", json={
        "message": "What is the wearable showing?",
        "session_id": "wearable_test",
        "sensors": {"hr": 99.99, "spo2": 99.99, "v2_sequence": [[42]]},
        "defer_audio": True,
    })
    assert response.json["response"] == reply
    messages = provider.chat.completions.create.call_args.kwargs["messages"]
    assert '"pulse": {"bpm": 73.2, "unit": "BPM", "measured": "just now"}' in messages[0]["content"]
    assert '"adc_counts": 2463.0' in messages[0]["content"]
    assert "99.99" not in messages[0]["content"]

    saved = client.post("/api/sessions/save", json={"session_id": "wearable_test"})
    assert saved.json["wearable_summary"]["average_hr"] == 73.2
    assert saved.json["wearable_summary"]["sample_count"] == 1
    report = client.get("/session/wearable_test").get_data(as_text=True)
    assert "USB wearable" in report and "73.2 BPM" in report and "2463 ADC" in report
    assert "Simulated wearable" not in report


def test_chat_and_report_use_firmware_v2_readings(tmp_path, monkeypatch):
    reader = firmware_only_reader()
    monkeypatch.setattr(app_module, "wearable_reader", reader)
    monkeypatch.setattr(app_module, "db_manager", DatabaseManager(str(tmp_path / "wearable_v2.db")))
    monkeypatch.setenv("GROQ_API_KEY", "test")
    provider = Mock()
    reply = "Your heart rate is 72.4 BPM and the SpO2 estimate is 97.8%."
    provider.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
    )
    monkeypatch.setattr(app_module, "Groq", lambda **kwargs: provider)
    client = app_module.app.test_client()

    client.get("/api/wearable?session_id=wearable_v2")
    reader.ingest(READING_V2)
    response = client.post("/chat", json={
        "message": "What are my heart rate and oxygen?",
        "session_id": "wearable_v2",
        "defer_audio": True,
    })
    assert response.json["response"] == reply
    prompt = provider.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert '"spo2_estimate": {"percent": 97.8' in prompt
    assert '"m_per_s2": 0.02' in prompt and '"activity": "still"' in prompt
    assert '"skin_temperature": {"celsius": null, "status": "sensor_missing", "temperature_sensor": "missing"}' in prompt

    saved = client.post("/api/sessions/save", json={"session_id": "wearable_v2"})
    assert saved.json["wearable_summary"]["average_spo2"] == 97.8
    report = client.get("/session/wearable_v2").get_data(as_text=True)
    assert "97.8 %" in report and "0.02 m/s²" in report


def test_report_shows_check_in_trends(tmp_path, monkeypatch):
    reader = firmware_only_reader()
    start = monotonic() - 120
    reader.snapshot("trend_report")
    reader.sessions["trend_report"] = start
    for second in range(120):
        reader.ingest(reading_line(hr=70.0 + (second > 90) * 6, temp=f"{33 + second / 400:.2f} C"), now=start + second)
    db = DatabaseManager(str(tmp_path / "trend_report.db"))
    monkeypatch.setattr(app_module, "db_manager", db)
    db.save_session({"session_id": "trend_report", "transcript": [], "wearable_summary": reader.summary("trend_report")})
    report = app_module.app.test_client().get("/session/trend_report").get_data(as_text=True)
    assert "Pulse range" in report and "70–76 BPM, measured 100% of the time" in report
    assert "Skin temperature vs start" in report and "Hand still" in report
    assert "2 min 0 s of wearable data" in report and "Not measured at this sensor position" in report
    assert "Unavailable" not in report
