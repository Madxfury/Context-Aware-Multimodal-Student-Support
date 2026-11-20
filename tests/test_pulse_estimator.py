"""Pulse estimator checks on synthetic waveforms with known answers (no hardware needed)."""
import numpy as np
import pytest

from services.pulse_estimator import PulseEstimator, spo2_from_ratio

RATE = 99.76  # the LY board's measured sample rate differs slightly from 100 Hz


def waveform(bpm, seconds=30.0, ratio=0.5, inverted_red=False, noise=8.0, interference=True, seed=1):
    """Hand-like PPG: 0.05% pulse with harmonics, breathing drift, noise, and an
    IR-only non-harmonic component like the one seen on the LY board."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    f = bpm / 60.0
    shape = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(4 * np.pi * f * t + 0.6) + 0.2 * np.sin(6 * np.pi * f * t + 1.1)
    ir_dc, red_dc = 90_000.0, 105_000.0
    ir_ac = 0.0005 * ir_dc / 1.4
    breathing = 30.0 * np.sin(2 * np.pi * 0.25 * t)
    ir = ir_dc - ir_ac * shape + breathing + rng.normal(0, noise, t.size)
    if interference:
        ir += 0.4 * ir_ac * np.sin(2 * np.pi * 1.33 * f * t)
    red_ac = ratio * ir_ac * red_dc / ir_dc
    red = red_dc + (1 if inverted_red else -1) * red_ac * shape + breathing + rng.normal(0, noise, t.size)
    return red, ir


def run(red, ir, movement=0.03, index_gap_at=None):
    estimator = PulseEstimator(RATE)
    result, index = None, 0
    for start in range(0, len(ir), 10):
        if index_gap_at is not None and start == index_gap_at:
            index += 1000
        pairs = list(zip(red[start:start + 10], ir[start:start + 10]))
        estimator.add_samples(index, pairs, RATE)
        index += len(pairs)
        estimator.add_motion(movement)
        if start and start % 100 == 0:
            result = estimator.estimate(start / RATE)
    return result, estimator


@pytest.mark.parametrize("bpm", [48, 60, 66, 72, 90, 110, 130, 150])
def test_rate_is_found_within_one_bpm(bpm):
    result, _ = run(*waveform(bpm))
    assert result is not None
    assert abs(result.bpm - bpm) <= 1.0


def test_a_rhythm_moving_the_channels_apart_is_not_a_pulse():
    # Blood moves both channels together. An inverted copy in red is interference.
    result, estimator = run(*waveform(66, inverted_red=True, interference=False))
    assert result is None
    assert estimator.last["light_interference_bpm"] == pytest.approx(66, abs=1)


def test_in_phase_channels_give_the_ratio_spo2():
    result, estimator = run(*waveform(72, ratio=0.5, interference=False))
    assert estimator.last["red_ir_correlation"] > 0.9
    assert result.spo2 == pytest.approx(spo2_from_ratio(0.5), abs=1.0)


def test_movement_blocks_new_readings():
    result, estimator = run(*waveform(72), movement=0.6)
    assert result is None
    assert estimator.last["still"] is False


def test_no_skin_contact_blocks_readings():
    red, ir = waveform(72)
    result, estimator = run(red - 100_000 + 3_000, ir - 90_000 + 3_000)
    assert result is None
    assert estimator.last["contact"] is False


def test_noise_alone_is_not_a_pulse():
    rng = np.random.default_rng(7)
    n = int(30 * RATE)
    result, _ = run(105_000 + rng.normal(0, 20, n), 90_000 + rng.normal(0, 20, n))
    assert result is None


def test_index_gap_restarts_the_window():
    red, ir = waveform(80, seconds=14)
    result, estimator = run(red, ir, index_gap_at=600)
    # 14 s of signal but only 8 s after the gap: not enough for a 10 s window.
    assert not estimator.window_ready()
    assert result is None


def test_readings_expire_without_new_accepted_windows():
    _, estimator = run(*waveform(70))
    last_time = estimator.accepted[-1][0]
    assert estimator.current(last_time + 5) is not None
    assert estimator.current(last_time + 25) is None


def test_continuity_rejects_a_single_jump_and_relocks_on_a_real_change():
    estimator = PulseEstimator(RATE)
    decisions = [estimator._continuous(t, bpm) for t, bpm in enumerate([45, 66, 66.3, 66.5, 45, 66.2])]
    # 45 alone cannot start a reading; 66 needs a second agreeing window; the later 45 is a jump.
    assert decisions == [False, False, True, True, False, True]
    later = [estimator._continuous(t, bpm) for t, bpm in enumerate([80, 80.5, 81, 81.2], start=10)]
    assert later == [False, False, True, True]  # three agreeing windows confirm the new rate


def test_spo2_needs_three_consistent_in_phase_windows():
    estimator = PulseEstimator(RATE)
    estimator.accepted.extend([(1.0, 70, 97.0, 0.4), (2.0, 70, None, 0.4), (3.0, 70, 97.5, 0.4)])
    assert estimator.current(3.0).spo2 is None
    estimator.accepted.clear()
    estimator.accepted.extend([(1.0, 70, 97.0, 0.4), (2.0, 70, 92.0, 0.4), (3.0, 70, 97.5, 0.4)])
    assert estimator.current(3.0).spo2 is None
    estimator.accepted.clear()
    estimator.accepted.extend([(1.0, 70, 97.0, 0.4), (2.0, 70, 96.0, 0.4), (3.0, 70, 97.5, 0.4)])
    assert estimator.current(3.0).spo2 == 97.0


def test_brief_movement_in_the_window_blocks_the_reading():
    red, ir = waveform(72)
    estimator = PulseEstimator(RATE)
    index = 0
    for start in range(0, len(ir), 10):
        estimator.add_samples(index, list(zip(red[start:start + 10], ir[start:start + 10])), RATE)
        index += 10
        estimator.add_motion(0.6 if start == (len(ir) // 10 - 30) * 10 else 0.03)  # one reading ~3 s before the end
    estimator.estimate(len(ir) / RATE)
    assert estimator.last["still"] is False


def test_room_light_flicker_in_red_is_ignored():
    """LY board on the hand: red dominated by aliased light flicker (harmonic family at
    66 BPM), IR carrying the real 85 BPM pulse plus weak flicker."""
    rng = np.random.default_rng(3)
    t = np.arange(int(30 * RATE)) / RATE
    pulse = np.sin(2 * np.pi * 85 / 60 * t) + 0.3 * np.sin(4 * np.pi * 85 / 60 * t + 0.5)
    flicker_f = 66 / 60
    flicker = sum(amplitude * np.sin(2 * np.pi * k * flicker_f * t + k) for k, amplitude in ((1, 1.0), (2, 1.2), (3, 0.6)))
    ir = 90_000 - 30 * pulse + 8 * flicker + rng.normal(0, 6, t.size)
    red = 105_000 - 3 * pulse + 40 * flicker + rng.normal(0, 6, t.size)
    result, estimator = run(red, ir)
    assert result is not None and abs(result.bpm - 85) <= 1.0
    assert estimator.last["light_interference_bpm"] == pytest.approx(66, abs=1)
