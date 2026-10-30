import numpy as np
import time

class SyntheticWearableSimulator:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        
        # State for random walk to ensure continuous transitions between calls
        self.current_gsr_adc = 2000.0
        self.current_eda = 2.0
        self.current_temp = 36.5
        
        self.last_timestamp = time.time() if seed is None else 1600000000.0
        
    def generate_window(self, scenario="baseline", duration_sec=60):
        """
        Generates a 60-second window of synthetic data.
        Returns both arbitrary hardware telemetry (IR, RED, GSR_ADC) 
        and WESAD-compatible signals (BVP, EDA, TEMP, ACC) with appropriate frequencies.
        """
        
        # Define scenario drift/means
        if scenario == "baseline":
            hr_bpm = 70.0
            eda_drift = -0.01  # Relaxing
            gsr_drift = 5.0    # Resistance increasing
            temp_target = 36.5
        elif scenario == "low_stress":
            hr_bpm = 85.0
            eda_drift = 0.05
            gsr_drift = -10.0
            temp_target = 36.7
        elif scenario == "moderate_stress":
            hr_bpm = 100.0
            eda_drift = 0.1
            gsr_drift = -20.0
            temp_target = 37.0
        elif scenario == "high_stress":
            hr_bpm = 120.0
            eda_drift = 0.3
            gsr_drift = -40.0
            temp_target = 37.5
        elif scenario == "recovery":
            hr_bpm = 80.0
            eda_drift = -0.1
            gsr_drift = 20.0
            temp_target = 36.6
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        # WESAD Frequencies
        freq_bvp = 64
        freq_eda = 4
        freq_temp = 4
        freq_acc = 32
        
        # Hardware Telemetry Frequencies (Arbitrary for demo)
        freq_hw = 50
        
        # Time arrays
        t_hw = np.linspace(self.last_timestamp, self.last_timestamp + duration_sec, duration_sec * freq_hw)
        t_bvp = np.linspace(self.last_timestamp, self.last_timestamp + duration_sec, duration_sec * freq_bvp)
        t_eda = np.linspace(self.last_timestamp, self.last_timestamp + duration_sec, duration_sec * freq_eda)
        t_temp = np.linspace(self.last_timestamp, self.last_timestamp + duration_sec, duration_sec * freq_temp)
        t_acc = np.linspace(self.last_timestamp, self.last_timestamp + duration_sec, duration_sec * freq_acc)
        
        self.last_timestamp += duration_sec
        
        # --- 1. WESAD-Compatible Signals (Model validation strictly for integration) ---
        # BVP (64Hz) - simulated pulse
        hr_hz = hr_bpm / 60.0
        bvp = 50.0 * np.sin(2 * np.pi * hr_hz * t_bvp) + self.rng.normal(0, 5, len(t_bvp))
        
        # EDA (4Hz) - random walk with drift
        eda_noise_mean = eda_drift / len(t_eda)
        eda_noise = self.rng.normal(eda_noise_mean, 0.05, len(t_eda))
        eda = self.current_eda + np.cumsum(eda_noise)
        eda = np.clip(eda, 0.1, 50.0)
        self.current_eda = eda[-1]
        
        # TEMP (4Hz) - gradual drift toward target
        temp_noise = self.rng.normal(0, 0.01, len(t_temp))
        temp_diff = (temp_target - self.current_temp) / (len(t_temp))
        temp = self.current_temp + np.cumsum(np.full(len(t_temp), temp_diff) + temp_noise)
        self.current_temp = temp[-1]
        
        # ACC (32Hz)
        acc_noise_scale = 0.5 if "stress" in scenario else 0.1
        acc = self.rng.normal(0, acc_noise_scale, (len(t_acc), 3))
        
        # --- 2. Raw Hardware Telemetry (ESP32-S3 Mock) ---
        ir = 100000.0 + 1000.0 * np.sin(2 * np.pi * hr_hz * t_hw) + self.rng.normal(0, 100, len(t_hw))
        red = 90000.0 + 900.0 * np.sin(2 * np.pi * hr_hz * t_hw) + self.rng.normal(0, 90, len(t_hw))
        
        gsr_noise_mean = gsr_drift / len(t_hw)
        gsr_noise = self.rng.normal(gsr_noise_mean, 5.0, len(t_hw))
        gsr_adc = self.current_gsr_adc + np.cumsum(gsr_noise)
        gsr_adc = np.clip(gsr_adc, 0, 4095)
        self.current_gsr_adc = gsr_adc[-1]
        
        return {
            "wesad_compatible": {
                "bvp": bvp,
                "eda": eda,
                "temp": temp,
                "acc": acc
            },
            "hardware_telemetry": {
                "ir": ir,
                "red": red,
                "gsr_adc": gsr_adc,
                "temp_raw": temp, # Same underlying physics
                "acc_raw": acc    # Same underlying physics
            }
        }

def extract_synthetic_v2_sequence(wesad_signals):
    """
    Passes the WESAD-compatible synthetic signals through the exact V2 preprocessing logic.
    Extracts 14 statistical features for each 5-second sub-window inside the 60-second window.
    Returns: list of 12 elements, each a list of 14 features (shape: 12x14).
    """
    bvp = wesad_signals["bvp"]
    eda = wesad_signals["eda"]
    temp = wesad_signals["temp"]
    acc = wesad_signals["acc"]
    
    # 60s window / 5s sub-window = 12 sub-windows
    sub_windows = 12
    bvp_len = len(bvp) // sub_windows # 320
    eda_len = len(eda) // sub_windows # 20
    temp_len = len(temp) // sub_windows # 20
    acc_len = len(acc) // sub_windows # 160
    
    sequence = []
    
    for i in range(sub_windows):
        b_sub = bvp[i*bvp_len : (i+1)*bvp_len]
        e_sub = eda[i*eda_len : (i+1)*eda_len]
        t_sub = temp[i*temp_len : (i+1)*temp_len]
        a_sub = acc[i*acc_len : (i+1)*acc_len]
        
        # Exact V2 Extraction
        bvp_mean = np.mean(b_sub)
        bvp_std = np.std(b_sub)
        bvp_range = np.max(b_sub) - np.min(b_sub) if len(b_sub) > 0 else 0
        bvp_energy = np.sum(b_sub**2)
        eda_mean = np.mean(e_sub)
        eda_std = np.std(e_sub)
        eda_range = np.max(e_sub) - np.min(e_sub) if len(e_sub) > 0 else 0
        eda_slope = e_sub[-1] - e_sub[0] if len(e_sub) > 1 else 0
        temp_mean = np.mean(t_sub)
        temp_std = np.std(t_sub)
        temp_range = np.max(t_sub) - np.min(t_sub) if len(t_sub) > 0 else 0
        temp_slope = t_sub[-1] - t_sub[0] if len(t_sub) > 1 else 0
        
        acc_mag = np.sqrt(np.sum(a_sub**2, axis=1))
        acc_mean = np.mean(acc_mag)
        acc_std = np.std(acc_mag)
        
        feats = [float(bvp_mean), float(bvp_std), float(bvp_range), float(bvp_energy),
                 float(eda_mean), float(eda_std), float(eda_range), float(eda_slope),
                 float(temp_mean), float(temp_std), float(temp_range), float(temp_slope),
                 float(acc_mean), float(acc_std)]
        sequence.append(feats)
        
    return sequence
