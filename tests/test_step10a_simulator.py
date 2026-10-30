import unittest
import numpy as np
from services.synthetic_simulator import SyntheticWearableSimulator, extract_synthetic_v2_sequence

class TestSyntheticWearableSimulator(unittest.TestCase):
    def test_sensor_fields_and_shapes(self):
        sim = SyntheticWearableSimulator(seed=42)
        window = sim.generate_window("baseline", duration_sec=60)
        
        self.assertIn("wesad_compatible", window)
        self.assertIn("hardware_telemetry", window)
        
        wesad = window["wesad_compatible"]
        hw = window["hardware_telemetry"]
        
        # WESAD fields
        self.assertIn("bvp", wesad)
        self.assertIn("eda", wesad)
        self.assertIn("temp", wesad)
        self.assertIn("acc", wesad)
        
        # Hardware fields
        self.assertIn("ir", hw)
        self.assertIn("red", hw)
        self.assertIn("gsr_adc", hw)
        self.assertIn("temp_raw", hw)
        self.assertIn("acc_raw", hw)
        
        # Sampling structure
        self.assertEqual(len(wesad["bvp"]), 64 * 60)
        self.assertEqual(len(wesad["eda"]), 4 * 60)
        self.assertEqual(len(wesad["temp"]), 4 * 60)
        self.assertEqual(wesad["acc"].shape, (32 * 60, 3))
        
        self.assertEqual(len(hw["ir"]), 50 * 60)
        self.assertEqual(len(hw["red"]), 50 * 60)
        self.assertEqual(len(hw["gsr_adc"]), 50 * 60)
        
    def test_no_nans(self):
        sim = SyntheticWearableSimulator(seed=42)
        window = sim.generate_window("moderate_stress", duration_sec=60)
        
        for k, v in window["wesad_compatible"].items():
            self.assertFalse(np.isnan(v).any(), f"NaNs found in {k}")
            
        for k, v in window["hardware_telemetry"].items():
            self.assertFalse(np.isnan(v).any(), f"NaNs found in {k}")
            
    def test_deterministic_seed(self):
        sim1 = SyntheticWearableSimulator(seed=123)
        win1 = sim1.generate_window("low_stress", duration_sec=60)
        
        sim2 = SyntheticWearableSimulator(seed=123)
        win2 = sim2.generate_window("low_stress", duration_sec=60)
        
        np.testing.assert_array_equal(win1["wesad_compatible"]["bvp"], win2["wesad_compatible"]["bvp"])
        np.testing.assert_array_equal(win1["hardware_telemetry"]["ir"], win2["hardware_telemetry"]["ir"])
        
    def test_scenario_distributions(self):
        sim = SyntheticWearableSimulator(seed=42)
        
        # Generate baseline
        win_base = sim.generate_window("baseline", duration_sec=60)
        bvp_base_std = np.std(win_base["wesad_compatible"]["bvp"])
        
        # We need a new simulator to avoid random walk accumulation
        sim_stress = SyntheticWearableSimulator(seed=42)
        win_stress = sim_stress.generate_window("high_stress", duration_sec=60)
        
        # High stress has a higher heart rate frequency, which might change variance or frequency spectrum.
        # Temp should be higher.
        self.assertGreater(
            np.mean(win_stress["wesad_compatible"]["temp"]),
            np.mean(win_base["wesad_compatible"]["temp"])
        )
        
        # Accelerometer should have more variance
        self.assertGreater(
            np.var(win_stress["wesad_compatible"]["acc"]),
            np.var(win_base["wesad_compatible"]["acc"])
        )
        
        # EDA should drift higher
        self.assertGreater(
            np.mean(win_stress["wesad_compatible"]["eda"]),
            np.mean(win_base["wesad_compatible"]["eda"])
        )

    def test_timestamps_increase_correctly(self):
        sim = SyntheticWearableSimulator(seed=42)
        start_time = sim.last_timestamp
        sim.generate_window("baseline", duration_sec=60)
        self.assertEqual(sim.last_timestamp, start_time + 60)
        
    def test_v2_sequence_extraction(self):
        sim = SyntheticWearableSimulator(seed=42)
        window = sim.generate_window("baseline", duration_sec=60)
        
        v2_seq = extract_synthetic_v2_sequence(window["wesad_compatible"])
        
        # Should be a list of 12 sub-windows, each containing 14 features
        self.assertEqual(len(v2_seq), 12)
        self.assertEqual(len(v2_seq[0]), 14)
        
        # No NaNs in extracted features
        for feats in v2_seq:
            self.assertFalse(np.isnan(feats).any())

if __name__ == '__main__':
    unittest.main()
