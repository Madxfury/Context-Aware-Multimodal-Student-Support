import os
import pickle
import numpy as np
from glob import glob

DATA_DIR = "DataSet/WESAD/WESAD"

def inspect_wesad():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory {DATA_DIR} not found.")
        print("Please download WESAD and extract it here.")
        return

    subject_dirs = sorted(glob(os.path.join(DATA_DIR, "S*")))
    subjects = [os.path.basename(d) for d in subject_dirs if os.path.isdir(d)]
    
    print("=== WESAD Dataset Inspection ===")
    print(f"Total Subjects Found: {len(subjects)}")
    print(f"Subjects: {subjects}")
    print("-" * 30)

    if not subjects:
        print("No subject folders found. Did you extract it correctly?")
        return

    # Inspect first subject
    sample_subject = subjects[0]
    sample_file = os.path.join(DATA_DIR, sample_subject, f"{sample_subject}.pkl")
    
    if not os.path.exists(sample_file):
        print(f"Expected pickle file not found: {sample_file}")
        return
        
    print(f"Inspecting sample subject: {sample_subject}")
    
    try:
        # WESAD pickle files were created in Python 2, need latin1 encoding
        with open(sample_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
        print("\nAvailable Devices & Signals:")
        for device, signals in data['signal'].items():
            print(f"  [{device.upper()}]")
            for sig_name, sig_data in signals.items():
                print(f"    - {sig_name}: shape {sig_data.shape}, dtype {sig_data.dtype}")
                
        print("\nLabels Distribution:")
        labels = data['label']
        print(f"  Shape: {labels.shape}")
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_map = {
            0: 'Transient/Not defined',
            1: 'Baseline',
            2: 'Stress',
            3: 'Amusement',
            4: 'Meditation',
            5: 'Ignore/Recovery' # Varies by paper, usually 5-7 are recovery/ignore
        }
        for lbl, count in zip(unique_labels, counts):
            name = label_map.get(lbl, "Unknown")
            print(f"  Label {lbl} ({name}): {count} samples")

        print("\nSampling Rates (Standard WESAD):")
        print("  Chest: 700 Hz (ECG, EDA, EMG, Resp, Temp)")
        print("  Wrist: 64 Hz (BVP), 32 Hz (ACC), 4 Hz (EDA, Temp)")
        print("\nInspection Complete. Ready for preprocessing pipeline.")

    except Exception as e:
        print(f"Failed to load pickle file: {e}")

if __name__ == "__main__":
    inspect_wesad()
