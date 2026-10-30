import os
import pickle
import numpy as np
import torch
import json

DATA_DIR = "DataSet/WESAD/WESAD"
OUT_DIR = "DataSet/WESAD_Processed_V2"

WINDOW_SEC = 60
STRIDE_SEC = 30
SUB_WINDOW_SEC = 5
NUM_SUB_WINDOWS = WINDOW_SEC // SUB_WINDOW_SEC  # 12

CALIBRATION_SEC = 300  # First 5 minutes of Baseline

SUBJECTS_TRAIN = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13"]
SUBJECTS_VAL = ["S14", "S15"]
SUBJECTS_TEST = ["S16", "S17"]

def get_native_signal_slice(signal, start_sec, end_sec, hz):
    start_idx = int(start_sec * hz)
    end_idx = int(end_sec * hz)
    return signal[start_idx:end_idx]

def extract_features(bvp, eda, temp, acc):
    # 1. BVP (4 features)
    bvp_mean = np.mean(bvp)
    bvp_std = np.std(bvp)
    bvp_range = np.max(bvp) - np.min(bvp) if len(bvp) > 0 else 0
    bvp_energy = np.sum(bvp**2)
    
    # 2. EDA (4 features)
    eda_mean = np.mean(eda)
    eda_std = np.std(eda)
    eda_range = np.max(eda) - np.min(eda) if len(eda) > 0 else 0
    eda_slope = eda[-1][0] - eda[0][0] if len(eda) > 1 else 0
    
    # 3. TEMP (4 features)
    temp_mean = np.mean(temp)
    temp_std = np.std(temp)
    temp_range = np.max(temp) - np.min(temp) if len(temp) > 0 else 0
    temp_slope = temp[-1][0] - temp[0][0] if len(temp) > 1 else 0
    
    # 4. ACC (2 features)
    if len(acc) > 0:
        acc_mag = np.sqrt(np.sum(acc**2, axis=1))
        acc_mean = np.mean(acc_mag)
        acc_std = np.std(acc_mag)
    else:
        acc_mean = 0
        acc_std = 0
        
    return [bvp_mean, bvp_std, bvp_range, bvp_energy,
            eda_mean, eda_std, eda_range, eda_slope,
            temp_mean, temp_std, temp_range, temp_slope,
            acc_mean, acc_std]

def process_subject(subject_path):
    print(f"Processing {os.path.basename(subject_path)}...")
    pkl_file = os.path.join(subject_path, f"{os.path.basename(subject_path)}.pkl")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
    labels = data['label']
    bvp = data['signal']['wrist']['BVP']
    eda = data['signal']['wrist']['EDA']
    temp = data['signal']['wrist']['TEMP']
    acc = data['signal']['wrist']['ACC']
    
    # Find contiguous blocks
    is_base = (labels == 1).astype(int)
    is_stress = (labels == 2).astype(int)
    
    def get_blocks(is_target):
        diff = np.diff(is_target, prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        return list(zip(starts, ends))
        
    blocks_class0 = get_blocks(is_base)
    blocks_class1 = get_blocks(is_stress)
    
    windows_features = []
    y = []
    calibration_features = []
    
    for cls, blocks in [(0, blocks_class0), (1, blocks_class1)]:
        for start_700, end_700 in blocks:
            start_sec = start_700 / 700.0
            end_sec = end_700 / 700.0
            
            # Sliding window of 60 seconds
            curr_sec = start_sec
            while curr_sec + WINDOW_SEC <= end_sec:
                win_feats = []
                # Break into 12 sub-windows
                for sub_i in range(NUM_SUB_WINDOWS):
                    sub_start = curr_sec + (sub_i * SUB_WINDOW_SEC)
                    sub_end = sub_start + SUB_WINDOW_SEC
                    
                    b_sub = get_native_signal_slice(bvp, sub_start, sub_end, 64)
                    e_sub = get_native_signal_slice(eda, sub_start, sub_end, 4)
                    t_sub = get_native_signal_slice(temp, sub_start, sub_end, 4)
                    a_sub = get_native_signal_slice(acc, sub_start, sub_end, 32)
                    
                    feats = extract_features(b_sub, e_sub, t_sub, a_sub)
                    win_feats.append(feats)
                    
                is_calibration = (cls == 0 and (curr_sec + WINDOW_SEC - start_sec) <= CALIBRATION_SEC)
                if is_calibration:
                    calibration_features.extend(win_feats)
                else:
                    windows_features.append(win_feats)
                    y.append(cls)
                    
                curr_sec += STRIDE_SEC
                
    if len(windows_features) == 0:
        return None, None
        
    # Shape: [N_windows, 12, 14]
    X = np.array(windows_features)
    Y = np.array(y)
    
    # Subject-specific baseline calibration
    # shape of calibration_features is [M, 14] where M is the number of sub-windows in the first 5 mins
    if len(calibration_features) > 0:
        calib = np.array(calibration_features)
        mu = np.mean(calib, axis=0)
        sigma = np.std(calib, axis=0)
        # Avoid division by zero for signals that don't change
        sigma[sigma == 0] = 1e-8
        
        # Apply normalization to all windows for this subject
        X = (X - mu) / sigma
    else:
        print(f"Warning: No calibration data found for {os.path.basename(subject_path)}. Using unnormalized.")
        
    return X, Y

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    splits = {
        "train": SUBJECTS_TRAIN,
        "val": SUBJECTS_VAL,
        "test": SUBJECTS_TEST
    }
    
    for split_name, subjects in splits.items():
        split_x = []
        split_y = []
        for subj in subjects:
            subj_dir = os.path.join(DATA_DIR, subj)
            if not os.path.exists(subj_dir):
                continue
                
            x, y = process_subject(subj_dir)
            if x is not None:
                split_x.append(x)
                split_y.append(y)
                
        if split_x:
            X = np.concatenate(split_x, axis=0)
            Y = np.concatenate(split_y, axis=0)
            
            # Save to pt
            out_file = os.path.join(OUT_DIR, f"{split_name}.pt")
            torch.save({
                "x": torch.tensor(X, dtype=torch.float32),
                "y": torch.tensor(Y, dtype=torch.long)
            }, out_file)
            
            print(f"\n[{split_name.upper()}] Saved to {out_file}")
            print(f"X shape: {X.shape}")
            unique, counts = np.unique(Y, return_counts=True)
            print(f"Class dist: {dict(zip(unique, counts))}")

if __name__ == "__main__":
    main()
