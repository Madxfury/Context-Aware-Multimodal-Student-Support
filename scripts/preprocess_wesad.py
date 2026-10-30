import os
import pickle
import numpy as np
import scipy.interpolate
import torch
import json
from glob import glob
from tqdm import tqdm

DATA_DIR = "DataSet/WESAD/WESAD"
OUT_DIR = "DataSet/WESAD_Processed"
TARGET_HZ = 64
WINDOW_SEC = 60
STRIDE_SEC = 30
WINDOW_SAMPLES = WINDOW_SEC * TARGET_HZ
STRIDE_SAMPLES = STRIDE_SEC * TARGET_HZ

SUBJECTS_TRAIN = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13"]
SUBJECTS_VAL = ["S14", "S15"]
SUBJECTS_TEST = ["S16", "S17"]

def resample_signal(signal, orig_fs, target_fs, target_len=None):
    """Resamples a 1D or 2D signal using linear interpolation."""
    orig_len = len(signal)
    orig_time = np.linspace(0, orig_len / orig_fs, orig_len)
    
    if target_len is None:
        target_len = int(orig_len * (target_fs / orig_fs))
        
    target_time = np.linspace(0, orig_len / orig_fs, target_len)
    
    if len(signal.shape) == 1 or signal.shape[1] == 1:
        f = scipy.interpolate.interp1d(orig_time, signal.flatten(), bounds_error=False, fill_value="extrapolate")
        res = f(target_time).reshape(-1, 1)
    else:
        res = np.zeros((target_len, signal.shape[1]))
        for i in range(signal.shape[1]):
            f = scipy.interpolate.interp1d(orig_time, signal[:, i], bounds_error=False, fill_value="extrapolate")
            res[:, i] = f(target_time)
    return res

def find_contiguous_blocks(labels, target_label):
    """Returns list of (start_idx, end_idx) for contiguous blocks of target_label."""
    is_target = (labels == target_label).astype(int)
    diff = np.diff(is_target, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))

def process_subject(subject_path):
    print(f"Processing {os.path.basename(subject_path)}...")
    pkl_file = os.path.join(subject_path, f"{os.path.basename(subject_path)}.pkl")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
    labels = data['label']
    duration = len(labels) / 700.0
    target_len = int(duration * TARGET_HZ)
    
    # 1. Resample signals to TARGET_HZ
    bvp = data['signal']['wrist']['BVP']
    eda = data['signal']['wrist']['EDA']
    temp = data['signal']['wrist']['TEMP']
    acc = data['signal']['wrist']['ACC']
    
    bvp_resampled = resample_signal(bvp, 64, TARGET_HZ, target_len=target_len)
    eda_resampled = resample_signal(eda, 4, TARGET_HZ, target_len=target_len)
    temp_resampled = resample_signal(temp, 4, TARGET_HZ, target_len=target_len)
    acc_resampled = resample_signal(acc, 32, TARGET_HZ, target_len=target_len)
    
    # Combine into shape [N, 6] (BVP, EDA, TEMP, ACC_X, ACC_Y, ACC_Z)
    combined = np.hstack([bvp_resampled, eda_resampled, temp_resampled, acc_resampled])
    
    # 2. Find blocks in 700Hz and convert to TARGET_HZ indices
    # Label 1 = Baseline -> Class 0
    # Label 2 = Stress -> Class 1
    blocks_class0 = find_contiguous_blocks(labels, 1)
    blocks_class1 = find_contiguous_blocks(labels, 2)
    
    windows = []
    y = []
    
    for cls, blocks in [(0, blocks_class0), (1, blocks_class1)]:
        for start_700, end_700 in blocks:
            start_hz = int((start_700 / 700.0) * TARGET_HZ)
            end_hz = int((end_700 / 700.0) * TARGET_HZ)
            
            # Slide window
            idx = start_hz
            while idx + WINDOW_SAMPLES <= end_hz:
                win = combined[idx : idx + WINDOW_SAMPLES, :]
                windows.append(win)
                y.append(cls)
                idx += STRIDE_SAMPLES
                
    return np.array(windows), np.array(y)

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Dataset not found at {DATA_DIR}. Please download it first.")
        return
        
    os.makedirs(OUT_DIR, exist_ok=True)
    
    splits = {
        "train": SUBJECTS_TRAIN,
        "val": SUBJECTS_VAL,
        "test": SUBJECTS_TEST
    }
    
    data_dict = {}
    
    for split_name, subjects in splits.items():
        split_x = []
        split_y = []
        for subj in subjects:
            subj_dir = os.path.join(DATA_DIR, subj)
            if not os.path.exists(subj_dir):
                print(f"Warning: Subject {subj} not found. Skipping.")
                continue
            x, y = process_subject(subj_dir)
            split_x.append(x)
            split_y.append(y)
            
        if split_x:
            data_dict[split_name] = {
                "x": np.vstack(split_x),
                "y": np.concatenate(split_y)
            }
        else:
            print(f"No data for split {split_name}!")
            data_dict[split_name] = {"x": np.empty(0), "y": np.empty(0)}
            
    # Calculate global mean and std from TRAIN
    train_x = data_dict["train"]["x"]
    # train_x shape: [N_windows, WINDOW_SAMPLES, 6]
    # We want mean/std across windows and time, so per-channel shape: [6]
    # Reshape to [N_windows * WINDOW_SAMPLES, 6]
    train_pixels = train_x.reshape(-1, 6)
    channel_mean = np.mean(train_pixels, axis=0)
    channel_std = np.std(train_pixels, axis=0)
    
    print("\n--- Normalization Stats (from Train) ---")
    print(f"Mean: {channel_mean}")
    print(f"Std: {channel_std}")
    
    # Apply to all
    for split_name in ["train", "val", "test"]:
        if data_dict[split_name]["x"].size > 0:
            norm_x = (data_dict[split_name]["x"] - channel_mean) / channel_std
            data_dict[split_name]["x"] = torch.tensor(norm_x, dtype=torch.float32)
            data_dict[split_name]["y"] = torch.tensor(data_dict[split_name]["y"], dtype=torch.long)
            
            # Save to pt
            out_file = os.path.join(OUT_DIR, f"{split_name}.pt")
            torch.save(data_dict[split_name], out_file)
            
            print(f"\n[{split_name.upper()}]")
            print(f"X shape: {data_dict[split_name]['x'].shape}")
            print(f"Y shape: {data_dict[split_name]['y'].shape}")
            unique, counts = np.unique(data_dict[split_name]['y'].numpy(), return_counts=True)
            print(f"Class distribution: {dict(zip(unique, counts))}")
            print(f"Saved to {out_file}")
            
    # Save metadata
    meta = {
        "target_hz": TARGET_HZ,
        "window_sec": WINDOW_SEC,
        "stride_sec": STRIDE_SEC,
        "channels": ["BVP", "EDA", "TEMP", "ACC_X", "ACC_Y", "ACC_Z"],
        "mean": channel_mean.tolist(),
        "std": channel_std.tolist(),
        "subjects": splits
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)
        
    print("\nPreprocessing complete. Data saved to", OUT_DIR)

if __name__ == "__main__":
    main()
