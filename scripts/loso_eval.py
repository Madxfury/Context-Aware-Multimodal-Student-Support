import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json

DATA_DIR = "DataSet/WESAD/WESAD"
WINDOW_SEC = 60
STRIDE_SEC = 30
SUB_WINDOW_SEC = 5
NUM_SUB_WINDOWS = 12
CALIBRATION_SEC = 300
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

ALL_SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17"]

# --- PREPROCESSING ---
def get_native_signal_slice(signal, start_sec, end_sec, hz):
    start_idx = int(start_sec * hz)
    end_idx = int(end_sec * hz)
    return signal[start_idx:end_idx]

def extract_features(bvp, eda, temp, acc):
    bvp_mean = np.mean(bvp)
    bvp_std = np.std(bvp)
    bvp_range = np.max(bvp) - np.min(bvp) if len(bvp) > 0 else 0
    bvp_energy = np.sum(bvp**2)
    eda_mean = np.mean(eda)
    eda_std = np.std(eda)
    eda_range = np.max(eda) - np.min(eda) if len(eda) > 0 else 0
    eda_slope = eda[-1][0] - eda[0][0] if len(eda) > 1 else 0
    temp_mean = np.mean(temp)
    temp_std = np.std(temp)
    temp_range = np.max(temp) - np.min(temp) if len(temp) > 0 else 0
    temp_slope = temp[-1][0] - temp[0][0] if len(temp) > 1 else 0
    if len(acc) > 0:
        acc_mag = np.sqrt(np.sum(acc**2, axis=1))
        acc_mean = np.mean(acc_mag)
        acc_std = np.std(acc_mag)
    else:
        acc_mean, acc_std = 0, 0
    return [bvp_mean, bvp_std, bvp_range, bvp_energy,
            eda_mean, eda_std, eda_range, eda_slope,
            temp_mean, temp_std, temp_range, temp_slope,
            acc_mean, acc_std]

def process_subject(subject_path):
    pkl_file = os.path.join(subject_path, f"{os.path.basename(subject_path)}.pkl")
    if not os.path.exists(pkl_file):
        return None, None
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    labels = data['label']
    bvp = data['signal']['wrist']['BVP']
    eda = data['signal']['wrist']['EDA']
    temp = data['signal']['wrist']['TEMP']
    acc = data['signal']['wrist']['ACC']
    
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
            curr_sec = start_sec
            while curr_sec + WINDOW_SEC <= end_sec:
                win_feats = []
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
        
    X = np.array(windows_features)
    Y = np.array(y)
    
    if len(calibration_features) > 0:
        calib = np.array(calibration_features)
        mu = np.mean(calib, axis=0)
        sigma = np.std(calib, axis=0)
        sigma[sigma == 0] = 1e-8
        X = (X - mu) / sigma
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long)

class MemoryDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# --- MODEL ---
class PhysiologicalLSTM(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, num_layers=1, bidirectional=False, num_classes=2):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_out_dim, num_classes)
        
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        if self.bidirectional:
            hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1)
        else:
            hidden = hn[-1,:,:]
        out = self.fc(hidden)
        return out

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0
    all_preds_t = torch.tensor(all_preds)
    all_labels_t = torch.tensor(all_labels)
    tp = ((all_preds_t == 1) & (all_labels_t == 1)).sum().item()
    tn = ((all_preds_t == 0) & (all_labels_t == 0)).sum().item()
    fp = ((all_preds_t == 1) & (all_labels_t == 0)).sum().item()
    fn = ((all_preds_t == 0) & (all_labels_t == 1)).sum().item()
    
    acc = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    return {
        'loss': avg_loss, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
        'cm': [[tn, fp], [fn, tp]]
    }

def train_model(model, train_loader, val_loader, class_weights):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_f1 = -1.0
    best_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
        val_metrics = evaluate(model, val_loader, criterion)
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_state = model.state_dict()
            
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, criterion

def main():
    print(f"Starting Leave-Subject-Out Evaluation on {DEVICE}")
    
    # 1. Process all subjects and load to memory
    subj_data = {}
    valid_subjects = []
    print("Preprocessing subjects...")
    for subj in ALL_SUBJECTS:
        path = os.path.join(DATA_DIR, subj)
        x, y = process_subject(path)
        if x is not None:
            # Check if subject has both classes post-calibration
            unique_classes = torch.unique(y).tolist()
            if 0 in unique_classes and 1 in unique_classes:
                subj_data[subj] = (x, y)
                valid_subjects.append(subj)
            else:
                print(f"Excluding {subj}: Missing Baseline or Stress windows post-calibration. (Classes found: {unique_classes})")
        else:
            print(f"Excluding {subj}: No valid data found or file missing.")

    print(f"\nValid subjects for LOSO: {valid_subjects}")
    
    results = {}
    
    # 2. Leave-Subject-Out Loop
    for test_subj in valid_subjects:
        print(f"\n================ LOSO Test: {test_subj} ================")
        
        # Test set is the held out subject
        te_x, te_y = subj_data[test_subj]
        
        # Validation set will be the NEXT 2 valid subjects (wrapping around)
        test_idx = valid_subjects.index(test_subj)
        val_idx_1 = (test_idx + 1) % len(valid_subjects)
        val_idx_2 = (test_idx + 2) % len(valid_subjects)
        val_subjs = [valid_subjects[val_idx_1], valid_subjects[val_idx_2]]
        
        # Train set is everyone else
        train_subjs = [s for s in valid_subjects if s != test_subj and s not in val_subjs]
        
        print(f"Train ({len(train_subjs)}): {train_subjs}")
        print(f"Val ({len(val_subjs)}): {val_subjs}")
        
        tr_x = torch.cat([subj_data[s][0] for s in train_subjs], dim=0)
        tr_y = torch.cat([subj_data[s][1] for s in train_subjs], dim=0)
        
        v_x = torch.cat([subj_data[s][0] for s in val_subjs], dim=0)
        v_y = torch.cat([subj_data[s][1] for s in val_subjs], dim=0)
        
        print(f"Train shapes: X={tr_x.shape}, Y={tr_y.shape}")
        print(f"Val shapes: X={v_x.shape}, Y={v_y.shape}")
        print(f"Test shapes: X={te_x.shape}, Y={te_y.shape}")
        
        unique, counts = np.unique(tr_y.numpy(), return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_class0 = counts_dict.get(0, 0)
        num_class1 = counts_dict.get(1, 0)
        total = len(tr_y)
        w0 = total / (2.0 * num_class0) if num_class0 > 0 else 1.0
        w1 = total / (2.0 * num_class1) if num_class1 > 0 else 1.0
        class_weights = torch.tensor([w0, w1], dtype=torch.float32)
        
        train_loader = DataLoader(MemoryDataset(tr_x, tr_y), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(MemoryDataset(v_x, v_y), batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(MemoryDataset(te_x, te_y), batch_size=BATCH_SIZE, shuffle=False)
        
        # Train LSTM
        lstm = PhysiologicalLSTM(bidirectional=False)
        lstm, criterion = train_model(lstm, train_loader, val_loader, class_weights)
        lstm_test = evaluate(lstm, test_loader, criterion)
        
        # Train BiLSTM
        bilstm = PhysiologicalLSTM(bidirectional=True)
        bilstm, _ = train_model(bilstm, train_loader, val_loader, class_weights)
        bilstm_test = evaluate(bilstm, test_loader, criterion)
        
        results[test_subj] = {
            'LSTM': lstm_test,
            'BiLSTM': bilstm_test,
            'windows': {'base': (te_y == 0).sum().item(), 'stress': (te_y == 1).sum().item()}
        }
        
        print(f"LSTM Test -> Acc: {lstm_test['acc']:.4f}, Prec: {lstm_test['prec']:.4f}, Rec: {lstm_test['rec']:.4f}, F1: {lstm_test['f1']:.4f}")
        print(f"BiLSTM Test -> Acc: {bilstm_test['acc']:.4f}, Prec: {bilstm_test['prec']:.4f}, Rec: {bilstm_test['rec']:.4f}, F1: {bilstm_test['f1']:.4f}")

    print("\n================ LOSO SUMMARY STATISTICS ================")
    models = ['LSTM', 'BiLSTM']
    metrics = ['acc', 'prec', 'rec', 'f1']
    
    for m in models:
        print(f"\n--- {m} ---")
        for metric in metrics:
            vals = [results[s][m][metric] for s in valid_subjects]
            mean = np.mean(vals)
            std = np.std(vals)
            print(f"{metric.upper()}: {mean:.4f} ± {std:.4f}")

if __name__ == "__main__":
    main()
