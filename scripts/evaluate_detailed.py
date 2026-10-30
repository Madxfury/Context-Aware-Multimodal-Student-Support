import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json

class PhysiologicalLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=1, bidirectional=False, num_classes=2):
        super().__init__()
        self.bidirectional = bidirectional
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_out_dim, num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.transpose(1, 2)
        out, (hn, cn) = self.lstm(x)
        if self.bidirectional:
            hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1)
        else:
            hidden = hn[-1,:,:]
        out = self.fc(hidden)
        return out

class WESADDataset(Dataset):
    def __init__(self, split="train"):
        data = torch.load(os.path.join("DataSet/WESAD_Processed", f"{split}.pt"), map_location='cpu', weights_only=True)
        self.x = data['x']
        self.y = data['y']
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

def evaluate_model(model_class, weights_path, dataloader):
    model = model_class
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    
    # Per-class metrics
    prec_baseline = tn / (tn + fn) if (tn + fn) > 0 else 0
    rec_baseline = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_baseline = 2 * (prec_baseline * rec_baseline) / (prec_baseline + rec_baseline) if (prec_baseline + rec_baseline) > 0 else 0
    
    prec_stress = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_stress = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_stress = 2 * (prec_stress * rec_stress) / (prec_stress + rec_stress) if (prec_stress + rec_stress) > 0 else 0
    
    # Average confidences
    conf_class0 = all_probs[all_preds == 0, 0]
    conf_class1 = all_probs[all_preds == 1, 1]
    
    return {
        "acc": (tp + tn) / len(all_labels),
        "cm": [[tn, fp], [fn, tp]],
        "class_0": {"prec": prec_baseline, "rec": rec_baseline, "f1": f1_baseline, "avg_conf": conf_class0.mean() if len(conf_class0) > 0 else 0},
        "class_1": {"prec": prec_stress, "rec": rec_stress, "f1": f1_stress, "avg_conf": conf_class1.mean() if len(conf_class1) > 0 else 0}
    }

def main():
    test_dataset = WESADDataset("test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    lstm = PhysiologicalLSTM(bidirectional=False)
    bilstm = PhysiologicalLSTM(bidirectional=True)
    
    lstm_res = evaluate_model(lstm, "Models/Physiological/best_lstm.pth", test_loader)
    bilstm_res = evaluate_model(bilstm, "Models/Physiological/best_bilstm.pth", test_loader)
    
    results = {"LSTM": lstm_res, "BiLSTM": bilstm_res}
    
    print("\n=========================================")
    print("        DEEP EVALUATION RESULTS          ")
    print("=========================================")
    
    for name, res in results.items():
        print(f"\n--- {name} ---")
        print(f"Accuracy: {res['acc']:.4f}")
        print(f"Confusion Matrix: TN={res['cm'][0][0]}, FP={res['cm'][0][1]}, FN={res['cm'][1][0]}, TP={res['cm'][1][1]}")
        print(f"Baseline (Class 0): Prec={res['class_0']['prec']:.4f}, Rec={res['class_0']['rec']:.4f}, F1={res['class_0']['f1']:.4f}, Avg Conf={res['class_0']['avg_conf']:.4f}")
        print(f"Stress (Class 1): Prec={res['class_1']['prec']:.4f}, Rec={res['class_1']['rec']:.4f}, F1={res['class_1']['f1']:.4f}, Avg Conf={res['class_1']['avg_conf']:.4f}")

if __name__ == "__main__":
    main()
