import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import time

DATA_DIR = "DataSet/WESAD_Processed_V2"
OUT_DIR = "Models/Physiological_V2"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

class WESADDataset(Dataset):
    def __init__(self, split="train"):
        data = torch.load(os.path.join(DATA_DIR, f"{split}.pt"), weights_only=True)
        self.x = data['x']
        self.y = data['y']
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class PhysiologicalLSTM(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, num_layers=1, bidirectional=False, num_classes=2):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_out_dim, num_classes)
        
    def forward(self, x):
        # x shape: [B, 12, 14]
        out, (hn, cn) = self.lstm(x)
        
        # Get last hidden state
        if self.bidirectional:
            # Concat forward and backward final hidden states
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
            
    avg_loss = total_loss / len(dataloader.dataset)
    
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
    
    cm = [[tn, fp], [fn, tp]]
    
    metrics = {
        'loss': avg_loss,
        'acc': acc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'cm': cm
    }
    return metrics

def train_model(model, train_loader, val_loader, class_weights, model_name):
    print(f"\n--- Training {model_name} ---")
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_f1 = -1.0
    best_state = None
    best_metrics = None
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            
        train_loss /= len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_state = model.state_dict()
            best_metrics = val_metrics
            
    # Save best model
    os.makedirs(OUT_DIR, exist_ok=True)
    save_path = os.path.join(OUT_DIR, f"best_{model_name.lower()}.pth")
    torch.save(best_state, save_path)
    print(f"Saved best {model_name} to {save_path} (Val F1: {best_val_f1:.4f})")
    
    return best_state, best_metrics

def main():
    print(f"Using device: {DEVICE}")
    
    train_dataset = WESADDataset("train")
    val_dataset = WESADDataset("val")
    test_dataset = WESADDataset("test")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Calculate class weights for Class 0 (Baseline) and Class 1 (Stress)
    # y = 0: 414, y = 1: 225
    y_train = train_dataset.y.numpy()
    num_class0 = np.sum(y_train == 0)
    num_class1 = np.sum(y_train == 1)
    total = len(y_train)
    
    # inverse frequency
    weight0 = total / (2.0 * num_class0)
    weight1 = total / (2.0 * num_class1)
    class_weights = torch.tensor([weight0, weight1], dtype=torch.float32)
    print(f"Class Weights -> Baseline: {weight0:.4f}, Stress: {weight1:.4f}")
    
    # 1. Train LSTM
    lstm = PhysiologicalLSTM(bidirectional=False)
    lstm_best_state, lstm_val_metrics = train_model(lstm, train_loader, val_loader, class_weights, "LSTM")
    
    # 2. Train BiLSTM
    bilstm = PhysiologicalLSTM(bidirectional=True)
    bilstm_best_state, bilstm_val_metrics = train_model(bilstm, train_loader, val_loader, class_weights, "BiLSTM")
    
    # 3. Evaluate on Test Set
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    
    print("\n=========================================")
    print("        TEST SET EVALUATION              ")
    print("=========================================")
    
    # LSTM Test
    lstm.load_state_dict(lstm_best_state)
    lstm_test_metrics = evaluate(lstm, test_loader, criterion)
    
    # BiLSTM Test
    bilstm.load_state_dict(bilstm_best_state)
    bilstm_test_metrics = evaluate(bilstm, test_loader, criterion)
    
    results = {
        "LSTM": {
            "val": lstm_val_metrics,
            "test": lstm_test_metrics
        },
        "BiLSTM": {
            "val": bilstm_val_metrics,
            "test": bilstm_test_metrics
        }
    }
    
    for name, res in results.items():
        print(f"\n--- {name} Final Test Results ---")
        print(f"Accuracy:  {res['test']['acc']:.4f}")
        print(f"Precision: {res['test']['prec']:.4f}")
        print(f"Recall:    {res['test']['rec']:.4f}")
        print(f"F1 Score:  {res['test']['f1']:.4f}")
        print(f"Confusion Matrix:\n{np.array(res['test']['cm'])}")
        
    with open(os.path.join(OUT_DIR, "training_metadata.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nTraining metadata saved to {os.path.join(OUT_DIR, 'training_metadata.json')}")

if __name__ == "__main__":
    main()
