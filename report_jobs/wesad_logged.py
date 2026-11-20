"""Re-run of scripts/train_wesad_models.py with per-epoch logging for the report's curves.
It imports the project's own dataset class, model and evaluate() and keeps every setting
(Adam lr 0.001, batch 32, 20 epochs, class-weighted cross-entropy, best validation F1 kept).
Additions: fixed seed 42; train accuracy, validation loss/accuracy/F1 logged every epoch; test
probabilities saved for ROC curves. One fix, stated in the report: the original script keeps
`best_state = model.state_dict()`, which holds references that later epochs overwrite, so its
"best" model is really the last epoch. Here the best state is deep-copied."""
import os, sys, copy, json, random
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(PROJECT)
sys.path.insert(0, os.path.join(PROJECT, 'scripts'))
import train_wesad_models as T
OUT = os.path.join(PROJECT, 'report_jobs', 'outputs'); os.makedirs(OUT, exist_ok=True)
SEED = 42; DEV = T.DEVICE

def seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tr, va, te = T.WESADDataset('train'), T.WESADDataset('val'), T.WESADDataset('test')
y = tr.y.numpy(); w = torch.tensor([len(y) / (2.0 * (y == 0).sum()), len(y) / (2.0 * (y == 1).sum())], dtype=torch.float32)
info = {'seed': SEED, 'device': str(DEV), 'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
        'class_counts_train': [int((y == 0).sum()), int((y == 1).sum())],
        'class_counts_val': [int((va.y == 0).sum()), int((va.y == 1).sum())],
        'class_counts_test': [int((te.y == 0).sum()), int((te.y == 1).sum())], 'class_weights': w.tolist()}
rows = []
for name, bi in (('LSTM', False), ('BiLSTM', True)):
    seed()
    g = torch.Generator(); g.manual_seed(SEED)
    trl = DataLoader(tr, batch_size=T.BATCH_SIZE, shuffle=True, generator=g)
    val = DataLoader(va, batch_size=T.BATCH_SIZE); tel = DataLoader(te, batch_size=T.BATCH_SIZE)
    m = T.PhysiologicalLSTM(bidirectional=bi).to(DEV); crit = nn.CrossEntropyLoss(weight=w.to(DEV))
    opt = optim.Adam(m.parameters(), lr=T.LEARNING_RATE); best, best_state, best_ep = -1.0, None, None
    for ep in range(T.EPOCHS):
        m.train(); tot = cor = 0
        for x, yy in trl:
            x, yy = x.to(DEV), yy.to(DEV); opt.zero_grad(); out = m(x); loss = crit(out, yy); loss.backward(); opt.step()
            tot += loss.item() * x.size(0); cor += (out.argmax(1) == yy).sum().item()
        vm = T.evaluate(m, val, crit)
        rows.append({'model': name, 'epoch': ep + 1, 'train_loss': tot / len(tr), 'train_acc': cor / len(tr),
                     'val_loss': vm['loss'], 'val_acc': vm['acc'], 'val_f1': vm['f1']})
        print(rows[-1], flush=True)
        if vm['f1'] > best:
            best, best_ep, best_state = vm['f1'], ep + 1, copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state); tm = T.evaluate(m, tel, crit); m.eval()
    with torch.no_grad():
        P = torch.cat([torch.softmax(m(x.to(DEV)), 1).cpu() for x, _ in tel]).numpy()
    np.savetxt(os.path.join(OUT, f'wesad_{name.lower()}_test_probs.csv'), np.column_stack([te.y.numpy(), P]),
               delimiter=',', header='y_true,p_baseline,p_stress', comments='')
    torch.save(best_state, os.path.join(OUT, f'wesad_best_{name.lower()}_report.pth'))
    info[name] = {'best_epoch': best_ep, 'best_val_f1': best, 'test': tm}
import pandas as pd
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'wesad_epochs.csv'), index=False)
json.dump(info, open(os.path.join(OUT, 'wesad_report_info.json'), 'w'), indent=2, default=float)
print(json.dumps(info, indent=2, default=float))
