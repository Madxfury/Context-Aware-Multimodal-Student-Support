"""Retrain Swin-FANE for the report with the protocol of utilities/train_model.ipynb:
phase 1 = backbone frozen, head trained 5 epochs (AdamW lr 5e-4, batch 32, cross-entropy);
phase 2 = whole model fine-tuned epochs 6-15 (AdamW lr 1e-5); best validation accuracy in phase 2 kept.
Differences from the notebook, all for the report: the deduplicated split in fane_split_report.csv
(the 406 never-seen test images held out), a fixed seed, and train/val loss and accuracy logged every epoch.
If the measured speed means full training would take more than MAX_HOURS, the smaller
swin_tiny_patch4_window7_224 backbone is trained instead with the same protocol, and the log says so."""
import os, sys, json, time, copy
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(__file__))
from fane_common import *

MAX_HOURS = float(os.environ.get('MAX_HOURS', '3'))
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'outputs'); os.makedirs(OUT, exist_ok=True)
root = fane_root(); dev = device()
split = pd.read_csv(os.path.join(HERE, 'fane_split_report.csv'))
tr, va, te = (split[split.split == s].reset_index(drop=True) for s in ('train', 'val', 'test'))
info = {'device': str(dev), 'torch': torch.__version__, 'seed': SEED, 'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
        'protocol': 'phase1 head-only 5 ep AdamW 5e-4; phase2 full 10 ep AdamW 1e-5; batch 32; CE; best val acc in phase 2'}
log_path = os.path.join(OUT, 'fane_retrain_epochs.csv')

def loaders(bs):
    g = torch.Generator(); g.manual_seed(SEED)
    return (DataLoader(CsvImages(tr, root), batch_size=bs, shuffle=True, generator=g),
            DataLoader(CsvImages(va, root), batch_size=bs), DataLoader(CsvImages(te, root), batch_size=bs))

def probe(backbone, bs=32, steps=8):
    """Seconds per training step with the backbone frozen and unfrozen, and per eval step."""
    seed_everything(); m = CustomSwinTransformer(backbone, pretrained=True).to(dev); crit = nn.CrossEntropyLoss()
    x = torch.randn(bs, 3, 224, 224, device=dev); y = torch.randint(0, 7, (bs,), device=dev); res = {}
    for mode in ('frozen', 'full'):
        for p in m.backbone.parameters(): p.requires_grad = (mode == 'full')
        opt = optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-5); m.train()
        for i in range(steps + 2):
            if i == 2: t0 = time.time()
            opt.zero_grad(); crit(m(x), y).backward(); opt.step()
        if dev.type == 'mps': torch.mps.synchronize()
        res[mode] = (time.time() - t0) / steps
    m.eval(); t0 = time.time()
    with torch.no_grad():
        for _ in range(steps): m(x)
    if dev.type == 'mps': torch.mps.synchronize()
    res['eval'] = (time.time() - t0) / steps
    del m; return res

def estimate_hours(r, bs=32):
    nb_tr, nb_va = -(-len(tr) // bs), -(-len(va) // bs)
    return (5 * (nb_tr * r['frozen'] + nb_va * r['eval']) + 10 * (nb_tr * r['full'] + nb_va * r['eval'])) / 3600

backbone = 'swin_base_patch4_window7_224'
r = probe(backbone); h = estimate_hours(r); info['probe_swin_base'] = {**r, 'estimated_hours': h}
print(f'Swin-Base estimate: {h:.2f} h', r, flush=True)
if h > MAX_HOURS:
    backbone = 'swin_tiny_patch4_window7_224'
    r = probe(backbone); info['probe_swin_tiny'] = {**r, 'estimated_hours': estimate_hours(r)}
    info['note'] = f'Swin-Base estimated at {h:.1f} h (> {MAX_HOURS} h), so the smaller Swin-Tiny backbone was trained.'
    print(info['note'], flush=True)
info['backbone'] = backbone

seed_everything()
train_loader, val_loader, test_loader = loaders(32)
model = CustomSwinTransformer(backbone, pretrained=True).to(dev)
crit = nn.CrossEntropyLoss()

def run(loader, train, opt=None):
    model.train(train); tot = cor = n = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            if train: opt.zero_grad()
            out = model(x); loss = crit(out, y)
            if train: loss.backward(); opt.step()
            tot += loss.item() * x.size(0); cor += (out.argmax(1) == y).sum().item(); n += x.size(0)
    return tot / n, cor / n

rows, best, best_state, t_start = [], -1.0, None, time.time()
for epoch in range(15):
    phase = 1 if epoch < 5 else 2
    if epoch == 0:
        for p in model.backbone.parameters(): p.requires_grad = False
        opt = optim.AdamW(model.classifier.parameters(), lr=5e-4)
    if epoch == 5:
        for p in model.parameters(): p.requires_grad = True
        opt = optim.AdamW(model.parameters(), lr=1e-5)
    t0 = time.time(); trl, tra = run(train_loader, True, opt); val, vaa = run(val_loader, False)
    rows.append({'epoch': epoch + 1, 'phase': phase, 'train_loss': trl, 'train_acc': tra, 'val_loss': val, 'val_acc': vaa, 'seconds': time.time() - t0})
    pd.DataFrame(rows).to_csv(log_path, index=False)
    print(rows[-1], flush=True)
    if phase == 2 and vaa > best:
        best, best_state = vaa, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
info['best_val_acc'] = best; info['best_epoch'] = int(max((r for r in rows if r['phase'] == 2), key=lambda r: r['val_acc'])['epoch'])
info['train_hours'] = (time.time() - t_start) / 3600
torch.save(best_state, os.path.join(OUT, f'swin_fane_retrained_{backbone.split("_")[1]}.pth'))
model.load_state_dict(best_state); model.to(dev)
probs_frame(te, predict(model, test_loader, dev), ['file_in_split']).to_csv(os.path.join(OUT, 'retrained_swin_test406_probs.csv'), index=False)
probs_frame(va, predict(model, val_loader, dev)).to_csv(os.path.join(OUT, 'retrained_swin_val_probs.csv'), index=False)
json.dump(info, open(os.path.join(OUT, 'fane_retrain_info.json'), 'w'), indent=2)
print(json.dumps(info, indent=2))
