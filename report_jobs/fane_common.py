"""Shared pieces for the report's FANE jobs. Model and preprocessing copied from
utilities/train_model.ipynb and app.py (same Swin head, same 224x224 ImageNet normalisation)."""
import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']  # ImageFolder order, as in app.py
SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomSwinTransformer(nn.Module):
    """Identical to utilities/train_model.ipynb and app.py, with the backbone name as a parameter."""
    def __init__(self, backbone='swin_base_patch4_window7_224', pretrained=True, num_classes=7):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512), nn.ReLU(), nn.Dropout(p=0.6), nn.Linear(512, num_classes))
    def forward(self, x):
        return self.classifier(self.backbone(x))

TRANSFORM = transforms.Compose([
    lambda img: img.convert('RGB'),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class CsvImages(Dataset):
    def __init__(self, frame, root):
        self.files = list(frame['file']); self.labels = [EMOTIONS.index(e) for e in frame['emotion']]; self.root = root
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        return TRANSFORM(Image.open(os.path.join(self.root, self.files[i]))), self.labels[i]

@torch.no_grad()
def predict(model, loader, dev):
    model.eval(); out = []
    for x, _ in loader:
        out.append(torch.softmax(model(x.to(dev)), dim=1).float().cpu())
    return torch.cat(out).numpy()

def probs_frame(frame, probs, extra_cols=()):
    df = frame[['file', 'emotion', *extra_cols]].reset_index(drop=True).copy()
    for i, e in enumerate(EMOTIONS):
        df[f'p_{e}'] = probs[:, i]
    df['predicted'] = [EMOTIONS[i] for i in probs.argmax(1)]
    df['confidence'] = probs.max(1)
    return df

def fane_root():
    for c in [os.path.expanduser('~/Downloads/Samvya_Dataset/1_facial_emotion_FANE'),
              os.path.join(os.path.dirname(__file__), 'data', 'Samvya_Dataset', '1_facial_emotion_FANE')]:
        if os.path.isdir(os.path.join(c, 'images')):
            return c
    raise SystemExit('FANE images not found. Unzip ~/Downloads/Samvya_Dataset_Full.zip into ~/Downloads first.')
