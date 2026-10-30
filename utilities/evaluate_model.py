# ==============================
# evaluate_model.py — Final Safe Version
# ==============================

import torch
from torchvision import datasets, transforms
import timm
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
import os

# ------------------------------
# 1. Configuration
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

data_dir = "/Users/sanskarparab/CC Emotion Detection /Facial-Expression-Recognition-FER-for-Mental-Health-Detection-/traintestsplit/test"
model_path = "/Users/sanskarparab/CC Emotion Detection /Facial-Expression-Recognition-FER-for-Mental-Health-Detection-/Models/Swin_Transformer/Swin_Transformer_best_model.pth"
num_classes = 7  # For FANE dataset

# ------------------------------
# 2. Safe Image Loader
# ------------------------------
def safe_loader(path):
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        print(f"⚠️ Skipping corrupted image: {path}")
        return None

class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = safe_loader(path)
        while sample is None:
            # Skip bad images gracefully
            index = (index + 1) % len(self.samples)
            path, target = self.samples[index]
            sample = safe_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

# ------------------------------
# 3. Data Transform
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = SafeImageFolder(root=data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# ------------------------------
# 4. Load Model
# ------------------------------
checkpoint = torch.load(model_path, map_location=device)
if "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]

model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes)
missing, unexpected = model.load_state_dict(checkpoint, strict=False)
print(f"➡️ Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
print("✅ Model ready!\n")

model.to(device)
model.eval()

# ------------------------------
# 5. Evaluate
# ------------------------------
correct = 0
total = 0
print("🧪 Starting evaluation...")

with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"\n📊 Evaluation Complete — Accuracy: {accuracy:.2f}%")
