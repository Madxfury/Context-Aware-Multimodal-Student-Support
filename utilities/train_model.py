# utilities/train_model.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from tqdm import tqdm

# Custom model class
class CustomSwinTransformer(torch.nn.Module):
    def __init__(self, pretrained=True, num_classes=7):
        super(CustomSwinTransformer, self).__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained, num_classes=0)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.backbone.num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.6),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


# Training function
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct.double() / total
    return epoch_loss, epoch_acc.item()


if __name__ == "__main__":
    # Transform ensures all images are RGB + same size + normalized
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # ✅ Converts grayscale → RGB
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ✅ Change this path to your dataset location
    dataset_root = "/Users/sanskarparab/CC Emotion Detection /Facial-Expression-Recognition-FER-for-Mental-Health-Detection-/traintestsplit/train"

    # Load your dataset
    train_dataset = datasets.ImageFolder(root=dataset_root, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Device selection (use MPS for Apple Silicon)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize model, loss, optimizer
    model = CustomSwinTransformer(pretrained=True, num_classes=7).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(5):
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}: Loss={loss:.4f}, Accuracy={acc:.4f}")

    # Save model after training
    torch.save(model, "fane_swin_model.pth")
    print("✅ Model saved as fane_swin_model.pth")
