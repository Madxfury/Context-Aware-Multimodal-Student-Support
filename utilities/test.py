import torch
path = "/Users/sanskarparab/CC Emotion Detection /Facial-Expression-Recognition-FER-for-Mental-Health-Detection-/Models/Swin_Transformer/Swin_Transformer_best_model.pth"

try:
    data = torch.load(path, map_location='cpu')
    print("✅ Model file loaded successfully!")
    print(type(data))
except Exception as e:
    print("❌ Error:", e)

