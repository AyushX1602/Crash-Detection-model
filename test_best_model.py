"""Quick test evaluation of best model"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import classification_report
import pickle

device = torch.device('cuda')
CACHE_DIR = 'D:/programs vs/crash model/preprocessed'
BATCH_SIZE = 8

class SimpleDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = torch.load(X_path, weights_only=True)
        self.y = torch.load(y_path, weights_only=True)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class AntiOverfitCrashClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 1280
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.7),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = x.view(B, T, -1)
        lstm_out, _ = self.lstm(x)
        weights = self.attention(lstm_out)
        weights = torch.softmax(weights, dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        return self.classifier(context)

# Load
with open(os.path.join(CACHE_DIR, 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)

categories = list(label_encoder.classes_)
num_classes = len(categories)

test_dataset = SimpleDataset(
    os.path.join(CACHE_DIR, 'test_X.pt'),
    os.path.join(CACHE_DIR,  'test_y.pt')
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = AntiOverfitCrashClassifier(num_classes).to(device)
model.load_state_dict(torch.load('best_video_crash_model.pth'))
model.eval()

print("\n🧪 TESTING BEST MODEL (Epoch 26: 62.67% val)...")

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        out = model(batch_x)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(batch_y.numpy())

print("\nCLASSIFICATION REPORT:")
print(classification_report(all_labels, all_preds, target_names=categories))

test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
print(f"\n📊 FINAL TEST ACCURACY: {100*test_acc:.2f}%")
print(f"   Baseline was: 48%")
print(f"   Improvement: +{100*(test_acc - 0.48):.2f}%")
