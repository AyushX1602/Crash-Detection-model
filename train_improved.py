"""
ANTI-OVERFITTING TRAINING - OPTIMIZED FOR 6GB GPU
SEQUENCE_LENGTH=16, Batch=8, ALL anti-overfitting features
Expected: 65-72% test accuracy
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import classification_report
import pickle
from tqdm import tqdm
import random
import numpy as np
from collections import Counter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# CONFIG
CACHE_DIR = 'D:/programs vs/crash model/preprocessed'
BATCH_SIZE = 8          # ✅ Can use 8 with SEQ=16
EPOCHS = 80
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.15
SEQUENCE_LENGTH = 16

assert torch.cuda.is_available(), "CUDA required!"
device = torch.device('cuda')

print(f"\n{'='*60}")
print("🎯 ANTI-OVERFITTING TRAINING (6GB GPU OPTIMIZED)")
print('='*60)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Sequence: {SEQUENCE_LENGTH} frames | Batch: {BATCH_SIZE}")
print('='*60)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# DATASET
class StrongAugmentedDataset(Dataset):
    def __init__(self, X_path, y_path, augment=False):
        self.X = torch.load(X_path, weights_only=True)
        self.y = torch.load(y_path, weights_only=True)
        self.augment = augment
        print(f"  Loaded: {self.X.shape} (augment={augment})")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        frames = self.X[idx].clone()
        label = self.y[idx]
        
        if self.augment:
            if random.random() < 0.5:
                frames = torch.flip(frames, dims=[3])
            
            brightness = random.uniform(0.6, 1.4)
            frames = frames * brightness
            frames = torch.clamp(frames, 0, 1)
            
            if random.random() < 0.3:
                contrast = random.uniform(0.7, 1.3)
                mean = frames.mean()
                frames = (frames - mean) * contrast + mean
                frames = torch.clamp(frames, 0, 1)
            
            if random.random() < 0.3:
                T = frames.shape[0]
                drop_count = random.randint(1, 3)
                drop_indices = random.sample(range(1, T-1), min(drop_count, T-2))
                for di in drop_indices:
                    frames[di] = frames[di-1]
            
            if random.random() < 0.2:
                noise = torch.randn_like(frames) * 0.05
                frames = frames + noise
                frames = torch.clamp(frames, 0, 1)
        
        return frames, label




print("\n📥 Loading dataset...")

train_dataset = StrongAugmentedDataset(
    os.path.join(CACHE_DIR, 'train_X.pt'),
    os.path.join(CACHE_DIR, 'train_y.pt'),
    augment=True
)
val_dataset = StrongAugmentedDataset(
    os.path.join(CACHE_DIR, 'val_X.pt'),
    os.path.join(CACHE_DIR, 'val_y.pt'),
    augment=False
)
test_dataset = StrongAugmentedDataset(
    os.path.join(CACHE_DIR, 'test_X.pt'),
    os.path.join(CACHE_DIR, 'test_y.pt'),
    augment=False
)

with open(os.path.join(CACHE_DIR, 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)

categories = list(label_encoder.classes_)
num_classes = len(categories)

# Class weights
train_labels = train_dataset.y.numpy()
class_counts = Counter(train_labels)
total = len(train_labels)
class_weights = torch.tensor([total / (num_classes * class_counts[i]) for i in range(num_classes)], dtype=torch.float32).to(device)
print(f"\n📊 Class weights: {class_weights.cpu().numpy()}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                         pin_memory=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

print(f"   Dataset: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

# MODEL
class AntiOverfitCrashClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # MobileNetV2 (fits better in 6GB than EfficientNet)
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 1280
        
        # Unfreeze last 4 layers
        for i, layer in enumerate(self.features):
            if i < len(self.features) - 4:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        
        # Classifier with heavy dropout
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


print("\n🧠 Building model...")
model = AntiOverfitCrashClassifier(num_classes).to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Trainable params: {trainable:,}")

# TRAINING
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
total_steps = EPOCHS * len(train_loader)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE * 10,
    total_steps=total_steps,
    pct_start=0.1,
    anneal_strategy='cos'
)
scaler = torch.amp.GradScaler('cuda')

print("\n" + "="*60)
print("🚀 TRAINING CONFIGURATION")
print("="*60)
print(f"   ✅ Dropout: 0.7 (classifier), 0.5 (LSTM)")
print(f"   ✅ Weight decay: {WEIGHT_DECAY}")
print(f"   ✅ Label smoothing: {LABEL_SMOOTHING}")
print(f"   ✅ Class balancing + Strong augmentation")
print("="*60 + "\n")

best_acc = 0.0
patience = 20
no_improve = 0

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            out = model(batch_x)
            loss = criterion(out, batch_y)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        train_loss += loss.item()
        train_correct += (out.argmax(1) == batch_y).sum().item()
        train_total += batch_y.size(0)
        
        pbar.set_postfix(loss=f'{train_loss/(pbar.n+1):.3f}', acc=f'{100*train_correct/train_total:.1f}%')
    
    # Validate
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                out = model(batch_x)
            
            val_correct += (out.argmax(1) == batch_y).sum().item()
            val_total += batch_y.size(0)
    
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    gap = (train_acc - val_acc) * 100
    
    gap_status = "✅" if abs(gap) < 5 else ("⚠️" if abs(gap) < 10 else "🚨")
    print(f"Epoch {epoch+1:2d}: Train={100*train_acc:.2f}% | Val={100*val_acc:.2f}% | Gap={gap:+.1f}% {gap_status}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_video_crash_model.pth')
        print(f"         ✅ New best: {100*val_acc:.2f}%")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break

# FINAL TEST
print("\n" + "="*60)
print("🧪 FINAL TESTING")
print("="*60)

model.load_state_dict(torch.load('best_video_crash_model.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        out = model(batch_x)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(batch_y.numpy())

print("\n" + classification_report(all_labels, all_preds, target_names=categories))

test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
print(f"\n📊 FINAL RESULTS:")
print(f"   Best Val Accuracy:  {100*best_acc:.2f}%")
print(f"   Test Accuracy:      {100*test_acc:.2f}%")
print(f"   Improvement:        +{100*(test_acc - 0.48):.2f}% vs 48% baseline")

torch.save(model.state_dict(), 'video_crash_detection_model.pth')
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("\n✅ Training complete!")
