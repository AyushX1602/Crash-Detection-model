# 🚗 Video Crash Detection Model

AI-powered crash severity detection using deep learning (MobileNetV2 + BiLSTM + Attention).

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 61.33% |
| **Val Accuracy** | 62.67% |
| **Improvement** | +13.33% over baseline |

**Per-Class Performance:**
- Major: 72% precision
- Minor: 58% precision  
- Moderate: 55% precision

## 🚀 Quick Start

### 1. Setup Environment
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install torch torchvision opencv-python scikit-learn tqdm gradio
```

### 2. Run Web Interface
```bash
python app_interface.py
```
Open http://localhost:7860 and upload crash videos!

### 3. Train Your Own Model
```bash
# Preprocess videos
python preprocess_videos.py

# Train model
python train_improved.py

# Evaluate
python test_best_model.py
```

## 📁 Project Structure

```
├── best_video_crash_model.pth    # Trained model (28.6 MB)
├── label_encoder.pkl             # Class labels
├── app_interface.py              # Web interface (Gradio)
├── train_improved.py             # Training script
├── preprocess_videos.py          # Video preprocessing
├── test_best_model.py            # Model evaluation
└── README.md                     # This file
```

## 🎯 Model Architecture

- **CNN Backbone:** MobileNetV2 (ImageNet pretrained)
- **Temporal Model:** BiLSTM (2 layers, 256 hidden units)
- **Attention:** Temporal attention pooling
- **Sequence Length:** 16 frames per video
- **Input Size:** 224×224 RGB frames

## 🔧 Configuration

**Anti-Overfitting Features:**
- Dropout: 0.7 (classifier), 0.5 (LSTM)
- Weight decay: 0.05
- Label smoothing: 0.15
- Class balancing
- Strong data augmentation

**Hardware:**
- GPU: NVIDIA RTX 4050 (6GB)
- Batch size: 8
- Training time: ~30 minutes

## 📹 Web Interface Features

- **Video Upload:** Drag & drop crash videos
- **Real-time Prediction:** Analyzes 16 frames
- **Confidence Scores:** Shows probabilities for all classes
- **Supported Formats:** MP4, AVI, MOV, MKV

## 📈 Training Results

```
Epoch 26 (Best):
  Train: 88.36%
  Val:   62.67%
  Gap:   25.7% (overfitting present)
```

## 🛠️ Requirements

```
python>=3.11
torch>=2.1.0
torchvision>=0.16.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
gradio>=4.0.0
tqdm>=4.66.0
```

## 📝 Usage Examples

### Python API
```python
import torch
from app_interface import load_model, extract_frames, predict_crash

# Load model
model, label_encoder, categories = load_model()

# Predict on video
result, confidence = predict_crash("path/to/video.mp4")
print(result)
```

### Command Line
```bash
python test_best_model.py  # Run on test set
```

## 🎓 Model Training Details

**Dataset:**
- 1500 videos (1050 train, 225 val, 225 test)
- 3 classes: major, minor, moderate
- Balanced distribution

**Training Configuration:**
- Optimizer: AdamW (LR=5e-5, weight decay=0.05)
- Scheduler: OneCycleLR
- Loss: CrossEntropyLoss (label smoothing=0.15)
- Early stopping: patience=20

## 🚧 Known Limitations

- Overfitting persists despite heavy regularization
- Accuracy below ideal target (61% vs 65-72% goal)
- Limited by 6GB GPU memory (SEQUENCE_LENGTH=16 max)

## 🔮 Future Improvements

1. **More Data:** Expand dataset to 5000+ videos
2. **3D ConvNets:** Try C3D, I3D, or R(2+1)D architectures
3. **Ensemble:** Combine multiple models
4. **Better Hardware:** Use 12GB+ GPU for longer sequences
5. **Transfer Learning:** Pre-train on Kinetics-400/700

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📧 Contact

GitHub: [@AyushX1602](https://github.com/AyushX1602)

---

**Built with PyTorch • Gradio • MobileNetV2**
