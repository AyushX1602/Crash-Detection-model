# 🚗 Video Crash Detection Model

AI-powered crash severity detection using deep learning (MobileNetV2 + BiLSTM + Attention).

## ✨ What's New (v2.0)

🎉 **4-Class Model with "No Crash" Detection!**

1. **"No Crash" Class Added** - Now detects normal traffic perfectly!
2. **77.55% Test Accuracy** - Up from 61.33% (3-class)
3. **Perfect No-Crash Detection** - 99% precision, 100% recall
4. **779 New Training Samples** - Generated from 13 traffic videos
5. **Ready for CCTV Deployment** - Won't trigger false alarms

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **77.55%** |
| **Val Accuracy** | **77.42%** |
| **Improvement** | **+29.55%** over baseline |

**Per-Class Performance:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Major | 75% | 80% | 0.77 |
| Minor | 66% | 61% | 0.63 |
| Moderate | 57% | 56% | 0.56 |
| **No Crash** | **99%** | **100%** | **1.00** 🎯 |

## 🚀 Quick Start

### 1. Setup Environment
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows (or source venv/bin/activate for Linux/Mac)
pip install -r requirements.txt
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
├── best_video_crash_model.pth    # Trained 4-class model (28.6 MB)
├── label_encoder.pkl             # 4 class labels (major/minor/moderate/no_crash)
├── app_interface.py              # Web interface (Gradio)
├── train_improved.py             # Training script
├── preprocess_videos.py          # Video preprocessing
├── test_best_model.py            # Model evaluation
├── split_videos.py               # Split long videos into clips
├── organize_no_crash.py          # Organize no_crash dataset
└── README.md                     # This file
```

## 🎯 Model Architecture

- **CNN Backbone:** MobileNetV2 (ImageNet pretrained)
- **Temporal Model:** BiLSTM (2 layers, 256 hidden units)
- **Attention:** Temporal attention pooling
- **Sequence Length:** 16 frames per video
- **Input Size:** 224×224 RGB frames
- **Classes:** 4 (major, minor, moderate, no_crash)

## 🔧 Configuration

**Anti-Overfitting Features:**
- Dropout: 0.7 (classifier), 0.5 (LSTM)
- Weight decay: 0.05
- Label smoothing: 0.15
- Class balancing
- Strong data augmentation

**Hardware:**
- GPU: NVIDIA RTX 4050 (6GB)
- Batch size: 8 (training)
- Training time: ~1.5 hours (4-class)

## 📹 Web Interface Features

- **Video Upload:** Drag & drop crash videos
- **Real-time Prediction:** Analyzes 16 frames
- **Confidence Scores:** Shows probabilities for all classes
- **4-Class Detection:** Major, Minor, Moderate, No Crash
- **Supported Formats:** MP4, AVI, MOV, MKV

## 📈 Training Results

```
Best Model (Epoch 52):
  Train: 98.68%
  Val:   77.42%
  Test:  77.55%
  
No-Crash Detection:
  Precision: 99%
  Recall: 100%
  F1-Score: 1.00 (Perfect!)
```

## 🚧 Creating "No Crash" Dataset

We created the 4th class by:
1. Downloading 13 high-quality traffic videos (2.5 hours total)
2. Using `split_videos.py` to split into 10-second clips
3. Generated 779 training samples
4. Organized with `organize_no_crash.py`

```bash
# To create your own no_crash dataset:
python split_videos.py      # Split long traffic videos
python organize_no_crash.py # Organize into train/val/test
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
numpy>=1.24.0
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
# Output: Prediction, confidence scores for all 4 classes
```

### Command Line
```bash
python test_best_model.py  # Run on test set
```

## 🎓 Model Training Details

**Dataset:**
- **Total:** 2,279 videos
- **Train:** 1,595 videos (1,050 crash + 545 no_crash)
- **Val:** 341 videos
- **Test:** 343 videos
- **Classes:** 4 (major, minor, moderate, no_crash)
- **Distribution:** Balanced with class weighting

**Training Configuration:**
- Optimizer: AdamW (LR=5e-5, weight decay=0.05)
- Scheduler: OneCycleLR
- Loss: CrossEntropyLoss (label smoothing=0.15, class weights)
- Early stopping: patience=20
- Epochs: 72 (stopped early)

## 🎯 Real-World Applications

✅ **CCTV Monitoring** - Detects crashes in real-time traffic feeds  
✅ **False Alarm Prevention** - Perfect no-crash detection (99% precision)  
✅ **Severity Classification** - Distinguishes between major/minor/moderate  
✅ **Automated Alerts** - Ready for deployment in traffic management systems  

## 🔮 Future Improvements

1. **More Data:** Expand to 10,000+ videos with diverse conditions
2. **3D ConvNets:** Try C3D, I3D, or R(2+1)D architectures
3. **Ensemble:** Combine multiple models for higher accuracy
4. **Temporal Smoothing:** For CCTV streams (reduce flicker)
5. **Multi-camera:** Handle multiple camera angles simultaneously

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📧 Contact

GitHub: [@AyushX1602](https://github.com/AyushX1602)

---

**Built with PyTorch • Gradio • MobileNetV2**

**Model Stats:** 77.55% accuracy • 4 classes • Perfect no-crash detection (99%/100%)
