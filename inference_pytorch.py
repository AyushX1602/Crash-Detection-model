"""
Video Accident Detection - PyTorch Inference Script
Use this to make predictions on new videos after training
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import pickle
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'video_crash_detection_model.pth'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

SEQUENCE_LENGTH = 16
IMG_SIZE = 224

# ============================================================================
# MODEL ARCHITECTURE (same as training)
# ============================================================================

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=None)
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x


class VideoCrashClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size=128):
        super().__init__()
        self.cnn = CNNEncoder()
        self.cnn_output_size = 1280
        
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.4
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        
        cnn_features = []
        for t in range(seq_len):
            frame_features = self.cnn(x[:, t])
            cnn_features.append(frame_features)
        
        cnn_features = torch.stack(cnn_features, dim=1)
        lstm_out, _ = self.lstm(cnn_features)
        last_hidden = lstm_out[:, -1, :]
        output = self.classifier(last_hidden)
        return output


# ============================================================================
# LOAD MODEL
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load label encoder
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)
print(f"Classes: {list(label_encoder.classes_)}")

# Load model
model = VideoCrashClassifier(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("✅ Model loaded")


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def extract_frames(video_path, sequence_length=SEQUENCE_LENGTH, img_size=IMG_SIZE):
    """Extract frames from video for prediction"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None
    
    frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            frame = frame / 255.0
            frames.append(frame)
        else:
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((img_size, img_size, 3)))
    
    cap.release()
    
    while len(frames) < sequence_length:
        frames.append(frames[-1] if frames else np.zeros((img_size, img_size, 3)))
    
    return np.array(frames[:sequence_length], dtype=np.float32)


def predict_video_crash(video_path):
    """
    Predict accident severity from video
    
    Returns:
        dict with prediction results
    """
    frames = extract_frames(video_path)
    
    if frames is None:
        return {
            'error': 'Could not process video',
            'severity': None,
            'confidence': 0.0
        }
    
    # Convert to tensor: (T, H, W, C) -> (1, T, C, H, W)
    frames_tensor = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0)
    frames_tensor = frames_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(frames_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    predicted_class = probabilities.argmax().item()
    predicted_label = label_encoder.classes_[predicted_class]
    confidence = probabilities[predicted_class].item()
    
    # Get all probabilities
    all_probs = {
        label: prob.item()
        for label, prob in zip(label_encoder.classes_, probabilities)
    }
    
    return {
        'severity': predicted_label,
        'confidence': confidence,
        'all_probabilities': all_probs
    }


def predict_batch(video_paths):
    """Predict multiple videos"""
    results = []
    
    for i, video_path in enumerate(video_paths):
        print(f"Processing {i+1}/{len(video_paths)}: {os.path.basename(video_path)}")
        result = predict_video_crash(video_path)
        result['video_path'] = video_path
        results.append(result)
    
    return results


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("VIDEO CRASH DETECTION - INFERENCE")
    print("="*70)
    
    test_video = "path/to/your/test_video.mp4"  # CHANGE THIS
    
    if os.path.exists(test_video):
        result = predict_video_crash(test_video)
        
        print(f"\nVideo: {test_video}")
        print(f"Predicted Severity: {result['severity']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nAll Probabilities:")
        for severity, prob in result['all_probabilities'].items():
            print(f"  {severity}: {prob:.2%}")
    else:
        print(f"\n⚠️ Test video not found: {test_video}")
        print("Please update 'test_video' path in the script")
