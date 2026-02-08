"""
Video Crash Detection - Web Interface
Upload a video and get crash severity prediction!
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import pickle
import gradio as gr
from pathlib import Path

# Configuration
SEQUENCE_LENGTH = 16
IMG_SIZE = 224
MODEL_PATH = 'best_video_crash_model.pth'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Architecture (same as training)
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


def extract_frames(video_path, seq_len=SEQUENCE_LENGTH, img_size=IMG_SIZE):
    """Extract evenly spaced frames from video"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None
    
    # Sample evenly spaced frames
    indices = np.linspace(0, total_frames - 1, seq_len, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            frames.append(frame)
        else:
            # Use last frame if read fails
            frames.append(frames[-1] if frames else np.zeros((img_size, img_size, 3), dtype=np.uint8))
    
    cap.release()
    
    # Ensure we have exactly seq_len frames
    while len(frames) < seq_len:
        frames.append(frames[-1])
    
    # Convert to tensor: [T, H, W, C] -> [T, C, H, W]
    frames_array = np.array(frames[:seq_len], dtype=np.float32) / 255.0
    frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2)  # [T, C, H, W]
    
    return frames_tensor.unsqueeze(0)  # [1, T, C, H, W]


def load_model():
    """Load trained model and label encoder"""
    # Load label encoder
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    
    categories = list(label_encoder.classes_)
    num_classes = len(categories)
    
    # Load model
    model = AntiOverfitCrashClassifier(num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    return model, label_encoder, categories


# Load model at startup
print("🔄 Loading model...")
model, label_encoder, categories = load_model()
print(f"✅ Model loaded! Classes: {categories}")


def predict_crash(video_path):
    """
    Main prediction function for Gradio interface
    
    Args:
        video_path: Path to uploaded video file
    
    Returns:
        prediction_text: Formatted prediction result
        confidence_dict: Dictionary of class confidences
    """
    try:
        # Extract frames
        print(f"📹 Processing video: {video_path}")
        frames = extract_frames(video_path)
        
        if frames is None:
            return "❌ Error: Could not read video file", {}
        
        # Run inference
        with torch.no_grad():
            frames = frames.to(device)
            outputs = model(frames)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = outputs.argmax(1).item()
            confidence = probabilities[predicted_class].item()
        
        # Get prediction
        predicted_label = categories[predicted_class]
        
        # Create confidence dictionary for all classes
        confidence_dict = {
            categories[i]: float(probabilities[i].item())
            for i in range(len(categories))
        }
        
        # Format result
        prediction_text = f"""
## 🎯 Prediction Results

**Crash Severity:** {predicted_label.upper()}  
**Confidence:** {confidence*100:.2f}%

### Class Probabilities:
"""
        for cat, prob in confidence_dict.items():
            bar = "█" * int(prob * 20)
            prediction_text += f"\n- **{cat}**: {prob*100:.1f}% {bar}"
        
        # Add interpretation
        prediction_text += f"\n\n### 📊 Interpretation:\n"
        if confidence > 0.7:
            prediction_text += f"✅ **High confidence** - The model is {confidence*100:.1f}% certain this is a **{predicted_label}** crash."
        elif confidence > 0.5:
            prediction_text += f"⚠️ **Moderate confidence** - The model predicts **{predicted_label}** severity with {confidence*100:.1f}% confidence."
        else:
            prediction_text += f"❓ **Low confidence** - The model is uncertain. Top prediction is **{predicted_label}** but review other probabilities."
        
        return prediction_text, confidence_dict
        
    except Exception as e:
        error_msg = f"❌ Error during prediction: {str(e)}"
        print(error_msg)
        return error_msg, {}


# Create Gradio Interface
with gr.Blocks(title="🚗 Crash Detection AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚗 Video Crash Detection System
    
    Upload a video to detect crash severity using AI
    
    **Model Info:**
    - Accuracy: 77.55% (Test)
    - Classes: Major, Minor, Moderate, **No Crash**
    - Frames analyzed: 16 per video
    """)
    
    with gr.Row():
        with gr.Column():
            video_input = gr.File(
                label="📹 Upload Video File",
                file_types=["video"],
                type="filepath"
            )
            
            predict_btn = gr.Button("🔍 Analyze Video", variant="primary", size="lg")
            
            gr.Markdown("""
            ### 📝 Instructions:
            1. Click "Upload Video File" above
            2. Select a crash video (.mp4, .avi, .mov, .mkv)
            3. Click "🔍 Analyze Video"
            4. View crash severity prediction below
            
            **Supported formats:** MP4, AVI, MOV, MKV, WEBM
            
            **Note:** Processing takes ~5-10 seconds per video
            """)
        
        with gr.Column():
            prediction_output = gr.Markdown(label="Prediction Results")
            confidence_plot = gr.Label(label="Confidence Scores", num_top_classes=3)
    
    gr.Markdown("""
    ---
    ### 💡 Tips
    - Use videos under 30 seconds for faster processing
    - Clear/good footage works best
    - Model analyzes 16 evenly-spaced frames
    """)
    
    # Connect prediction
    predict_btn.click(
        fn=predict_crash,
        inputs=video_input,
        outputs=[prediction_output, confidence_plot]
    )
    
    gr.Markdown("""
    ---
    **Model Details:**
    - Architecture: MobileNetV2 + BiLSTM + Attention
    - Training: 2,279 videos, 4 classes (major/minor/moderate/no_crash)
    - Performance: 77.55% test accuracy, 77.42% validation
    - GPU: Optimized for NVIDIA RTX 4050
    - **No Crash Detection:** 99% precision, 100% recall (Perfect!)
    """)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Crash Detection Web Interface")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")
    print(f"Classes: {categories}")
    print("="*60)
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True to get public URL
        show_error=True
    )
