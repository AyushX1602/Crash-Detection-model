# 🚀 CLIP Crash Detection - Installation Guide

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for first-time model download)
- (Optional) CUDA-capable GPU for faster inference

---

## ⚡ Quick Installation

### **Step 1: Install Dependencies**

```bash
pip install -r requirements_clip.txt
```

**OR install individually:**

```bash
pip install torch torchvision transformers pillow fastapi uvicorn python-multipart
```

---

### **Step 2: First Run (Auto-downloads Model)**

```bash
cd image_model
python crash_detector_clip.py crash.jpg
```

**⚠️ IMPORTANT:** The CLIP model (605MB) is **NOT included** in this repository due to GitHub's 100MB file limit.

**First run will automatically download the model:**

```
🔥 Initializing CLIP Crash Detector...
   Device: cpu
   Downloading from Hugging Face: openai/clip-vit-base-patch32
   (First time only - ~605MB)
   
Downloading pytorch_model.bin: 100%|██████| 605MB/605MB [02:30<00:00]
   
✅ CLIP detector ready!

🎯 PREDICTION: CRASH
   Crash Probability: 96.6%
   ✅ HIGH CONFIDENCE - Auto-verify recommended
```

**Download time:** ~2-3 minutes (one-time only)

**Subsequent runs:** Instant! Model cached at `C:\Users\<you>\.cache\huggingface\`

---

### **Step 3: Verify It Works**

```bash
# Should run instantly now (model is cached)
python crash_detector_clip.py download.jpg
```

**Expected:** 
```
🔥 Loading from cached model
🎯 PREDICTION: NORMAL
```

---

## 📦 Model Download (Automatic)

**⚠️ The CLIP model is NOT bundled in this repository** (too large for GitHub - 605MB exceeds 100MB limit).

**How it works:**
1. On **first run**, the model auto-downloads from Hugging Face (~605MB)
2. Model is cached locally at: `C:\Users\<username>\.cache\huggingface\hub\`
3. **All subsequent runs** use the cached model (instant load!)

**Model Details:**
- **Name:** openai/clip-vit-base-patch32
- **Size:** ~605MB
- **Source:** Hugging Face (official OpenAI release)
- **Format:** PyTorch (.bin)
- **License:** MIT

**Benefits:**
- ✅ Always get the latest official model
- ✅ No extra download steps needed
- ✅ Automatic caching system
- ✅ Smaller repository clone (~50MB vs 650MB)

---

## 🔧 Installation Options

### **Option 1: Standard Installation (Recommended)**

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd crash-detection

# 2. Install Python dependencies
pip install -r image_model/requirements_clip.txt

# 3. First run - auto-downloads model (~605MB, ~2-3 min)
cd image_model
python crash_detector_clip.py crash.jpg

# Output:
#   Downloading from Hugging Face... (first time only)
#   🎯 PREDICTION: CRASH (96.6% confidence)

# 4. Subsequent runs - instant!
python crash_detector_clip.py download.jpg
```

---

### **Option 2: Download Model from Hugging Face**

If the bundled model is missing, it will auto-download:

```bash
# Run any script - model downloads automatically
python crash_detector_clip.py test.jpg

# Downloaded to: C:\Users\<username>\.cache\huggingface\hub\
```

---

## 🐳 Docker Installation (Optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY image_model/requirements_clip.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements_clip.txt

# Copy application
COPY image_model/ .

# Expose API port
EXPOSE 8000

# Run API server
CMD ["python", "api_clip.py"]
```

**Build and run:**
```bash
docker build -t crash-detection-clip .
docker run -p 8000:8000 crash-detection-clip
```

---

## 🧪 Verify Everything Works

### **Test 1: Single Image Detection**

```bash
python crash_detector_clip.py your_image.jpg
```

### **Test 2: Run Test Suite**

```bash
python test_clip.py
```

### **Test 3: Start API Server**

```bash
python api_clip.py
# Visit: http://localhost:8000/docs
```

---

## ⚙️ GPU Acceleration (Optional)

**If you have NVIDIA GPU:**

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# The system will automatically use GPU if available
```

**Verify GPU is being used:**
```python
import torch
print(torch.cuda.is_available())  # Should print: True
```

---

## 📊 System Requirements

### **Minimum:**
- CPU: Dual-core 2GHz+
- RAM: 4GB
- Storage: 2GB free
- OS: Windows 10/11, Linux, macOS

### **Recommended:**
- CPU: Quad-core 3GHz+
- RAM: 8GB+
- GPU: NVIDIA GPU with 4GB+ VRAM
- Storage: 5GB free

---

## 🐛 Troubleshooting

### **"No module named transformers"**
```bash
pip install transformers
```

### **"CUDA out of memory"**
```python
# In crash_detector_clip.py, force CPU:
device = 'cpu'  # Change from 'auto'
```

### **Slow inference (>2 seconds)**
- Use GPU if available
- Reduce image size
- Check system resources

### **Model not found**
```bash
# Delete and let it re-download:
rm -rf clip_model/
python crash_detector_clip.py test.jpg
```

---

## ✅ Post-Installation Checklist

- [ ] Dependencies installed (`pip list` shows transformers, torch, etc.)
- [ ] Test script runs successfully
- [ ] Model loads without errors
- [ ] Inference time < 1 second (CPU) or < 200ms (GPU)
- [ ] API server starts (if using api_clip.py)

---

## 🆘 Support

**Issues?** Check:
1. Python version: `python --version` (needs 3.8+)
2. Dependencies: `pip list | grep -E "torch|transformers"`
3. Disk space: Ensure 2GB+ free
4. Firewall: Allow Python if using API

---

## 🎯 Next Steps

After installation:
1. Read `USAGE_GUIDE.md` for integration examples
2. Test on your own images
3. Integrate with SOS system
4. Deploy to production

**Ready to detect crashes!** 🚀
