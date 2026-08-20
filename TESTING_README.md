# 🔬 FRAMES Security Testing Lab

This is a **local penetration testing interface** for researching the FRAMES teacher attendance system's face detection and anti-spoofing algorithms.

## ⚠️ Important Notice

This application is for **legitimate security research only** on the official FRAMES APK that you have extracted. It does NOT:
- Connect to any FRAMES backend
- Modify attendance records
- Violate any government systems

## 🚀 Quick Start (Local Testing)

### Prerequisites
- Python 3.11+
- TensorFlow 2.13+
- OpenCV

### Installation

```bash
# Navigate to project directory
cd /path/to/frames-test

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Install dependencies
pip install tensorflow opencv-python numpy pillow streamlit

# Run the local testing app
streamlit run app_local.py
```

### Access the Interface

Open your browser to: **http://localhost:8501**

## 🔍 Testing Modes

### 1. **Single Image Analysis**
- Deep dive into a single image
- See raw model outputs and statistics
- Analyze confidence scores for both models
- Export results as JSON

### 2. **Model Introspection**
- View model architecture
- Check input/output shapes and data types
- Understand model specifications

### 3. **Attack Vector Testing**
Test common bypass techniques:
- Brightness/Contrast manipulation
- Blurring and noise addition
- Face masks/covering
- Print attacks (photos)
- Video replay attacks

### 4. **Batch Testing**
- Test multiple images at once
- Generate CSV reports
- Analyze patterns across many samples

## 🎯 Security Testing Focus

### What to Look For:
1. **Detection Bypass**
   - Can you evade face detection?
   - What scores bypass the threshold?

2. **Anti-Spoofing Bypass**
   - Does the model accept photos?
   - Can adversarial images bypass it?

3. **Edge Cases**
   - Extreme lighting conditions
   - Partial faces
   - Multiple faces
   - No faces
   - Upside-down faces
   - Very blurry images
   - Pixelated images

4. **Threshold Analysis**
   - What scores are accepted?
   - How consistent are the scores?
   - Can you engineer specific scores?

## 📊 Understanding the Outputs

### Face Detection Model
- **Input**: RGB image, normalized to 0-1 or 0-255
- **Output**: Confidence score (higher = more likely face detected)
- **Threshold**: Typically 0.5 (50% confidence)

### Anti-Spoofing Model
- **Input**: RGB image, normalized to 0-1 or 0-255
- **Output**: Liveness score (higher = more likely real face)
- **Threshold**: Typically 0.5 (50% confidence)

## 🛠️ Advanced Usage

### Raw Model Output Mode
Enable in sidebar to see:
- Full output arrays
- Flattened values
- Statistical distributions (min, max, mean, std)

### Custom Thresholds
Adjust detection thresholds in sidebar to:
- Test threshold sensitivity
- Find edge cases
- Analyze score distributions

## 📝 Research Findings Template

When you find vulnerabilities, document:

```json
{
  "vulnerability": "Description of the attack",
  "attack_type": "e.g., Brightness Manipulation",
  "success_rate": "How often does it work?",
  "requirements": "What is needed to perform the attack?",
  "impact": "What access does this give?",
  "remediation": "How can this be fixed?"
}
```

## ⚖️ Ethical Guidelines

This research is permitted for:
- ✅ Finding and reporting vulnerabilities responsibly
- ✅ Improving FRAMES security
- ✅ Academic and educational purposes
- ✅ Authorized security testing

This research is NOT for:
- ❌ Bypassing attendance for personal gain
- ❌ Manipulating other teachers' records
- ❌ Illegal access to government systems
- ❌ Commercial exploitation

## 📚 Files

- `app_local.py` - Advanced security testing interface (local only)
- `app.py` - Streamlit Cloud version (limited functionality)
- `models/face_model.tflite` - Face detection model
- `models/FaceAntiSpoofing.tflite` - Liveness detection model
- `requirements.txt` - Python dependencies

## 🔗 Resources

- [TensorFlow Lite Documentation](https://www.tensorflow.org/lite)
- [Adversarial ML Research](https://adversarial.ml/)
- [Face Recognition Security](https://cvpr2021.thecvf.com/)

---

**Created for legitimate security research on the FRAMES teacher attendance system.**
