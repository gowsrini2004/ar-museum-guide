# 🤖 Real ML Model - Setup Guide

## What's New

You now have a **real image recognition model** using ResNet50 transfer learning! This replaces the random simulation with actual ML predictions.

## 🏗️ Architecture

```
Mobile App → FastAPI Server → ResNet50 Model → Predictions
                ↓
         Knowledge Grounder → Curator-Verified Info
```

## 🚀 How to Run

### Step 1: Install ML Dependencies

```bash
cd F:\PROJECT\ar-museum-guide
.\venv\Scripts\activate
pip install torch torchvision pillow fastapi uvicorn python-multipart numpy
```

### Step 2: Start the ML API Server

```bash
python backend/api_server.py
```

You'll see:
```
🤖 Model: ResNet50 Transfer Learning
🌐 API: http://localhost:8000
```

### Step 3: Start the Web Server (in another terminal)

```bash
python run_ar_server.py
```

### Step 4: Open on Your Phone

```
http://192.168.1.5:8080/ar_ml_demo.html
```

## ✨ What It Does

1. **Takes a photo** on your phone
2. **Uploads to ML API** (port 8000)
3. **ResNet50 analyzes** the image
4. **Returns predictions** with confidence scores
5. **Shows AR overlay** with artifact info

## 🎯 Current Model Status

**Model**: ResNet50 pre-trained on ImageNet
**Status**: ⚠️ Not yet fine-tuned on museum artifacts
**Accuracy**: Will improve after training on your artifact images

## 📸 Training Your Own Model

To train on your own artifact images:

### 1. Collect Images

Create this structure:
```
data/
  training/
    artifact_1/
      image1.jpg
      image2.jpg
      ...
    artifact_2/
      image1.jpg
      ...
    artifact_3/
      ...
```

### 2. Run Training Script

```python
from backend.real_recognition_model import ModelTrainer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load your dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder('data/training', transform=transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train
trainer = ModelTrainer(num_classes=len(dataset.classes))

for epoch in range(10):
    loss, acc = trainer.train_epoch(train_loader)
    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.2f}%")

# Save
trainer.save_model('models/artifact_model.pth')
```

### 3. Use Trained Model

Update `api_server.py`:
```python
recognition_model = ArtifactRecognitionModel(
    num_classes=3,
    model_path='models/artifact_model.pth'
)
```

## 🔬 Model Performance

Current (Pre-trained):
- Uses ImageNet features
- Works on general objects
- Not optimized for artifacts

After Training:
- Fine-tuned on your artifacts
- >85% accuracy target
- Distinguishes similar artifacts

## 📊 API Endpoints

### POST /predict
Upload image, get predictions

**Request:**
```
POST http://localhost:8000/predict
Content-Type: multipart/form-data
file: <image>
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "artifact_name": "Ancient Greek Amphora",
    "confidence": 0.92,
    "category": "Pottery",
    ...
  },
  "top_3_predictions": [...],
  "explanation": "...",
  "sources": [...],
  "inference_time_ms": 245
}
```

### GET /health
Check API status

## 🎓 For Your 40% Demo

This shows:
- ✅ Real ML model (ResNet50)
- ✅ Transfer learning approach
- ✅ API integration
- ✅ Mobile interface
- ✅ Knowledge grounding
- ✅ End-to-end pipeline

## 🐛 Troubleshooting

**"Module not found: torch"**
```bash
pip install torch torchvision
```

**"API not responding"**
- Make sure API server is running on port 8000
- Check firewall settings

**"Low confidence scores"**
- Normal for pre-trained model
- Will improve after training on your artifacts

---

**Ready for real ML-powered artifact recognition!** 🚀
