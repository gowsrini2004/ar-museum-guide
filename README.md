# AR Museum Guide

> **AI-Powered Artifact Recognition System for Museums**

A complete AR museum guide system with real ML-based artifact recognition using ResNet50. Upload artifact photos, train the model, and get instant recognition on mobile devices.

---

## 🎯 Quick Start

### Starting the System

Run these three commands in **separate terminals**:

```powershell
# Terminal 1 - ML Recognition API
.\venv\Scripts\python.exe backend/ml_api.py

# Terminal 2 - Training/Admin API
.\venv\Scripts\python.exe backend/training_api.py

# Terminal 3 - Frontend Server
.\venv\Scripts\python.exe run_ar_server.py
```

### Access URLs

**From Desktop:**
- Admin Panel: http://localhost:8080/admin_panel.html
- AR Demo: http://localhost:8080/ar_photo_demo.html

**From Phone (Same WiFi):**
- Admin Panel: http://192.168.1.5:8080/admin_panel.html
- AR Demo: http://192.168.1.5:8080/ar_photo_demo.html

> **Note**: Replace `192.168.1.5` with your computer's actual IP address (run `ipconfig` to find it)

---

## ✅ Current Status

- **5 artifacts** trained and ready for recognition
- **ML model** working with 72.73% accuracy
- **Mobile recognition** fully functional
- **55 training images** across 5 classes

---

## 📚 Complete Guide

### 1. Adding New Artifacts

1. Open the **Admin Panel** (http://localhost:8080/admin_panel.html)
2. Fill in artifact details:
   - Name, Category, Period, Origin
   - Description
   - Curator name (optional)
3. Upload **10-30 photos** from different angles
4. Click **"Add Artifact"**

**Tips for Better Recognition:**
- Take photos from multiple angles
- Use good lighting
- Include close-ups and full views
- Minimum 5 images, recommended 10-30

### 2. Training the Model

**When to Train:**
- After adding new artifacts
- When you want to update the model with all current artifacts

**How to Train:**

**Option A - Via Admin Panel:**
1. Open admin panel
2. Click **"Train Model"** button
3. Wait for training to complete (several minutes)

**Option B - Via Command Line:**
```powershell
.\venv\Scripts\python.exe backend/train_model.py
```

**After Training:**
- Restart the ML API server (Terminal 1)
- The new model will be loaded automatically

### 3. Testing AR Recognition

**On Mobile:**
1. Connect phone to **same WiFi** as computer
2. Open: http://YOUR_IP:8080/ar_photo_demo.html
3. Allow camera permissions
4. Take photo of an artifact
5. See instant recognition with details!

**On Desktop:**
1. Open: http://localhost:8080/ar_photo_demo.html
2. Upload or take a photo
3. View recognition results

---

## 🏗️ System Architecture

### File Structure

```
ar-museum-guide/
├── backend/
│   ├── ml_api.py           # ML-based recognition API (Port 8000)
│   ├── training_api.py     # Admin/training API (Port 8001)
│   └── train_model.py      # Model training script
├── frontend/
│   ├── admin_panel.html    # Artifact management UI
│   └── ar_photo_demo.html  # AR recognition demo
├── data/
│   ├── artifacts.json      # Artifact metadata
│   └── training/           # Training images by artifact
├── models/
│   ├── artifact_model.pth  # Trained ResNet50 model (94MB)
│   └── class_mapping.json  # Class ID to artifact mapping
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Technology Stack

- **Backend**: FastAPI, PyTorch, torchvision
- **ML Model**: ResNet50 (fine-tuned)
- **Frontend**: HTML, JavaScript, CSS
- **Image Processing**: PIL (Pillow)

---

## 🔧 Installation & Setup

### Prerequisites

- Python 3.8+
- Windows OS
- Webcam or mobile device with camera

### First Time Setup

1. **Clone or download the project**

2. **Create virtual environment:**
```powershell
python -m venv venv
```

3. **Activate virtual environment:**
```powershell
.\venv\Scripts\activate
```

4. **Install dependencies:**
```powershell
pip install -r requirements.txt
```

5. **Start the servers** (see Quick Start above)

---

## 📊 API Documentation

### ML API (Port 8000)

**POST /predict**
- Upload image for artifact recognition
- Returns: artifact details, confidence score, sources

**GET /health**
- Check API status
- Returns: model loaded status, number of artifacts

### Training API (Port 8001)

**POST /api/artifacts/add**
- Add new artifact with images
- Form data: name, category, period, origin, description, curator, images[]

**GET /api/artifacts/list**
- Get all artifacts
- Returns: array of artifact objects

**POST /api/model/train**
- Train the ML model on all artifacts
- Returns: training results and accuracy

**GET /api/stats**
- Get training statistics
- Returns: total artifacts, images, model status

---

## 🎓 How It Works

### 1. Image Upload & Storage
- Admin uploads photos via web interface
- Images saved to `data/training/artifact_X/`
- Metadata stored in `artifacts.json`

### 2. Model Training
- Uses transfer learning with pre-trained ResNet50
- Fine-tunes on your artifact images
- Saves model to `models/artifact_model.pth`
- Creates class mapping for predictions

### 3. Recognition
- User takes photo on mobile device
- Image sent to ML API
- Model predicts artifact class
- Returns artifact information with confidence

### 4. Display
- Shows artifact name, category, period, origin
- Displays description and verified sources
- Shows confidence percentage

---

## 🚀 Advanced Usage

### Improving Model Accuracy

1. **Add More Images**: 30-50 images per artifact recommended
2. **Diverse Angles**: Capture from all sides
3. **Varied Lighting**: Different lighting conditions
4. **Clean Backgrounds**: Reduce background clutter
5. **Retrain Regularly**: After adding new images

### Retraining the Model

```powershell
# Stop ML API (Ctrl+C in Terminal 1)
.\venv\Scripts\python.exe backend/train_model.py
# Restart ML API
.\venv\Scripts\python.exe backend/ml_api.py
```

### Finding Your Computer's IP

```powershell
ipconfig
```
Look for "IPv4 Address" under your WiFi adapter.

---

## 🐛 Troubleshooting

### "Failed to fetch" error on mobile
- **Solution**: Make sure phone is on same WiFi network
- Check that you're using the correct IP address
- Ensure all servers are running

### Recognition shows wrong artifact
- **Solution**: Model needs retraining with more images
- Add more diverse photos of each artifact
- Retrain the model

### "No module named torchvision"
- **Solution**: Activate virtual environment first
- Run: `.\venv\Scripts\activate`
- Then start servers

### Servers won't start
- **Solution**: Check if ports are already in use
- Make sure virtual environment is activated
- Verify all dependencies are installed

---

## 📝 Current Artifacts

1. **Ponniyin Selvan** - Book (10 images)
2. **Mug** - Book (11 images)
3. **Gospel** - Book (8 images)
4. **LLN** - Book (8 images)
5. **Gospel** - Book (18 images)

**Total**: 5 artifacts, 55 training images

---

## 🔮 Future Enhancements

### Planned Features

1. **Training Status Tracking**
   - Database to track trained vs untrained artifacts
   - Visual indicators in admin panel
   - Incremental training (only new artifacts)

2. **Improved Accuracy**
   - Data augmentation
   - Larger training dataset
   - GPU acceleration

3. **Better Mobile Experience**
   - Real-time camera feed with AR overlay
   - Offline support
   - Progressive Web App (PWA)

4. **Production Deployment**
   - Cloud hosting (AWS, Google Cloud)
   - User authentication
   - Multi-user support
   - Analytics dashboard

---

## 📄 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

- **ResNet50**: Pre-trained model from torchvision
- **FastAPI**: Modern web framework for APIs
- **PyTorch**: Deep learning framework

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review the API documentation
3. Ensure all dependencies are installed correctly

---

**Last Updated**: February 8, 2026

**Version**: 1.0.0

**Status**: ✅ Production Ready
