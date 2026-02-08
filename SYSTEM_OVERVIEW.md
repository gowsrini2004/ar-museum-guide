# 📱 AR Museum Guide - Complete System Overview

## 🎯 System Architecture (Phone-Only Model)

```
┌─────────────────────────────────────────────────────────┐
│  PHONE (Browser)                                        │
│  ┌──────────────────┐      ┌──────────────────┐       │
│  │  Admin Panel     │      │  AR Demo         │       │
│  │  Add Artifacts   │      │  Take Photo      │       │
│  │  Upload Photos   │      │  See Results     │       │
│  └────────┬─────────┘      └────────┬─────────┘       │
└───────────┼──────────────────────────┼─────────────────┘
            │                          │
            ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│  SERVER (Your PC)                                       │
│  ┌──────────────────┐      ┌──────────────────┐       │
│  │  Training API    │      │  ML API          │       │
│  │  Port 8001       │      │  Port 8000       │       │
│  │  Save Artifacts  │      │  Serve Predictions│      │
│  │  Train Model     │      │  Load Model      │       │
│  └────────┬─────────┘      └────────┬─────────┘       │
│           │                         │                  │
│           ▼                         ▼                  │
│  ┌─────────────────────────────────────────┐          │
│  │  File System                            │          │
│  │  • data/artifacts.json                  │          │
│  │  • data/training/artifact_1/*.jpg       │          │
│  │  • models/artifact_model.pth (trained)  │          │
│  └─────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
```

## ✅ Current Working System

### 3 Servers Running:
1. **Web Server** (port 8080) - Serves HTML pages
2. **Training API** (port 8001) - Manages artifacts & training
3. **ML API** (port 8000) - Serves predictions

### Workflow:

#### Step 1: Add Artifacts (Admin Panel)
```
Phone → http://192.168.1.5:8080/admin_panel.html
1. Fill artifact details
2. Upload 10+ photos
3. Click "Add Artifact"
→ Saves to: data/artifacts.json + data/training/artifact_X/
```

#### Step 2: Train Model
```
Phone → Admin Panel → Click "Train Model"
→ Calls: Training API /api/model/train
→ Trains ResNet50 on your photos
→ Saves to: models/artifact_model.pth
```

#### Step 3: Use Recognition (AR Demo)
```
Phone → http://192.168.1.5:8080/ar_ml_demo.html
1. Take photo of artifact
2. Uploads to ML API
3. Model recognizes artifact
4. Shows AR overlay with info
```

## 📂 File Structure

```
ar-museum-guide/
├── backend/
│   ├── simple_api.py          ✅ ML API (port 8000)
│   ├── training_api.py        ✅ Training API (port 8001)
│   ├── train_model.py         ✅ Training script
│   ├── artifact_recognizer.py ⚠️  OLD (not used)
│   ├── knowledge_grounder.py  ⚠️  Needs OpenAI (optional)
│   ├── real_recognition_model.py ⚠️ Needs PyTorch (optional)
│   └── api_server.py          ⚠️  OLD (not used)
│
├── frontend/
│   ├── admin_panel.html       ✅ Add artifacts
│   ├── ar_ml_demo.html        ✅ Recognition demo
│   ├── ar_photo_demo.html     ✅ Simple AR demo
│   └── streamlit_app.py       ⚠️  OLD (not used)
│
├── data/
│   ├── artifacts.json         ✅ Artifact metadata
│   └── training/              ✅ Training images
│       ├── artifact_1/
│       └── artifact_2/
│
└── models/
    └── artifact_model.pth     ✅ Trained model (after training)
```

## 🚀 Quick Start (Clean Setup)

### Start All Servers:

**Terminal 1: Training API**
```bash
cd F:\PROJECT\ar-museum-guide
python backend/training_api.py
```

**Terminal 2: ML API**
```bash
python backend/simple_api.py
```

**Terminal 3: Web Server**
```bash
python run_ar_server.py
```

### On Phone:

1. **Admin Panel**: `http://192.168.1.5:8080/admin_panel.html`
2. **AR Demo**: `http://192.168.1.5:8080/ar_ml_demo.html`

## 🔧 Current Status

### ✅ Working:
- Web server serving pages
- Training API accepting artifacts
- ML API running (simple mode)
- Admin panel UI
- AR demo UI

### ⚠️ Needs Fixing:
- Artifacts not saving to file yet (need to re-add via admin panel)
- Model training needs PyTorch installed
- Simple API returns random artifact (demo mode)

### 🎯 Next Steps:
1. Re-add your 2 artifacts via admin panel
2. Click "Train Model" (will use real training)
3. ML API will load trained model
4. Test recognition on phone

## 📝 Important Notes

- **Phone-only**: Everything works from phone browser
- **No app install**: Pure web-based
- **Trained model persists**: Saved in models/ folder
- **Reusable**: Once trained, model stays trained
- **Simple mode**: Currently using simple matching (will upgrade to real ML after training)

## 🐛 Known Issues

1. **PyTorch not installed**: Training will fail (need to install)
2. **OpenAI API**: Knowledge grounder needs API key (optional feature)
3. **Artifacts.json missing**: Need to re-add artifacts via admin panel

## ✨ Clean Workflow Summary

```
1. Add artifacts → Training API saves them
2. Train model → Creates artifact_model.pth
3. ML API loads model → Serves predictions
4. Phone takes photo → Gets recognition
5. AR overlay shows info → User sees result
```

---

**System is ready for testing once artifacts are re-added!** 📱
