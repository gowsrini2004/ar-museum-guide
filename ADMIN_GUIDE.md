# 🎯 Admin Panel - Complete Guide

## What You Can Do

1. **Add new artifacts** with photos
2. **Upload 10+ photos** of each artifact
3. **Train the ML model** on your artifacts
4. **Use trained model** for predictions

## 🚀 Quick Start

### Step 1: Start Training API Server

```bash
cd F:\PROJECT\ar-museum-guide
python backend/training_api.py
```

Server runs on: `http://localhost:8001`

### Step 2: Start Web Server

```bash
python run_ar_server.py
```

### Step 3: Open Admin Panel

Open in browser: `http://localhost:8080/admin_panel.html`

## 📸 Adding Your Book Artifact

### 1. Fill in Details

- **Name**: e.g., "Python Programming Book"
- **Category**: "Book"
- **Period**: "2024"
- **Origin**: "Publisher Name"
- **Description**: Detailed description of the book
- **Curator**: Your name

### 2. Upload Photos

**Take 10-30 photos of your book:**
- Front cover (straight on)
- Front cover (slight angles - left, right)
- Back cover
- Spine
- Different lighting conditions
- Different distances
- Different backgrounds

**Tips for better recognition:**
- Good lighting
- Clear, focused images
- Various angles (5-10° rotations)
- Different distances (close-up, medium, far)
- Avoid blurry photos

### 3. Click "Add Artifact"

Your artifact and photos are saved!

## 🤖 Training the Model

### When to Train

- After adding 2+ artifacts
- After uploading new photos
- When you want to update the model

### How to Train

1. Click **"🚀 Train Model"** button
2. Wait 2-5 minutes (depends on number of images)
3. See training progress and accuracy
4. Model is automatically saved

### What Happens During Training

```
1. Loads all your photos
2. Splits into training (80%) and validation (20%)
3. Fine-tunes ResNet50 on your artifacts
4. Saves best model based on validation accuracy
5. Creates class mapping for predictions
```

## 📊 Understanding the Stats

**Total Artifacts**: Number of artifacts added
**Training Images**: Total photos across all artifacts
**Model Status**: 
- "Not Trained" - Need to train
- "Trained ✅" - Ready to use

## 🎯 Using the Trained Model

### Step 1: Start ML API (with trained model)

```bash
python backend/api_server.py
```

It will automatically load your trained model!

### Step 2: Open Mobile Demo

```
http://192.168.1.5:8080/ar_ml_demo.html
```

### Step 3: Take Photo

Point camera at your book → It recognizes it! 🎉

## 📁 File Structure

```
data/
  training/
    artifact_1/          ← Your book photos
      artifact_1_1.jpg
      artifact_1_2.jpg
      ...
    artifact_2/          ← Another artifact
      ...
  artifacts.json         ← Artifact metadata

models/
  artifact_model.pth     ← Trained model
  class_mapping.json     ← Class names
```

## 🔧 Advanced: Manual Training

If you prefer command line:

```bash
# Make sure photos are in data/training/
python backend/train_model.py
```

## 💡 Tips for Best Results

### Photo Quality
- ✅ 10-30 photos per artifact
- ✅ Various angles and lighting
- ✅ Clear, focused images
- ❌ Avoid very blurry photos
- ❌ Don't use same photo multiple times

### Training
- Start with 2-3 artifacts
- Add more artifacts over time
- Retrain when adding new artifacts
- Aim for >85% validation accuracy

### Recognition
- Test with photos similar to training
- Good lighting helps
- Clear view of artifact
- Similar distance as training photos

## 🎓 For Your 40% Demo

This shows:
1. ✅ **Data collection system** - Admin panel
2. ✅ **Training pipeline** - Automated training
3. ✅ **Real ML model** - ResNet50 fine-tuned
4. ✅ **End-to-end workflow** - Add → Train → Predict
5. ✅ **Production-ready** - Complete system

## 🐛 Troubleshooting

**"Need at least 2 artifacts to train"**
- Add more artifacts first

**"Training failed"**
- Check photos are valid images
- Ensure at least 5 photos per artifact

**"Low accuracy (<70%)"**
- Add more photos per artifact
- Ensure photos are clear and varied
- Train for more epochs

**"Model not loading"**
- Check models/artifact_model.pth exists
- Retrain the model

---

**Ready to build your custom artifact recognition system!** 🚀
