# 🚀 Complete System Test - Quick Start

## All Servers Running

### 1. Training API Server
**Port**: 8001
**URL**: http://localhost:8001
**Purpose**: Add artifacts and train model

### 2. Web Server  
**Port**: 8080
**URL**: http://localhost:8080
**Purpose**: Serve admin panel and mobile demo

### 3. ML API Server (start after training)
**Port**: 8000
**URL**: http://localhost:8000
**Purpose**: Serve predictions

## 📸 Step-by-Step Testing

### Step 1: Add Your Book (Admin Panel)

1. **Open**: http://localhost:8080/admin_panel.html

2. **Fill in details**:
   - Name: Your book name
   - Category: Book
   - Period: 2024
   - Origin: Publisher
   - Description: About the book
   - Curator: Your name

3. **Upload 10 photos** of your book:
   - Drag & drop or click to upload
   - Take from different angles
   - Good lighting

4. **Click "Add Artifact"**

### Step 2: Train the Model

1. In admin panel, click **"🚀 Train Model"**
2. Wait 2-5 minutes
3. See "Model trained successfully!"

### Step 3: Start ML API

Open new terminal:
```bash
cd F:\PROJECT\ar-museum-guide
python backend/api_server.py
```

### Step 4: Test on Mobile

1. **On your phone**, open: http://192.168.1.5:8080/ar_ml_demo.html
2. **Take photo** of your book
3. **See recognition** with AR overlay!

## ✅ What to Test

- [ ] Admin panel loads
- [ ] Can upload photos
- [ ] Artifact is added
- [ ] Training works
- [ ] ML API starts with trained model
- [ ] Mobile demo recognizes your book
- [ ] AR overlay shows correct info

## 🐛 If Something Fails

**Admin panel not loading?**
- Check web server is running on port 8080

**Can't upload photos?**
- Check training API is running on port 8001

**Training fails?**
- Need at least 5 photos
- Check photos are valid images

**Recognition doesn't work?**
- Make sure ML API loaded the trained model
- Check model file exists in models/artifact_model.pth

---

**Ready to test the complete system!** 🎉
