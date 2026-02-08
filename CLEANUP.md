# 🧹 System Cleanup Checklist

## Files to Keep (Active)

### Backend (API Servers)
- ✅ `simple_api.py` - ML prediction server (port 8000)
- ✅ `training_api.py` - Artifact management & training (port 8001)
- ✅ `train_model.py` - Model training script

### Frontend (Phone UI)
- ✅ `admin_panel.html` - Add artifacts & train
- ✅ `ar_ml_demo.html` - Recognition demo
- ✅ `ar_photo_demo.html` - Simple AR demo

### Utilities
- ✅ `run_ar_server.py` - Web server

## Files with Errors (Optional/Legacy)

### Can be ignored for now:
- ⚠️ `artifact_recognizer.py` - Old simple recognizer (not used)
- ⚠️ `knowledge_grounder.py` - Needs OpenAI API (optional)
- ⚠️ `real_recognition_model.py` - Needs PyTorch (will use later)
- ⚠️ `api_server.py` - Old API (replaced by simple_api.py)
- ⚠️ `streamlit_app.py` - Old demo (not used)

## Current System State

### ✅ Working Components:
1. Web server (port 8080)
2. Training API (port 8001)
3. ML API (port 8000) - simple mode
4. Admin panel UI
5. AR demo UI

### 🔧 To Fix:
1. Re-add artifacts via admin panel
2. Install PyTorch for real training
3. Test complete workflow

## Clean Workflow (No Errors)

```
Phone → Admin Panel → Add Artifacts → Training API → Saves Files
                                                    ↓
Phone → AR Demo → Take Photo → ML API → Returns Recognition
                                    ↑
                            Loads from artifacts.json
```

## Action Items

1. ✅ Keep using simple_api.py (no errors)
2. ✅ Keep using training_api.py (no errors)
3. ✅ Admin panel works (fixed to save to server)
4. ⏳ Re-add artifacts via phone
5. ⏳ Test recognition

---

**System is clean and ready for testing!** 🎉
