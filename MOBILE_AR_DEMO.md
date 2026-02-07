# 📱 Mobile AR Demo - Quick Start

## What This Does

A **browser-based AR application** that works on your phone! Point your camera at any object and see museum artifact information overlaid in augmented reality.

## 🚀 How to Use

### Step 1: Start the Server

```bash
cd F:\PROJECT\ar-museum-guide
.\venv\Scripts\python.exe run_ar_server.py
```

### Step 2: Find Your Computer's IP Address

**On Windows:**
```bash
ipconfig
```
Look for "IPv4 Address" (e.g., 192.168.1.5)

### Step 3: Open on Your Phone

1. Make sure your phone is on the **same WiFi** as your computer
2. Open browser on your phone
3. Go to: `http://YOUR_IP:8080/ar_mobile_demo.html`
   - Example: `http://192.168.1.5:8080/ar_mobile_demo.html`

### Step 4: Allow Camera Access

When prompted, click "Allow" to let the browser access your camera.

### Step 5: Point and Scan!

Point your phone camera at any object (book, cup, anything) and the AR overlay will appear with artifact information!

## ✨ Features

- ✅ Real camera access (uses your phone's camera)
- ✅ AR-style overlay with artifact information
- ✅ Confidence scores
- ✅ Source attribution (curator-verified)
- ✅ Beautiful UI with glassmorphism
- ✅ Auto-scanning (simulates recognition every few seconds)

## 📸 What You'll See

1. **Camera view** - Live feed from your phone
2. **Scan indicator** - Animated scanning frame
3. **AR Info Card** - Pops up with:
   - Artifact name
   - Category, period, origin
   - Description
   - Confidence score
   - Verified sources

## 🎯 For Your 40% Demo

This shows:
- ✅ Working AR interface on mobile
- ✅ Camera integration
- ✅ Knowledge grounding (shows sources)
- ✅ Real-time overlay
- ✅ Professional UI

Perfect for demonstrating to your advisor/committee!

## 🔧 Troubleshooting

**Can't access from phone?**
- Check both devices are on same WiFi
- Disable firewall temporarily
- Try `http://localhost:8080/ar_mobile_demo.html` on the computer first

**Camera not working?**
- Make sure you allowed camera permissions
- Try using Chrome or Safari on mobile
- Check if other apps can access camera

**Nothing happens when scanning?**
- The demo auto-detects after 2 seconds
- It simulates recognition (no real ML yet)
- Just point at any object and wait

## 📝 Note

This is a **simulation** for the demo. The actual recognition will be implemented with trained models in the full version. For now, it randomly shows one of the 3 sample artifacts to demonstrate the AR experience.

---

**Ready to impress your committee!** 🎓✨
