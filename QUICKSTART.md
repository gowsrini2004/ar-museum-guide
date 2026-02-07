# Quick Start Guide

## 🚀 Run the Prototype in 3 Steps

### Step 1: Install Dependencies
```bash
cd F:\PROJECT\ar-museum-guide
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Run the Demo
```bash
streamlit run frontend/streamlit_app.py
```

### Step 3: Try It Out
1. Browser opens automatically at `http://localhost:8501`
2. Click **"Simulate Artifact Scan"** button
3. See the AI pipeline in action:
   - Artifact recognition
   - Knowledge grounding (no hallucinations!)
   - AR-style display with source attribution

## 🎯 What This Demonstrates

✅ **Core AI Pipeline**: Detection → Recognition → Grounding → Display  
✅ **Knowledge Grounding**: RAG-based system prevents hallucinations  
✅ **Source Attribution**: All information traced to curator sources  
✅ **AR Simulation**: Shows how info will appear in AR overlay  

## 🔧 Optional: Add OpenAI API

For LLM-based generation (optional):
1. Copy `.env.example` to `.env`
2. Add your OpenAI API key
3. Restart the app

**Note**: Works without API key using template responses!

## 📊 Sample Artifacts Included

1. **Ancient Greek Amphora** (500 BCE)
2. **Egyptian Scarab Amulet** (1500 BCE)
3. **Ming Dynasty Vase** (1450 CE)

Each has curator-verified knowledge entries.

## 🎓 Research Highlights

This minimal prototype demonstrates the **key research contribution**:
- **Knowledge Grounding** prevents AI hallucinations
- All information comes from verified museum sources
- Critical for trustworthy AI in cultural heritage

## ❓ Troubleshooting

**Import errors?**
```bash
# Make sure you're in the right directory
cd F:\PROJECT\ar-museum-guide
# Activate virtual environment
venv\Scripts\activate
```

**Port already in use?**
```bash
streamlit run frontend/streamlit_app.py --server.port 8502
```
