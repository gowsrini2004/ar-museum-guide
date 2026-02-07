# AR Museum Guide - Minimal Working Prototype

**Final Year Engineering Project**  
AI-Powered Museum Experience with Knowledge Grounding

## 🎯 Project Overview

This system enhances museum visitor experience by combining AI and AR to provide accurate, interactive information about artifacts. The key innovation is **knowledge grounding** - ensuring all AI-generated content comes from curator-verified sources, preventing hallucinations.

### Research Contributions
1. **Fine-grained artifact recognition** with few-shot learning
2. **RAG-based knowledge grounding** for trustworthy AI
3. **AR user experience** design for museums
4. **User study** with real museum visitors

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd F:\PROJECT\ar-museum-guide
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set Up Environment (Optional)
For LLM-based generation, create `.env` file:
```bash
OPENAI_API_KEY=your_api_key_here
```

**Note**: The prototype works without API key using template responses.

### 3. Run the Demo
```bash
streamlit run frontend/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
ar-museum-guide/
├── backend/
│   ├── artifact_recognizer.py    # Recognition simulation
│   └── knowledge_grounder.py     # RAG-based grounding
├── frontend/
│   └── streamlit_app.py          # Demo UI
├── data/
│   └── sample_artifacts.json     # Sample artifact database
├── requirements.txt
└── README.md
```

## 🧪 How It Works

### 1. Artifact Recognition
- **Current**: Simulated recognition with sample artifacts
- **Future**: YOLOv8 detection + Vision Transformer recognition

### 2. Knowledge Grounding (Core Innovation!)
- Retrieves only curator-verified information
- Uses RAG (Retrieval-Augmented Generation)
- Prevents AI hallucinations
- Provides source attribution

### 3. AR Display
- **Current**: Streamlit simulation
- **Future**: Unity AR Foundation mobile app

## 📊 Current Status (Minimal Prototype)

✅ Project structure created  
✅ Sample artifact database (3 artifacts)  
✅ Knowledge grounding system  
✅ Streamlit demo UI  
⏳ Real artifact dataset collection  
⏳ ML model training  
⏳ Mobile AR app  
⏳ User study  

## 🎓 Research Timeline

**Phase 1 (40% - Current)**: 6-8 weeks
- Dataset creation (50-100 artifacts)
- Model training and evaluation
- User study
- Paper draft (60-70%)

**Phase 2 (30%)**: AR integration and scaling

**Phase 3 (30%)**: Publication and deployment

## 📝 Target Conferences

- CHI (Human-Computer Interaction)
- ISMAR (Mixed and Augmented Reality)
- Cultural Heritage Informatics

## 🔧 Next Steps

1. Partner with local museum
2. Collect artifact images and curator descriptions
3. Train recognition models
4. Implement full AR mobile app
5. Conduct user study
6. Write research paper

## 📄 License

Academic research project - Code will be open-sourced upon publication

---

**Author**: Final Year Engineering Student  
**Advisor**: TBD  
**Institution**: TBD
