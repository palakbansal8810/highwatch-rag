# Sample Queries & Outputs

Live results from indexing 9 real documents including a resume, TTS research report, lab manual, internship offer letter, and project documentation.

**System at time of test:**
- Documents indexed: 9
- Total chunks: 63
- Embedding model: all-MiniLM-L6-v2
- LLM: llama-3.1-8b-instant (via Groq)

---

## Query 1 — Question outside document scope

**Request**
```json
POST /ask
{
  "query": "What is Btech Syllabus?",
  "top_k": 5
}
```

**Response**

---

## Query 1 — Factual extraction from resume

**Request**
```json
POST /ask
{
  "query": "Tell me about Palak's technical skills?",
  "top_k": 5
}
```

**Response**
```json
{
  "answer": "Based on the resume, here are Palak's technical skills:\n\n**Programming Languages:** Python, SQL\n\n**AI/Data Science:** Supervised & Unsupervised Learning, Deep Learning, NLP (spaCy, NLTK), GenAI (LLMs, RAG)\n\n**Libraries:** NumPy, Pandas, Scikit-learn, TensorFlow, Hugging Face, LangChain, Langgraph, OpenCV\n\n**Frameworks & Databases:** Flask, FastAPI, MongoDB, ChromaDB, FAISS, GraphDB, Pinecone\n\n**Tools & Platforms:** N8N, Git, GitHub, Streamlit, Gradio, AWS EC2, VS Code, Selenium, BeautifulSoup\n\nAdditionally from tts_report: Unicode Tokenizer, ConvNeXt Text Encoder, Duration Predictor, Joint Style Encoder (EASV), DiT Backbone, CFM/ODE Solver.",
  "sources": [
    "Palak_Bansal_Resume (14).pdf",
    "Reference Document of Idea Abstract Submission.pdf",
    "tts_report 1.pdf",
    "Auth_Platform_Documentation (1).docx",
    "Concept_Note_Safer_Delhi.docx"
  ],
  "chunks_used": 5,
  "model": "llama-3.1-8b-instant",
  "query": "Tell me about Palak's technical skills?"
}
```

>  Retrieved skills from resume and cross-referenced with TTS report for additional technical context.

---

## Query 2 — Deep technical document understanding

**Request**
```json
POST /ask
{
  "query": "What's the main architecture of TTS report?",
  "top_k": 5
}
```

**Response**
```json
{
  "answer": "The TTS system uses a 3-stage group architecture:\n\n**Stage 1 — Text and Language Group:**\n- Unicode Tokenizer: reads input text, maps each character to a number\n- Language Embedding: tags each character with its language\n- ConvNeXt Text Encoder: builds contextual text features\n- Duration Predictor: determines how long each character lasts in audio\n\n**Stage 2 — Speaker and Emotion Group:**\n- Joint Style Encoder (EASV): reads reference audio for speaker identity and emotion\n- Infilling Mask: hides most reference audio during training to prevent overfitting\n\n**Stage 3 — Audio Generation Group:**\n- Fusion Layer: combines text, language, speaker, emotion into one tensor for the DiT\n- DiT Backbone: converts noise to mel spectrogram step by step\n- CFM / ODE Solver: runs 16-step process to clean noise into audio features\n- Vocoder: converts mel spectrogram to real audio at 24kHz\n- Streaming Output: sends audio in small pieces with clean joins\n\nCritical path: Stage 5 → Stage 7 → Stage 8 → Stage 9 → Stage 10 → Stage 11",
  "sources": [
    "tts_report 1.pdf"
  ],
  "chunks_used": 5,
  "model": "llama-3.1-8b-instant",
  "query": "What's the main architecture of TTS report?"
}
```

>  Extracted a detailed multi-stage architecture breakdown from a technical PDF, including component status and critical path.

---

---

## Query 3 — Structured content extraction from a lab manual

**Request**
```json
POST /ask
{
  "query": "What is in SSMDA lab manual?",
  "top_k": 5
}
```

**Response**
```json
{
  "answer": "The SSMDA lab manual is for the course 'Statistics, Statistical Modelling and Data Analytics'. It contains experiments in Scilab, SPSS, and R:
 
- Experiment 01: Basic matrix operations (Scilab)
- Experiment 02: Finding Eigenvalues and eigenvectors (Scilab)
- Experiment 03: Solving equations — Gauss Elimination, Gauss Jordan, Gauss Siedel (Scilab)
- Experiment 06: Plotting functions and finding derivatives
- Experiment 07: Presenting data as a frequency table (SPSS)
- Experiment 10: Scatter diagrams, residual plots, outliers, leverage and influential data points (R)
- Experiment 11: Calculating correlation (R)
- Experiment 12-13: Time series analysis and linear regression (R)
- Experiment 14: Probability and distributions (R)
 
Also includes instructions for creating time series data.",
  "sources": [
    "SSMDA_LAB_MANUAL (2).pdf",
    "tts_report 1.pdf"
  ],
  "chunks_used": 5,
  "model": "llama-3.1-8b-instant",
  "query": "What is in SSMDA lab manual?"
}
```

> Accurately extracted a structured list of 10+ experiments across 3 tools (Scilab, SPSS, R) from a dense academic PDF.
 
---
