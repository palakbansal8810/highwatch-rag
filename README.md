# Highwatch AI — RAG over Google Drive

A RAG (Retrieval-Augmented Generation) system that connects to Google Drive, indexes your documents, and answers natural language questions grounded in your files.

---

## Architecture

```
Google Drive → Parse → Chunk → Embed → FAISS Index
                                            ↓
User Query  → Embed → Retrieve Top-K → Groq LLM → Answer + Sources
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure environment**
```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/callback
GOOGLE_AUTH_MODE=oauth
```

- Groq API key → [console.groq.com](https://console.groq.com)
- Google credentials → [Google Cloud Console](https://console.cloud.google.com) → APIs & Services → OAuth 2.0 Client ID → add `http://localhost:8000/auth/callback` as redirect URI → enable Google Drive API

**3. Run**
```bash
python main.py
```

API docs at **http://localhost:8000/docs**

---

## Usage

**Authenticate with Google Drive**
```bash
# Get login URL
curl http://localhost:8000/auth/login

# Visit the auth_url in browser — Google redirects back automatically
```

**Sync your Drive**
```bash
curl -X POST http://localhost:8000/sync-drive \
  -H "Content-Type: application/json" \
  -d '{}'

# Poll for progress (sync runs in background)
curl http://localhost:8000/sync-status
```

**Ask questions**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main technical skills mentioned?", "top_k": 5}'
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/auth/login` | Get Google OAuth URL |
| GET | `/auth/callback` | OAuth callback (auto-handled by browser) |
| POST | `/sync-drive` | Start background Drive sync |
| GET | `/sync-status` | Poll sync progress |
| POST | `/ask` | Ask a question over your documents |
| GET | `/status` | System status + index info |
| GET | `/documents` | List indexed documents |
| GET | `/debug/config` | Verify env vars are loaded |

---

## Sample Output

**Request**
```json
POST /ask
{
  "query": "Tell me about the technical skills",
  "top_k": 5
}
```

**Response**
```json
{
  "answer": "Based on the resume, technical skills include:\n• Programming: Python\n• AI/ML: Supervised & Unsupervised Learning, Deep Learning, NLP, GenAI (LLMs, RAG)\n• Libraries: NumPy, Pandas, Scikit-learn, TensorFlow, Hugging Face, LangChain\n• Frameworks & DBs: Flask, FastAPI, MongoDB, ChromaDB, FAISS\n• Tools: Git, Streamlit, AWS EC2, Selenium",
  "sources": ["Palak_Bansal_Resume.pdf", "tts_report.pdf"],
  "chunks_used": 5,
  "model": "llama-3.1-8b-instant",
  "query": "Tell me about the technical skills"
}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Google Drive | OAuth2 via `google-api-python-client` |
| Parsing | `pdfplumber`, `python-docx` |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Store | FAISS (cosine similarity) |
| LLM | Groq (llama-3.1-8b-instant) |
