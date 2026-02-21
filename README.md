# Aloysia — Agentic Research Assistant

**Aloysia** is a production-ready, agentic research assistant that turns your PDFs, DOCX, and text files into a **page-aware, citation-rich knowledge base**. Powered by **Supabase & LangGraph**, it autonomously decides when to search your documents, compare papers, browse arXiv, or fetch live web data — all returning **academic-grade, page-level citations**.

Built for researchers, clinicians, and students, Aloysia runs on a **Mobile-First / Telegram-First** architecture, letting you query your entire library straight from your phone or desktop.

---

## Architecture

Aloysia uses a **split deployment** for optimal free-tier hosting:

| Layer | What runs there | Technology | Cost |
|:------|:----------------|:-----------|:-----|
| **Telegram Bot** | Always-on polling bot | Fly.io (Free) | $0 |
| **Web Dashboard** | Streamlit research UI | Hugging Face Spaces (Free) | $0 |
| **Shared DB** | Vector search + metadata + RLS | Supabase + pgvector | $0 |

```
┌─────────────────┐     ┌─────────────────┐
│  Telegram Bot   │     │  Streamlit App  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
           ┌─────────────────┐
           │  Vector DB      │
           └─────────────────┘
```

---

## What Aloysia Does

| Feature | Platform | Description |
|:--------|:---------|:------------|
| **Page-level RAG** | Both | Answers with `Source: paper.pdf, Page: 12, Author: WHO` |
| **Agentic Workflow** | Both | Autonomously picks between 10+ tools per query |
| **Mobile Research** | Telegram | Upload, search, summarize papers from your phone |
| **Rich Dashboard** | Web | Batch uploads, compare docs, generate reviews & bibliographies |
| **Account Sync** | Both | `/link your@email.com` connects Telegram to your web workspace |
| **Data Isolation** | Supabase | Row Level Security — you only ever see your own documents |

---

## Project Structure

```
aloysia/
├── code/
│   ├── bot_runner.py       # Standalone bot entry (Fly.io)
│   ├── telegram_bot.py     # Telegram bot handlers
│   ├── streamlit_app.py    # Web dashboard
│   ├── agent.py            # LangGraph graph + tools
│   ├── db.py               # Supabase VectorDB wrapper
│   ├── rag_init.py         # RAG singleton cache
│   ├── app.py              # Document parsing
│   └── export_utils.py     # Export helpers
├── deploy/
│   ├── fly/
│   │   ├── Dockerfile      # Bot container
│   │   └── fly.toml        # Fly.io config
│   └── huggingface/
│       ├── Dockerfile      # Streamlit container
│       └── README.md       # HF Space metadata
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Nago-01/aloysia.git
cd aloysia
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

Required keys:
- `GROQ_API_KEY` or `GEMINI_API_KEY` — LLM provider
- `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` — Vector DB
- `TELEGRAM_BOT_TOKEN` — For bot deployment

### 3. Run Locally
```bash
# Streamlit dashboard
streamlit run code/streamlit_app.py

# Telegram bot (separate terminal)
python -m code.telegram_bot
```

---


## Usage

### Telegram Bot
- `/start` — Onboarding
- `/help` — List all commands
- `/library` — View your indexed documents
- `/link your@email.com` — Sync with your Streamlit workspace
- **Send any PDF/DOCX/TXT** → indexed into your personal library
- **Ask any question** → agent searches your docs with citations

### Web Dashboard
- Sign in with email → isolated workspace
- Upload documents → chunked and embedded to Supabase
- Chat with documents or use **Tools** tab for bulk operations
- Sync with Telegram via `/link` command

---

## Example Queries

| You Ask | Aloysia Does |
|:--------|:-------------|
| `"What does the AMR paper say about resistance mechanisms?"` | `rag_search` → cited answer with page numbers |
| `"Compare AMR and PCOS papers on treatment"` | `compare_documents` → side-by-side analysis |
| `"Show me all my sources"` | `generate_bibliography` → formatted reference list |
| `"Search arXiv for recent LLM papers"` | `arxiv_search` → live academic results |
| `"What's the latest WHO stance on AMR?"` | `web_search` (after user confirmation) |

---

## Tech Stack

| Layer | Technology | Notes |
|:------|:-----------|:-------|
| **Primary LLM** | Groq `llama-3.3-70b-versatile` | Fast inference, tool-calling |
| **Fallback LLM** | Gemini `gemini-2.0-flash` | Activated if Groq fails |
| **Embeddings** | FastEmbed `BAAI/bge-small-en-v1.5` | Quantized ONNX, ~50MB RAM |
| **Vector DB** | Supabase (pgvector) | RLS per `user_id` |
| **Agent Framework** | LangGraph | State machine with QC loop |
| **Document parsing** | PyPDF2, python-docx | Page-aware metadata extraction |
| **Web Search** | Tavily | Real-time web results |
| **Mobile** | python-telegram-bot | Async polling |
| **Dashboard** | Streamlit | Hugging Face Spaces |

---

## License
MIT
