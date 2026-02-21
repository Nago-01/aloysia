---
title: Aloysia - Agentic Research Assistant
emoji: 📚
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: AI-powered research assistant with page-level citations
---

# Aloysia — Agentic Research Assistant

**Aloysia** is a production-ready, agentic research assistant that turns your PDFs, DOCX, and text files into a **page-aware, citation-rich knowledge base**. Powered by **Supabase & LangGraph**, it autonomously decides when to search your documents, compare papers, browse arXiv, or fetch live web data — all returning **academic-grade, page-level citations**.

Built for researchers, clinicians, and students, Aloysia runs on a **Mobile-First / Telegram-First** architecture, letting you query your entire library straight from your phone or desktop.

---

## Features

- **Page-level RAG** — Answers with `Source: paper.pdf, Page: 12, Author: WHO`
- **Agentic Workflow** — Autonomously picks between 10+ tools per query
- **Mobile Research** — Upload, search, summarize papers from Telegram
- **Rich Dashboard** — Batch uploads, compare docs, generate reviews & bibliographies
- **Account Sync** — `/link your@email.com` connects Telegram to your web workspace

---

## How to Use

1. **Sign in** with your email to create your workspace
2. **Upload** PDF, DOCX, or TXT files
3. **Ask questions** about your documents
4. Get answers with **page-level citations**

---

## Sync with Telegram

Link your account using `/link your@email.com` in the [@Aloysia_telegram_bot](https://t.me/Aloysia_telegram_bot)

---

## Tech Stack

| Layer | Technology |
|:------|:-----------|
| LLM | Groq / Gemini |
| Embeddings | FastEmbed |
| Vector DB | Supabase pgvector |
| Agent | LangGraph |
| Frontend | Streamlit |

---

Built by [Nago](https://github.com/Nago-01)
