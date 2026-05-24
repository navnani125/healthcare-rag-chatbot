# 🏥 Healthcare Medical Q&A Chatbot (RAG)

> A Retrieval-Augmented Generation (RAG) chatbot that answers medical questions
> using verified data from **PubMedQA** and **MedQuAD (NIH)**, powered by
> **LLaMA3-70b** via Groq, **ChromaDB** for vector search, and **sentence-transformers**
> for embeddings.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

<!-- Add a demo GIF here: ![Demo](assets/demo.gif) -->

---

## 🚀 Live Demo

👉 **[Try it on HuggingFace Spaces](https://huggingface.co/spaces/YOUR_USERNAME/healthcare-rag-chatbot)**

---

## 🧠 Architecture

```
User Question
     │
     ▼
[Embedding Model]          sentence-transformers/all-MiniLM-L6-v2
     │
     ▼
[Vector Search]            ChromaDB — cosine similarity (top-5 chunks)
     │
     ▼
[Context Assembly]         Retrieved chunks formatted with source tags
     │
     ▼
[LLM Generation]           LLaMA3-70b via Groq API (free tier)
     │
     ▼
[Answer + Sources]         Streamlit chat UI
```

## 📊 Data Sources

| Source     | Records   | Domain                        |
|------------|-----------|-------------------------------|
| PubMedQA   | ~1,000    | Biomedical research Q&A       |
| MedQuAD    | ~500      | NIH disease & drug Q&A        |

**Total:** ~1,500 documents → ~3,000+ chunks after splitting

---

## 🛠️ Tech Stack

| Component       | Tool                                    |
|-----------------|-----------------------------------------|
| Orchestration   | LangChain                               |
| Embeddings      | sentence-transformers (all-MiniLM-L6-v2)|
| Vector Store    | ChromaDB (local persistent)             |
| LLM             | LLaMA3-70b via Groq (free)             |
| UI              | Streamlit                               |
| Deployment      | HuggingFace Spaces                      |

---

## 📁 Project Structure

```
healthcare-rag-chatbot/
├── src/
│   ├── ingest.py        # Load & chunk PubMedQA + MedQuAD
│   ├── embeddings.py    # Embed chunks → ChromaDB
│   ├── retriever.py     # Semantic search
│   ├── llm.py           # Groq LLaMA3 generation
│   └── pipeline.py      # Full RAG chain
├── app/
│   └── app.py           # Streamlit chat UI
├── data/                # Auto-created: chunks.json + chroma_db/
├── notebooks/           # EDA + experiments
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone and install
```bash
git clone https://github.com/YOUR_USERNAME/healthcare-rag-chatbot.git
cd healthcare-rag-chatbot
pip install -r requirements.txt
```

### 2. Get a free Groq API key
Sign up at [console.groq.com](https://console.groq.com) — free, no credit card needed.

```bash
export GROQ_API_KEY=your_key_here
# or create a .env file:
echo "GROQ_API_KEY=your_key_here" > .env
```

### 3. Build the knowledge base (one-time setup)
```bash
# Step 1: Download and chunk the medical data
python src/ingest.py

# Step 2: Embed chunks into ChromaDB
python src/embeddings.py
```

### 4. Run the chatbot
```bash
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔍 Features

- **Multi-turn conversation** — follow-up questions maintain context
- **Source transparency** — toggle to see which chunks were retrieved
- **Relevance scores** — each retrieved source shows similarity score
- **Sidebar quick-questions** — example questions to get started
- **Medical disclaimer** — responsible AI, clearly marked as educational

---

## ⚠️ Disclaimer

This chatbot is for **educational and research purposes only**.
It is NOT a substitute for professional medical advice, diagnosis, or treatment.
Always consult a qualified healthcare provider for medical concerns.

---

## 📄 License

MIT — free to use, modify, and share.
