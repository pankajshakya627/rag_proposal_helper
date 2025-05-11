# RAG Proposal Helper 📄➜🤖

A lightweight Retrieval-Augmented Generation (RAG) service for drafting project proposals from PDF documents. It provides:

- **Fine-prints extraction**: Key project details (deadlines, deliverables, budgets, constraints) as concise bullet points.  
- **Chat interface**: Ask natural-language questions and get answers grounded in your PDFs.

---

## 🚀 Features

- **PDF ingestion & chunking**: Splits each PDF into overlapping text chunks.  
- **Embeddings & vector store**: Uses OpenAI embeddings + Chroma for similarity search.  
- **Fine-print extraction**: One-shot LLM prompts to extract mission-critical facts per document.  
- **LangGraph agent**: Modular retrieve → answer workflow.  
- **FastAPI endpoints**:  
  - `GET  /fine-prints`  
  - `POST /chat`  

---

## 📦 Getting Started

### 1. Clone & install

```bash
git clone pankajshakya627/rag_proposal_helper
cd rag_proposal_helper

python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
