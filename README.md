# RAG Proposal Helper

A lightweight Retrieval-Augmented Generation (RAG) service for drafting project proposals from PDF documents, with **chain-of-thought** reasoning for extracting mission-critical facts and *LangGraph* acts as the orchestration layer.

---

## 🚀 Features

* **PDF Ingestion & Chunking**: Splits each PDF into overlapping text chunks for better context coverage.
* **Embeddings & Vector Store**: Uses OpenAI embeddings combined with ChromaDB for efficient similarity search.
* **Chain-of-Thought Fine-Prints Extraction**: LLM reasons step-by-step (chain-of-thought) before summarizing key facts as bullet points per document.
* **LangGraph Agent**: Modular retrieve → answer workflow for clear separation of concerns.
* **FastAPI Endpoints**:

  * `GET /fine-prints`
  * `POST /chat`

---

## 📦 Getting Started

### 1. Clone & Install

```bash
git clone pankajshakya627/rag_proposal_helper
cd rag_proposal_helper

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file in the project root with your OpenAI API key:

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

### 3. Add PDFs

Place your project PDF files into:

```
proposal_rag/data/
```

---

## 🔧 Running the Pipeline

### 1. Ingest & Build Index with Chain-of-Thought

```bash
python ingest.py
```

* **Chunks PDFs** into `CHUNK_SIZE`-sized overlapping segments.
* **Indexes** embeddings in ChromaDB.
* **Extracts fine-prints** per document via chain-of-thought prompts:

  * LLM thinks step-by-step, then produces bullet points.
* **Outputs** results to `proposal_rag/fine_prints.json`.

### 2. Start the API

```bash
uvicorn proposal_rag.main:app --reload
```

* Served at `http://127.0.0.1:8000`
* Auto-reloads on code changes

---

## 🔗 API Endpoints

### `GET /fine-prints`

Returns JSON mapping each PDF name to its chain-of-thought–generated bullet points:

```json
{
  "01A6494.pdf": "- Step-by-step reasoning...\n- Key fact: Access protocols require...",
  "02B7831.pdf": "- Step-by-step reasoning...\n- Key fact: Budget constraints..."
}
```

### `POST /chat`

Interact with the RAG chatbot for ad-hoc questions:

* **Request**:

  ```json
  { "question": "What are the security requirements and site access protocols for IFPQ # 01A6494?" }
  ```
* **Response**:

  ```json
  { "answer": "All visitors must register at security, present ID, sign an NDA, and be escorted on-site after hours." }
  ```

---

## ⚙️ Configuration

All settings live in `proposal_rag/configs.py`:

* `DATA_DIR`, `VECTOR_DIR`: Paths for inputs & vector store.
* `EMBED_MODEL`, `CHAT_MODEL`: OpenAI model names.
* `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`: RAG retrieval parameters.
* Chain-of-thought prompt text located in `proposal_rag/rag/ingest.py`.

Modify and re-run `python ingest.py` to apply changes.

---

## 🛠 Troubleshooting

* **404 on `/fine-prints`**: Ensure ingestion completed successfully and `fine_prints.json` exists.
* **Irrelevant bullet points**: Tweak the CoT prompt in `ingest.py` or increase truncation limit.
* **Poor chat responses**: Increase `TOP_K` in `configs.py`, then re-ingest and restart the API.

---

## 🛠 Further Enhancements

* **Separate CoT reasoning & output calls** for clarity and intermediate inspection.
* **Chunk-level CoT** for extremely large documents.
* **Authentication**: Add API key or OAuth protection to FastAPI.
* **Dockerize**: Containerize the service for scalable deployment.

---

## 👋 Acknowledgements

Built with **FastAPI**, **LangChain**, **LangGraph**, **ChromaDB**, and **OpenAI** 
