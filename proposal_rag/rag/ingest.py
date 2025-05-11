"""
One-shot pipeline:
  ▸ Load PDFs
  ▸ Chunk + embed → Chroma
  ▸ Extract fine-prints via LLM
Run from repo root:
    PYTHONPATH=. python -m proposal_rag.rag.ingest
"""
import json
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from . import fine_prints
from ..configs import *

splitter   = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                            chunk_overlap=CHUNK_OVERLAP)
embeddings = OpenAIEmbeddings(model=EMBED_MODEL,
                              openai_api_key=OPENAI_API_KEY)

def load_pdfs():
    docs = []
    for pdf in DATA_DIR.glob("*.pdf"):
        pages = PyPDFLoader(str(pdf)).load()
        for p in pages:
            p.metadata["source_doc"] = pdf.name
        docs.extend(pages)
    return docs

def build_chroma(chunks):
    vs = Chroma.from_documents(chunks,
                               embeddings,
                               persist_directory=str(VECTOR_DIR))
    vs.persist()
    return vs

def extract_fine_prints(docs):
    llm = ChatOpenAI(model_name=CHAT_MODEL,
                     temperature=TEMPERATURE,
                     max_tokens=MAX_TOKENS,
                     openai_api_key=OPENAI_API_KEY)
    result = {}
    by_doc = {}
    for d in docs:
        by_doc.setdefault(d.metadata["source_doc"], []).append(d.page_content)

    for name, pages in tqdm(by_doc.items(), desc="Fine-prints"):
        joined = "\n".join(pages)[:12000]  # truncate for cost-safety
        msgs = [
            SystemMessage(content="You are an assistant helping to draft project proposals. "
                                  "Identify and bullet key fine-print facts (deadlines, deliverables, budgets, "
                                  "scope constraints, dependencies, etc.). Return concise bullet points."),
            HumanMessage(content=joined)
        ]
        bullets = llm.invoke(msgs).content.strip()
        result[name] = bullets
    return result

def main():
    DATA_DIR.mkdir(exist_ok=True)
    VECTOR_DIR.mkdir(exist_ok=True)

    docs = load_pdfs()
    if not docs:
        raise RuntimeError(f"No PDFs found in {DATA_DIR}")
    chunks = splitter.split_documents(docs)
    build_chroma(chunks)
    print(f"✅ Indexed {len(chunks)} chunks into Chroma")

    fine_dict = extract_fine_prints(docs)
    fine_prints.save(fine_dict)
    print(f"✅ Saved fine-prints → {FINE_JSON}")

if __name__ == "__main__":
    main()
