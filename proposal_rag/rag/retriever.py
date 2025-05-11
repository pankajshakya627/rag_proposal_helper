from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma                   # ← new import
from ..configs import *

embeddings = OpenAIEmbeddings(model=EMBED_MODEL,
                              openai_api_key=OPENAI_API_KEY)

def get_retriever():
    vs = Chroma(persist_directory=str(VECTOR_DIR),
                embedding_function=embeddings)
    return vs.as_retriever(search_kwargs={"k": TOP_K})
