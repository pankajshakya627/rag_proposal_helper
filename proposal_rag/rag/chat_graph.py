from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from .retriever import get_retriever
from ..configs import CHAT_MODEL, TEMPERATURE, MAX_TOKENS, OPENAI_API_KEY

# 1) Define state schema
class RAGState(TypedDict, total=False):
    question: str
    docs:     List  # List[Document]
    answer:   str

# 2) Instantiate retriever + LLM
retriever = get_retriever()
llm = ChatOpenAI(
    model_name=CHAT_MODEL,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    openai_api_key=OPENAI_API_KEY,
)

# 3) Node implementations
def retrieve_step(state: RAGState) -> RAGState:
    docs = retriever.invoke(state["question"])
    return {"question": state["question"], "docs": docs}

def answer_step(state: RAGState) -> RAGState:
    ctx = "\n\n---\n\n".join(d.page_content for d in state["docs"])
    messages = [
        SystemMessage(content=(
            "You are a helpful assistant. Use the context; "
            "if fact is missing, say you don't know."
        )),
        HumanMessage(content=f"Context:\n{ctx}\n\nQuestion: {state['question']}")
    ]
    resp = llm.invoke(messages).content
    return {"answer": resp}

# 4) Build the graph
sg = StateGraph(state_schema=RAGState)
sg.add_node("retrieve", retrieve_step)
sg.add_node("respond",  answer_step)

# 5) Wire edges *including* the entrypoint
sg.add_edge(START,    "retrieve")   # ← tells it where to begin
sg.add_edge("retrieve", "respond")
sg.add_edge("respond",  END)

# 6) Compile into a runnable agent
graph = sg.compile()
