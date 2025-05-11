import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR       = Path(__file__).resolve().parent
DATA_DIR       = BASE_DIR / "data"
VECTOR_DIR     = BASE_DIR / "vector_store"
FINE_JSON      = BASE_DIR / "fine_prints.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBED_MODEL    = "text-embedding-3-large"
CHAT_MODEL     = "gpt-4.1-nano"
TEMPERATURE    = 0.0
MAX_TOKENS     = 1024

CHUNK_SIZE     = 1024
CHUNK_OVERLAP  = 128
TOP_K          = 4
