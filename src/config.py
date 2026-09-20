"""
Central place to load configuration. Import this instead of calling
os.environ directly elsewhere, so there is exactly one spot that knows
where the .env file lives and what happens when a variable is missing.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Fails loudly (KeyError) if absent -- that is what you want for a key
# your whole system depends on. Silent fallback here just moves the
# failure somewhere confusing later.
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/llama-3.1-8b-instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

CHROMA_PATH = os.getenv("CHROMA_PATH", "./storage/chroma")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./storage/decisions.db")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
