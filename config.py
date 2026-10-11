"""
Central settings for the local LLM project.
Change PURPOSE / ASSISTANT_NAME / SYSTEM_PROMPT to point the assistant at YOUR use case.
"""
from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"          # put your PDFs / Word / text files here
DB_DIR = BASE_DIR / "chroma_db"       # vector database (created by ingest.py)
NOTES_FILE = BASE_DIR / "notes.md"    # used by the notes MCP server

# ---------- Models (served locally by Ollama) ----------
CHAT_MODEL = "qwen3:4b"               # fallback for slower laptops: "qwen3:1.7b" or "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"

# ---------- RAG settings ----------
COLLECTION = "knowledge"
CHUNK_SIZE = 300        # characters per chunk
CHUNK_OVERLAP = 50        # characters shared between neighbouring chunks
TOP_K = 4                # chunks returned per search (keep small on CPU)
MAX_DISTANCE = 0.8       # cosine distance cut-off; lower = stricter relevance

# ---------- The ONE purpose of this assistant ----------
ASSISTANT_NAME = "IT Helpdesk Assistant"
PURPOSE = "answering questions about our internal IT policies, procedures and how-to guides"

SYSTEM_PROMPT = f"""You are {ASSISTANT_NAME}, a local assistant whose ONLY job is {PURPOSE}.

Rules:
1. For any factual question, call the search_docs tool first.
2. Answer ONLY from what search_docs returns. Mention the source file name(s).
3. If search_docs returns NO_RELEVANT_RESULTS, reply: "I couldn't find that in the documents."
4. If the question is outside your purpose, politely say it's outside your scope.
5. Use save_note only when the user explicitly asks you to save or remember something.
Keep answers short and clear."""
