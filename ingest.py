"""
Step: build the RAG knowledge base.
Reads every PDF / DOCX / TXT / MD / CSV in ./docs, splits it into chunks,
embeds them locally with Ollama, and stores them in a local Chroma database.
Re-run this whenever you add or change documents (it rebuilds from scratch).
"""
from pathlib import Path

import chromadb
import ollama
from chromadb.config import Settings
from docx import Document
from pypdf import PdfReader

import config

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".csv"}


def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == ".docx":
        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def embed(texts: list[str]) -> list[list[float]]:
    # nomic-embed-text expects this prefix for stored documents
    return ollama.embed(model=config.EMBED_MODEL,
                        input=["search_document: " + t for t in texts]).embeddings


def main() -> None:
    config.DOCS_DIR.mkdir(exist_ok=True)
    files = [p for p in config.DOCS_DIR.rglob("*") if p.suffix.lower() in SUPPORTED]
    if not files:
        print(f"No documents found. Put files in: {config.DOCS_DIR}")
        return

    client = chromadb.PersistentClient(path=str(config.DB_DIR),
                                       settings=Settings(anonymized_telemetry=False))
    try:
        client.delete_collection(config.COLLECTION)
    except Exception:
        pass
    col = client.create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})

    total = 0
    for path in files:
        try:
            chunks = chunk_text(read_file(path), config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        except Exception as e:
            print(f"  ! Skipped {path.name}: {e}")
            continue
        if not chunks:
            print(f"  ! No text in {path.name} (scanned PDF?)")
            continue
        rel = str(path.relative_to(config.DOCS_DIR))
        for i in range(0, len(chunks), 32):              # embed in batches
            batch = chunks[i:i + 32]
            col.add(
                ids=[f"{rel}::{i + j}" for j in range(len(batch))],
                documents=batch,
                embeddings=embed(batch),
                metadatas=[{"source": rel, "chunk": i + j} for j in range(len(batch))],
            )
        total += len(chunks)
        print(f"  + {rel}: {len(chunks)} chunks")

    print(f"\nDone. {len(files)} file(s), {total} chunks stored in {config.DB_DIR}")


if __name__ == "__main__":
    main()
