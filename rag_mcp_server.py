"""
MCP server #1: exposes the RAG knowledge base as tools.
The agent starts this automatically - you don't need to run it yourself.
(Never print() in this file: stdout is the MCP channel.)
"""
import chromadb
import ollama
from chromadb.config import Settings
from mcp.server.fastmcp import FastMCP

import config

mcp = FastMCP("knowledge-base")
_client = chromadb.PersistentClient(path=str(config.DB_DIR),
                                    settings=Settings(anonymized_telemetry=False))


def _collection():
    return _client.get_or_create_collection(config.COLLECTION,
                                            metadata={"hnsw:space": "cosine"})


@mcp.tool()
def search_docs(query: str) -> str:
    """Search the internal document knowledge base. Use this for any question
    about company policies, procedures or guides. Returns relevant excerpts
    with their source file names, or NO_RELEVANT_RESULTS."""
    col = _collection()
    if col.count() == 0:
        return "The knowledge base is empty. Run ingest.py first."
    q = ollama.embed(model=config.EMBED_MODEL, input=["search_query: " + query]).embeddings[0]
    res = col.query(query_embeddings=[q], n_results=min(config.TOP_K, col.count()))

    parts = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if dist <= config.MAX_DISTANCE:
            parts.append(f"[source: {meta['source']} | relevance distance {dist:.2f}]\n{doc}")
    return "\n\n---\n\n".join(parts) if parts else "NO_RELEVANT_RESULTS"


@mcp.tool()
def list_sources() -> str:
    """List the document files currently in the knowledge base."""
    metas = _collection().get(include=["metadatas"])["metadatas"]
    sources = sorted({m["source"] for m in metas})
    return "\n".join(sources) if sources else "The knowledge base is empty."


if __name__ == "__main__":
    mcp.run()  # stdio transport
