# Local LLM with RAG + MCP (Windows 11, CPU only)

Everything runs on your laptop. No cloud, no API keys.

| Piece | What it is | File |
|---|---|---|
| Local LLM | `qwen3:4b` served by Ollama | `config.py` |
| RAG | Your docs -> chunks -> local embeddings -> Chroma vector DB | `ingest.py` |
| MCP server 1 | Exposes RAG as tools: `search_docs`, `list_sources` | `rag_mcp_server.py` |
| MCP server 2 | Action tools: `save_note`, `read_notes` | `notes_mcp_server.py` |
| Agent (MCP client) | Connects model + MCP servers, runs tool calls | `agent.py` |
| Specific purpose | Name, purpose and strict rules | `config.py` |

## Step 1 - Install Ollama
Download the Windows installer from https://ollama.com and install it. Check:
```
ollama --version
```

## Step 2 - Pull the models
```
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

## Step 3 - Install Python 3.11
From https://python.org. Tick "Add Python to PATH".

## Step 4 - Set up the project
Unzip this folder (e.g. to C:\local-llm), then in PowerShell:
```
cd C:\local-llm
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Step 5 - Set the purpose
Edit `ASSISTANT_NAME`, `PURPOSE` in `config.py`.

## Step 6 - Add your documents
Put PDF / DOCX / TXT / MD / CSV files into the `docs` folder
(a sample IT policy is included for testing).

## Step 7 - Build the RAG database
```
python ingest.py
```
Re-run whenever documents change.

## Step 8 - Chat
```
python agent.py
```
Try:
- `How long must passwords be?` -> calls search_docs, answers with source
- `What is the capital of France?` -> refuses, out of scope
- `Save a note: renew VPN licence in March` -> calls save_note (MCP server 2)

## Step 9 - Add your own MCP tools
Copy `notes_mcp_server.py`, write functions with `@mcp.tool()`
(SQL query, ticket lookup, SharePoint via Graph...), then add the script to
`SERVERS` in `agent.py`. Keep total tools to ~3-6 for a small model.

## Step 10 (optional, later) - Fine-tune
Only if answer style or tool calling is unreliable: QLoRA with Unsloth on
free Google Colab GPU, export GGUF, load with `ollama create`.

## Tuning
- Slow? `CHAT_MODEL = "qwen3:1.7b"`
- Says "couldn't find" too often? raise `MAX_DISTANCE` to 0.7
- Irrelevant answers? lower `MAX_DISTANCE` to 0.5
- Scanned PDFs have no text - OCR them first.
