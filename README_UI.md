# Streamlit UI for the local RAG + MCP assistant

Files (put them inside C:\local-llm, next to agent.py; they replace the earlier UI files):

- app.py                 the Streamlit chat page
- agent_core.py          keeps the MCP servers running and talks to the model
- requirements.txt       same as before plus streamlit (and mcp<2)
- .streamlit\config.toml turns off usage stats

## Run it

1. Make sure Ollama is running and the venv is active:
   ```
   cd C:\local-llm
   .venv\Scripts\activate
   ```
2. Install Streamlit once (about 50-100 MB, so allow for a slow connection):
   ```
   pip install --default-timeout=300 --retries 15 streamlit
   ```
3. Start the UI:
   ```
   streamlit run app.py
   ```
   Your browser opens http://localhost:8501. If Streamlit asks for an email
   address in the terminal the first time, just press Enter.
   If `streamlit` is not recognised, use: `python -m streamlit run app.py`

Stop it with Ctrl+C in the terminal. After changing config.py or agent.py, stop
and start it again.

## What you get

- Chat window with history; answers come from your documents and name the source.
- "Tools used" panel under each answer showing the MCP tool calls.
- Sidebar: Thinking mode switch, Show tool calls switch, MCP servers and their tools,
  indexed documents, a document uploader with "Save and re-index", saved notes,
  and Clear chat.
- Example questions you can click on a new chat.

## Thinking mode switch

- OFF (default): fastest. The app tells the model not to reason before answering.
- ON: the model reasons first. The reasoning appears in a collapsible
  "Reasoning" panel above the answer. Slower on a CPU, sometimes more careful.
- The switch is greyed out if the model has no thinking mode (for example llama3.2).
- Some models only come in a "thinking" version (for example the 2507 thinking
  build behind qwen3:4b). They reason even when the switch is OFF. The app hides
  that reasoning from the answer and shows a note under it. To get fast answers,
  choose a model without a thinking mode in config.py, e.g.
  CHAT_MODEL = "qwen3:4b-instruct", "qwen3:1.7b" or "llama3.2:3b"
  (pull it first with: ollama pull <name>).

## Notes

- The terminal running Streamlit prints MCP server log lines; that is normal.
- Conversation memory keeps your questions and the assistant's answers, but drops
  the raw search results and reasoning after each turn, so the context stays small.
- Context window is 8192 tokens. To change it, add NUM_CTX = 4096 (or 16384)
  to config.py.
