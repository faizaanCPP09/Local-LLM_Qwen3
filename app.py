"""
Streamlit chat UI for the local RAG + MCP assistant.
Run:  streamlit run app.py
"""
import contextlib
import io
import json
from pathlib import Path

import streamlit as st

import config
import ingest
from agent_core import Backend

st.set_page_config(page_title=config.ASSISTANT_NAME, page_icon=":speech_balloon:", layout="centered")

EXAMPLES = [
    "How long must passwords be?",
    "What should I do if my laptop is stolen?",
    "Which documents do you have?",
]


@st.cache_resource(show_spinner="Starting the local model connection and MCP servers...")
def get_backend() -> Backend:
    backend = Backend()
    backend.start()
    return backend


try:
    backend = get_backend()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not start the assistant: {e}")
    st.markdown(
        "Check that Ollama is running (`ollama list`), that the chat model and "
        "`nomic-embed-text` are installed, and that `python rag_mcp_server.py` "
        "starts without errors."
    )
    st.stop()


def new_chat() -> None:
    st.session_state.messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    st.session_state.ui = []


if "messages" not in st.session_state:
    new_chat()
    st.session_state.up_key = 0
    st.session_state.index_log = ""


def render_assistant(m: dict, show_tools: bool) -> None:
    """Draw one assistant message: reasoning panel, answer, then tool panel."""
    if m.get("reasoning"):
        with st.expander("Reasoning"):
            st.markdown(m["reasoning"])
    st.markdown(m["content"])
    if m.get("leaked"):
        st.caption(
            "This model reasons even with Thinking mode off, so it can't be switched off here. "
            "For faster answers, use a model without a thinking mode (see CHAT_MODEL in config.py)."
        )
    events = m.get("tools") or []
    if show_tools and events:
        with st.expander(f"Tools used ({len(events)})"):
            for e in events:
                st.markdown(f"**{e['tool']}**  `{json.dumps(e['args'], ensure_ascii=False)}`")
                out = e["output"]
                st.code(out if len(out) <= 1500 else out[:1500] + "\n... (truncated)", language=None)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header(config.ASSISTANT_NAME)
    st.caption(f"Chat model: {config.CHAT_MODEL}  \nEmbeddings: {config.EMBED_MODEL}  \nRuns fully on this computer.")

    thinking = st.toggle(
        "Thinking mode", value=False, disabled=not backend.thinking_capable,
        help="When on, the model reasons step by step before answering. "
             "Answers may be more careful, but they take much longer on a CPU.")
    if not backend.thinking_capable:
        st.caption("This model has no thinking mode.")
    show_tools = st.toggle("Show tool calls", value=True)

    st.subheader("MCP servers")
    for name, tool_names in backend.servers.items():
        st.markdown(f"- **{name}**: {', '.join(tool_names)}")

    st.subheader("Knowledge base")
    try:
        sources = backend.call_tool("list_sources")
    except Exception:  # noqa: BLE001
        sources = "(could not read the list)"
    if "empty" in sources.lower() or sources.startswith("("):
        st.caption(sources)
    else:
        for line in sources.splitlines():
            st.markdown(f"- {line}")

    files = st.file_uploader(
        "Add documents", type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    if st.button("Save and re-index", use_container_width=True):
        config.DOCS_DIR.mkdir(exist_ok=True)
        for f in files or []:
            (config.DOCS_DIR / Path(f.name).name).write_bytes(f.getbuffer())
        buf = io.StringIO()
        with st.spinner("Indexing documents..."):
            backend.stop()   # release the database before it is rebuilt
            try:
                with contextlib.redirect_stdout(buf):
                    ingest.main()
            except Exception as e:  # noqa: BLE001
                buf.write(f"\nIndexing failed: {e}")
            finally:
                backend.start()
        st.session_state.index_log = buf.getvalue().strip()
        st.session_state.up_key += 1
        st.rerun()
    if st.session_state.index_log:
        with st.expander("Last indexing result"):
            st.code(st.session_state.index_log, language=None)

    with st.expander("Saved notes"):
        try:
            st.text(backend.call_tool("read_notes"))
        except Exception:  # noqa: BLE001
            st.caption("(could not read notes)")

    if st.button("Clear chat", use_container_width=True):
        new_chat()
        st.rerun()

# --------------------------------------------------------------------- chat
st.title(config.ASSISTANT_NAME)
st.caption(f"Ask me about: {config.PURPOSE}.")

if not st.session_state.ui:
    st.write("Try one of these:")
    for i, q in enumerate(EXAMPLES):
        if st.button(q, key=f"ex_{i}"):
            st.session_state.pending = q
            st.rerun()

for m in st.session_state.ui:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            render_assistant(m, show_tools)
        else:
            st.markdown(m["content"])

prompt = st.chat_input("Ask a question...") or st.session_state.pop("pending", None)
if prompt:
    st.session_state.ui.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            spinner_text = "Thinking..." if thinking else "Working on it..."
            with st.spinner(spinner_text):
                answer, events, reasoning = backend.ask(
                    st.session_state.messages, prompt, thinking=thinking)
            reply = {
                "role": "assistant", "content": answer, "tools": events,
                # show the reasoning panel only when the user asked for thinking
                "reasoning": reasoning if thinking else "",
                # model reasoned although thinking was off
                "leaked": bool(reasoning) and not thinking,
            }
        except Exception as e:  # noqa: BLE001
            reply = {"role": "assistant", "content": f"Sorry, something went wrong: {e}",
                     "tools": [], "reasoning": "", "leaked": False}
        render_assistant(reply, show_tools)
    st.session_state.ui.append(reply)
