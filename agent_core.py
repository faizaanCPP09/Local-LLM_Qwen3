"""
Backend for the Streamlit UI.

Starts the MCP servers once and keeps them alive in a background thread that has
its own asyncio event loop. The UI (which re-runs on every click) just calls
ask() / call_tool() and never has to deal with asyncio itself.

It reuses SERVERS and MAX_TOOL_ROUNDS from agent.py, so adding an MCP server
there makes it show up in the UI too.
"""
import asyncio
import re
import sys
import threading
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient

import config
from agent import MAX_TOOL_ROUNDS, SERVERS

# Context window. Bigger = remembers more of the conversation, but uses more RAM.
NUM_CTX = getattr(config, "NUM_CTX", 8192)


def split_reasoning(text: str | None):
    """Split model output into (reasoning, answer).

    Handles <think>...</think> pairs and also output that only contains a closing
    </think> (some models add the opening tag in their template).
    """
    text = text or ""
    reasoning = [p.strip() for p in re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        before, _, text = text.rpartition("</think>")
        reasoning.append(before.replace("<think>", "").strip())
    return "\n\n".join(r for r in reasoning if r), text.strip()


def clean(text: str | None) -> str:
    """The answer only, with any reasoning removed."""
    return split_reasoning(text)[1]


class Backend:
    def __init__(self) -> None:
        self.loop = None
        self.thread = None
        self.tool_map = {}      # tool name -> MCP session
        self.tools = []         # tools in Ollama's format
        self.servers = {}       # server name -> [tool names]
        self.llm = None
        self.thinking_capable = False   # does the chat model have a thinking mode?
        self._life = None
        self._stop_event = None
        self._ready = threading.Event()
        self._error = None

    # ------------------------------------------------------------ lifecycle
    def start(self, timeout: float = 120) -> None:
        """Start the event-loop thread and connect to every MCP server."""
        self._ready.clear()
        self._error = None
        # Windows needs the Proactor loop to launch subprocesses (the MCP servers),
        # and Streamlit/Tornado can switch the default to the Selector loop.
        self.loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self._life = asyncio.run_coroutine_threadsafe(self._lifecycle(), self.loop)

        if not self._ready.wait(timeout):
            self._shutdown_loop()
            raise TimeoutError("Timed out while starting the MCP servers.")
        if self._error is not None:
            err = self._error
            self._shutdown_loop()
            raise RuntimeError(
                f"{err}. Run 'python rag_mcp_server.py' in a terminal to see the real error."
            ) from err

    def stop(self) -> None:
        """Close the MCP servers and stop the background thread."""
        if self.loop is None:
            return
        if self._stop_event is not None:
            self.loop.call_soon_threadsafe(self._stop_event.set)
        try:
            self._life.result(timeout=30)
        except Exception:
            pass
        self._shutdown_loop()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _shutdown_loop(self) -> None:
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)
            if self.thread is not None:
                self.thread.join(timeout=10)
            try:
                self.loop.close()
            except Exception:
                pass
        self.loop = self.thread = self._life = self._stop_event = None
        self.tool_map, self.tools, self.servers = {}, [], {}

    async def _lifecycle(self) -> None:
        # One long-lived task enters AND leaves the MCP connections, because the
        # underlying anyio task groups must be closed by the task that opened them.
        self._stop_event = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                await self._connect(stack)
                self._ready.set()
                await self._stop_event.wait()
        except Exception as e:  # noqa: BLE001
            if not self._ready.is_set():
                self._error = e
                self._ready.set()

    async def _connect(self, stack: AsyncExitStack) -> None:
        tool_map, tools, servers = {}, [], {}
        for name, script in SERVERS.items():
            params = StdioServerParameters(command=sys.executable,
                                           args=[str(config.BASE_DIR / script)])
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            servers[name] = []
            for t in (await session.list_tools()).tools:
                tool_map[t.name] = session
                servers[name].append(t.name)
                tools.append({"type": "function", "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                }})
        self.tool_map, self.tools, self.servers = tool_map, tools, servers
        self.llm = AsyncClient()
        # Ask Ollama whether this model has a thinking mode. Only models that do
        # accept the `think` option; others would return an error if we sent it.
        try:
            info = await self.llm.show(config.CHAT_MODEL)
            self.thinking_capable = "thinking" in (getattr(info, "capabilities", None) or [])
        except Exception:  # noqa: BLE001
            self.thinking_capable = config.CHAT_MODEL.startswith("qwen3")

    # ----------------------------------------------------------- public API
    def call_tool(self, name: str, args: dict | None = None, timeout: float = 60) -> str:
        """Call one MCP tool directly (used by the sidebar, not by the model)."""
        fut = asyncio.run_coroutine_threadsafe(self._call_tool(name, args or {}), self.loop)
        return fut.result(timeout)

    def ask(self, messages: list, user_text: str, thinking: bool = False, timeout: float = 900):
        """Run one user turn. Returns (answer, tool_events, reasoning).

        `messages` is the running conversation (system prompt first). After the turn
        it holds only the user message and the final answer, not the bulky tool
        output or reasoning, so the context stays small on a CPU-only laptop.
        `reasoning` is the model's thinking text ("" if there was none).
        """
        fut = asyncio.run_coroutine_threadsafe(self._ask(messages, user_text, thinking), self.loop)
        return fut.result(timeout)

    # ------------------------------------------------------------- internals
    async def _call_tool(self, name: str, args: dict) -> str:
        session = self.tool_map.get(name)
        if session is None:
            return f"Unknown tool: {name}"
        result = await session.call_tool(name, args or {})
        return "\n".join(c.text for c in result.content if hasattr(c, "text"))

    async def _ask(self, messages: list, user_text: str, thinking: bool):
        think_kw = {"think": bool(thinking)} if self.thinking_capable else {}
        start = len(messages)
        events, reasoning = [], []
        answer = "(stopped after too many tool calls)"
        try:
            messages.append({"role": "user", "content": user_text})
            for _ in range(MAX_TOOL_ROUNDS):
                resp = await self.llm.chat(
                    model=config.CHAT_MODEL, messages=messages, tools=self.tools,
                    options={"temperature": 0.2, "num_ctx": NUM_CTX}, **think_kw)
                msg = resp.message
                messages.append(msg)
                inline_reasoning, content = split_reasoning(msg.content)
                reasoning += [getattr(msg, "thinking", None) or "", inline_reasoning]
                if not msg.tool_calls:
                    answer = content or "(no answer)"
                    break
                for call in msg.tool_calls:
                    name = call.function.name
                    args = dict(call.function.arguments or {})
                    output = await self._call_tool(name, args)
                    events.append({"tool": name, "args": args, "output": output})
                    messages.append({"role": "tool", "content": output, "tool_name": name})
        except BaseException:
            del messages[start:]     # leave the conversation as it was
            raise
        del messages[start:]
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": answer})
        return answer, events, "\n\n".join(r.strip() for r in reasoning if r and r.strip())
