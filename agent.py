"""
The local assistant: Ollama model + MCP client.
- Starts every MCP server listed in SERVERS
- Gives their tools to the local model
- Runs the tool calls the model asks for and feeds results back
Run:  python agent.py
"""
import asyncio
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient

import config

# Add more MCP servers here: "name": "script.py"
SERVERS = {
    "knowledge": "rag_mcp_server.py",
    "notes": "notes_mcp_server.py",
}
MAX_TOOL_ROUNDS = 5


async def connect_servers(stack: AsyncExitStack):
    tool_map, ollama_tools = {}, []
    for name, script in SERVERS.items():
        params = StdioServerParameters(command=sys.executable,
                                       args=[str(config.BASE_DIR / script)])
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        for t in (await session.list_tools()).tools:
            tool_map[t.name] = session
            ollama_tools.append({"type": "function", "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            }})
        print(f"[MCP] connected '{name}': {[t for t, s in tool_map.items() if s is session]}")
    return tool_map, ollama_tools


async def call_tool(tool_map, name, args) -> str:
    session = tool_map.get(name)
    if session is None:
        return f"Unknown tool: {name}"
    result = await session.call_tool(name, args or {})
    return "\n".join(c.text for c in result.content if hasattr(c, "text"))


async def main():
    llm = AsyncClient()
    extra = {"think": False} if config.CHAT_MODEL.startswith("qwen3") else {}

    async with AsyncExitStack() as stack:
        tool_map, ollama_tools = await connect_servers(stack)
        messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
        print(f"\n{config.ASSISTANT_NAME} ready (model: {config.CHAT_MODEL}, fully local).")
        print("Type 'exit' to quit, '/reset' to clear the conversation.\n")

        while True:
            user = (await asyncio.to_thread(input, "You: ")).strip()
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                break
            if user == "/reset":
                messages = messages[:1]
                print("(conversation cleared)\n")
                continue

            messages.append({"role": "user", "content": user})
            for _ in range(MAX_TOOL_ROUNDS):
                resp = await llm.chat(model=config.CHAT_MODEL, messages=messages,
                                      tools=ollama_tools, options={"temperature": 0.2}, **extra)
                msg = resp.message
                messages.append(msg)
                if not msg.tool_calls:
                    print(f"\nAssistant: {msg.content}\n")
                    break
                for call in msg.tool_calls:
                    name, args = call.function.name, call.function.arguments
                    print(f"  [tool] {name}({args})")
                    output = await call_tool(tool_map, name, args)
                    messages.append({"role": "tool", "content": output, "tool_name": name})
            else:
                print("\nAssistant: (stopped after too many tool calls)\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
