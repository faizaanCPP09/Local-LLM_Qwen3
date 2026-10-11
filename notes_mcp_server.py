"""
MCP server #2: a simple action tool (save/read notes in a local file).
It shows the agent using more than one MCP server. Copy this pattern to add
your own tools later (SQL queries, ticket lookup, SharePoint, APIs...).
"""
from datetime import datetime

from mcp.server.fastmcp import FastMCP

import config

mcp = FastMCP("notes")


@mcp.tool()
def save_note(text: str) -> str:
    """Save a note to the local notes file. Only use when the user asks to save or remember something."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(config.NOTES_FILE, "a", encoding="utf-8") as f:
        f.write(f"- [{stamp}] {text}\n")
    return "Note saved."


@mcp.tool()
def read_notes() -> str:
    """Read all saved notes."""
    if not config.NOTES_FILE.exists():
        return "No notes yet."
    return config.NOTES_FILE.read_text(encoding="utf-8")


if __name__ == "__main__":
    mcp.run()
