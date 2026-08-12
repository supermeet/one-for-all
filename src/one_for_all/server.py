"""MCP server — the surface every client talks to.

Claude Code, Cursor, VS Code/Copilot and Gemini CLI all speak MCP, so one daemon
serves all of them and we ship no UI of our own.

The tool docstrings below are not documentation. They are the only thing the
model reads when deciding whether to call a tool, so they state *when* to call,
not just what the tool does. Vague descriptions get ignored; prescriptive
trigger conditions get used. Treat edits here as behaviour changes.
"""

from __future__ import annotations

from mcp.server import MCPServer

from . import __version__
from .store import Store

# MCP SDK 2.0 renamed FastMCP -> MCPServer. Same decorator/run surface.
mcp = MCPServer(
    "one-for-all",
    version=__version__,
    instructions=(
        "Persistent memory for this user, stored locally on their machine. "
        "Call recall() before assuming how they work or what they have already "
        "decided. Call remember() when you learn something that will still be "
        "true tomorrow."
    ),
)
store = Store()


@mcp.tool()
def remember(text: str, scope: str = "semantic", supersedes: int | None = None) -> str:
    """Save something about this user that will still matter tomorrow.

    Call this when the user states a preference, corrects you, makes a decision
    worth honouring later, or explains a constraint of their project. Prefer one
    specific sentence over a paragraph.

    Do NOT call this for anything the code or git history already records, or
    for details that only matter inside the current task.

    scope:
      semantic   - something that is true ("the API keys live in .env.local")
      procedural - how they want things done ("prefers stdlib over new deps")
      episodic   - something that happened ("tried Postgres in June, too heavy")

    supersedes: when this corrects an earlier memory, pass that memory's id.
    The old one is retired rather than deleted, so the change stays auditable.
    """
    try:
        memory_id = store.remember(text, scope=scope, supersedes=supersedes)
    except ValueError as exc:
        return f"Not stored: {exc}"
    note = f", replacing #{supersedes}" if supersedes else ""
    return f"Remembered #{memory_id} ({scope}){note}."


@mcp.tool()
def recall(query: str, limit: int = 8) -> str:
    """Retrieve what is already known about this user, relevant to `query`.

    Call this at the START of a task, before assuming how they work, what they
    have already decided, or how this project is set up. Also call it when you
    are about to suggest an approach — they may have rejected it before.

    Returns memories one per line as: [id] (scope) text
    """
    hits = store.search(query, limit=limit)
    # Logged whether or not anything was found: an empty recall is itself a
    # signal about retrieval quality, and it is the case we most want to fix.
    store.log_decision(
        kind="recall",
        context=query,
        decision=",".join(str(m.id) for m in hits) or "none",
    )
    if not hits:
        return "No relevant memories. Proceed, and consider remembering what you learn."
    return "\n".join(m.render() for m in hits)


@mcp.tool()
def list_recent(limit: int = 20, scope: str | None = None) -> str:
    """List the most recently stored memories, newest first.

    Use when the user asks what you know about them, or to audit for stale or
    wrong entries. Optionally filter by scope.
    """
    items = store.recent(limit=limit, scope=scope)
    if not items:
        return "Nothing stored yet."
    return "\n".join(m.render() for m in items)


@mcp.tool()
def memory_history(memory_id: int) -> str:
    """Show how a belief changed: the given memory and everything that
    superseded it, oldest first.

    Use when a memory looks wrong or contradictory and you need to see what it
    used to say.
    """
    chain = store.history(memory_id)
    if not chain:
        return f"No memory #{memory_id}."
    return "\n".join(
        f"{'→ ' if i else ''}{m.render()}" for i, m in enumerate(chain)
    )


@mcp.tool()
def forget(memory_id: int) -> str:
    """Permanently delete a memory.

    Use only for things that should never have been stored — noise, secrets,
    a mistaken write. To CORRECT something that was true and no longer is, call
    remember(..., supersedes=id) instead so the history survives.
    """
    return "Deleted." if store.forget(memory_id) else f"No memory #{memory_id}."


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
