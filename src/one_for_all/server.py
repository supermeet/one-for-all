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
from .critic import Critic
from .curator import Curator
from .router import Router
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

# The model layer is built lazily and never at import time. The client spawns
# this process on startup and kills it on exit, so anything slow or fallible
# here shows up as "the MCP server does not work" — and the memory tools must
# keep working on a laptop with no key, no GPU and no network (invariant 6).
_model_layer: tuple[Router, Curator, Critic] | None = None

# The id of the last curation, so `recall_was_incomplete` can label it without
# the model having to track one. This is the automatic training signal from
# MODEL.md §1.1: the label falls out of ordinary use.
_last_curation: int | None = None


def _layer() -> tuple[Router, Curator, Critic]:
    global _model_layer
    if _model_layer is None:
        router = Router(store=store)
        _model_layer = (router, Curator(router, store), Critic(router, store))
    return _model_layer


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

    If what you needed is missing from the result, call recall_was_incomplete()
    — that is how retrieval gets better.
    """
    global _last_curation

    # Fetch wider than asked, then let the curator narrow it. Retrieval ranks
    # by keyword overlap; the curator ranks by relevance to the actual task,
    # and those disagree often enough to be worth a small model.
    candidates = store.search(query, limit=limit * 3, mark_used=False)

    # Logged whether or not anything was found: an empty recall is itself a
    # signal about retrieval quality, and it is the case we most want to fix.
    store.log_decision(
        kind="recall",
        context=query,
        decision=",".join(str(m.id) for m in candidates) or "none",
    )
    if not candidates:
        return "No relevant memories. Proceed, and consider remembering what you learn."

    try:
        curation = _layer()[1].curate(query, candidates)
        hits, _last_curation = curation.selected[:limit], curation.decision_id
    except Exception:
        # Curation is an optimisation over a working recall. If the model layer
        # is misconfigured or unreachable, returning the raw hits is the whole
        # of the damage — memory must not need a model to work.
        hits, _last_curation = candidates[:limit], None

    store.mark_used([m.id for m in hits])
    return "\n".join(m.render() for m in hits)


@mcp.tool()
def recall_was_incomplete(looking_for: str = "") -> str:
    """Report that the last recall() left out something you needed.

    Call this whenever you called recall() and then had to ask the user, guess,
    or search the codebase for context that should have been remembered.

    This is not an apology and it costs nothing — it is the single most valuable
    signal the system collects. Each report labels a real retrieval failure, and
    those labels are what the curator is later trained and measured on.
    """
    if _last_curation is None:
        return "Noted, but there was no curated recall to label."
    _layer()[1].record_miss(_last_curation, looking_for)
    return "Recorded as a miss. That case will be used to improve retrieval."


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


@mcp.tool()
def critique(code: str, context: str = "") -> str:
    """Get a second opinion on code before presenting it as finished.

    Call this after writing or substantially changing code, especially where a
    bug would be expensive or where you are unsure a library already does this.
    It reads the code cold, with this user's stated preferences in hand.

    It is quiet by design: an empty response means nothing cleared the
    confidence bar, which is the common and correct case. Do not re-ask, and do
    not treat silence as approval to skip your own review.
    """
    router, _, critic = _layer()
    # The critic is one of the few components that gets to see memories it did
    # not ask for. "Does this match how the user writes" is unanswerable
    # without them, and it is the question a generic linter cannot ask.
    prefs = store.search(context or code[:200], limit=5)
    result = critic.review(code, context=context, memories=prefs)

    if result.error:
        # A critic that cannot reach a model has no opinion. That is not an
        # error the user needs to hear about (MODEL.md §9).
        return ""
    if not result.spoke:
        return ""
    header = f"Second opinion ({result.model}), {len(result.findings)} finding(s):\n"
    return header + result.render()


@mcp.tool()
def model_status() -> str:
    """Show which backends and models are reachable, and what each role would
    pick right now.

    Use when recall or critique behaves unexpectedly, or when the user asks
    what this is running on. Model ids are fetched live, so this reflects
    reality rather than configuration.
    """
    try:
        router, _, _ = _layer()
        return f"{router.status()}\n\nstore: {store.stats()}"
    except Exception as exc:
        return f"Model layer unavailable: {exc}\n\nstore: {store.stats()}"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
