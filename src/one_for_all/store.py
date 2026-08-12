"""Local memory store — the durable record of one person's context.

Everything lives in a single SQLite file on the user's machine. Nothing syncs
anywhere. Deleting that file deletes the memory, completely.

Two design decisions worth knowing before reading the code:

*Memory scopes.* Memories are typed as episodic (something that happened),
semantic (something that is true), or procedural (how this person wants things
done). They are retrieved and aged differently, so the distinction is structural
rather than a label.

*Temporal validity.* Nothing is ever updated in place and nothing is silently
deleted. A memory that stops being true is superseded by a newer one, and the
old row survives with a pointer forward. "What did I believe last month, and
when did that change?" stays answerable, and a wrong correction is recoverable.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_dir

# Retrieved memories are only useful if the model can act on them, so they are
# capped: a recall that returns an essay defeats the point.
MAX_MEMORY_CHARS = 2000

SCOPES = ("episodic", "semantic", "procedural")

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY,
    scope         TEXT NOT NULL,        -- episodic | semantic | procedural
    text          TEXT NOT NULL,
    source        TEXT,                 -- which client/session wrote it
    created_at    REAL NOT NULL,
    superseded_by INTEGER,              -- id of the memory that replaced this
    superseded_at REAL,
    used_count    INTEGER NOT NULL DEFAULT 0,
    last_used     REAL,
    FOREIGN KEY (superseded_by) REFERENCES memories(id)
);

CREATE INDEX IF NOT EXISTS idx_memories_live
    ON memories(superseded_by) WHERE superseded_by IS NULL;

-- FTS5 gives us BM25-ranked keyword search for free. That is one of the two
-- legs of hybrid retrieval; vector similarity is the other and arrives later.
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    text, content='memories', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF text ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;

-- Every decision the daemon makes, and what happened next.
--
-- Nothing reads this yet. It exists from the first commit because the
-- self-improvement work later is only possible if the history is already there,
-- and instrumentation cannot be retrofitted onto months of past use.
CREATE TABLE IF NOT EXISTS decisions (
    id       INTEGER PRIMARY KEY,
    ts       REAL NOT NULL,
    kind     TEXT NOT NULL,             -- recall | critique | curate
    context  TEXT,                      -- what it saw
    decision TEXT,                      -- what it chose
    outcome  TEXT                       -- filled in later, once known
);
"""


@dataclass(frozen=True)
class Memory:
    id: int
    scope: str
    text: str
    source: str | None
    created_at: float

    def render(self) -> str:
        return f"[{self.id}] ({self.scope}) {self.text}"


def default_db_path() -> Path:
    return Path(user_data_dir("one-for-all", appauthor=False)) / "store.db"


class Store:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    # -- writing ----------------------------------------------------------

    def remember(
        self,
        text: str,
        scope: str = "semantic",
        source: str | None = None,
        supersedes: int | None = None,
    ) -> int:
        """Store a memory. If it corrects an existing one, pass its id as
        `supersedes` — the old memory is retired, not destroyed."""
        text = text.strip()
        if not text:
            raise ValueError("refusing to store an empty memory")
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
        if len(text) > MAX_MEMORY_CHARS:
            raise ValueError(
                f"memory is {len(text)} chars; keep them under {MAX_MEMORY_CHARS}. "
                "Store the durable fact, not the transcript."
            )

        cur = self.db.execute(
            "INSERT INTO memories (scope, text, source, created_at) VALUES (?, ?, ?, ?)",
            (scope, text, source, time.time()),
        )
        new_id = int(cur.lastrowid)

        if supersedes is not None:
            self.db.execute(
                "UPDATE memories SET superseded_by = ?, superseded_at = ? WHERE id = ?",
                (new_id, time.time(), supersedes),
            )
        self.db.commit()
        return new_id

    def forget(self, memory_id: int) -> bool:
        """Hard-delete. For corrections prefer `remember(..., supersedes=id)`,
        which keeps the history. This is for things that should never have been
        written at all."""
        cur = self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.db.commit()
        return cur.rowcount > 0

    # -- reading ----------------------------------------------------------

    def search(self, query: str, limit: int = 8) -> list[Memory]:
        """Retrieve live memories relevant to `query`, best match first.

        The single retrieval path in the system — every component recalls
        through here, so improving retrieval is a change to one function.
        Today: BM25 over FTS5. Next: fuse in vector similarity and rank by a
        combined score.
        """
        rows = self.db.execute(
            """
            SELECT m.* FROM memories_fts f
            JOIN memories m ON m.id = f.rowid
            WHERE memories_fts MATCH ? AND m.superseded_by IS NULL
            ORDER BY rank
            LIMIT ?
            """,
            (_fts_query(query), limit),
        ).fetchall()
        self._mark_used([r["id"] for r in rows])
        return [_to_memory(r) for r in rows]

    def recent(self, limit: int = 20, scope: str | None = None) -> list[Memory]:
        sql = "SELECT * FROM memories WHERE superseded_by IS NULL"
        params: list[object] = []
        if scope:
            sql += " AND scope = ?"
            params.append(scope)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [_to_memory(r) for r in self.db.execute(sql, params).fetchall()]

    def history(self, memory_id: int) -> list[Memory]:
        """Walk the supersession chain forward from `memory_id`: what this
        belief was, and everything it became."""
        chain: list[Memory] = []
        current: int | None = memory_id
        seen: set[int] = set()
        while current is not None and current not in seen:
            seen.add(current)
            row = self.db.execute(
                "SELECT * FROM memories WHERE id = ?", (current,)
            ).fetchone()
            if row is None:
                break
            chain.append(_to_memory(row))
            current = row["superseded_by"]
        return chain

    def _mark_used(self, ids: list[int]) -> None:
        """Usage is the decay signal — it is how a live memory is eventually
        told apart from landfill."""
        if not ids:
            return
        now = time.time()
        self.db.executemany(
            "UPDATE memories SET used_count = used_count + 1, last_used = ? WHERE id = ?",
            [(now, i) for i in ids],
        )
        self.db.commit()

    # -- decision log -----------------------------------------------------

    def log_decision(self, kind: str, context: str, decision: str) -> int:
        cur = self.db.execute(
            "INSERT INTO decisions (ts, kind, context, decision) VALUES (?, ?, ?, ?)",
            (time.time(), kind, context, decision),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def record_outcome(self, decision_id: int, outcome: str) -> None:
        self.db.execute(
            "UPDATE decisions SET outcome = ? WHERE id = ?", (outcome, decision_id)
        )
        self.db.commit()

    def stats(self) -> dict[str, int]:
        live = "SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL"
        return {
            "live_memories": self.db.execute(live).fetchone()[0],
            "superseded": self.db.execute(
                "SELECT COUNT(*) FROM memories WHERE superseded_by IS NOT NULL"
            ).fetchone()[0],
            "decisions_logged": self.db.execute(
                "SELECT COUNT(*) FROM decisions"
            ).fetchone()[0],
        }


def _fts_query(query: str) -> str:
    """FTS5 reads bare punctuation as query operators, so an apostrophe or a
    hyphen in ordinary user text raises a syntax error. Quote each term."""
    terms = [t for t in query.replace('"', " ").split() if t.strip("-+*():")]
    return " OR ".join(f'"{t}"' for t in terms) or '""'


def _to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id=row["id"],
        scope=row["scope"],
        text=row["text"],
        source=row["source"],
        created_at=row["created_at"],
    )
