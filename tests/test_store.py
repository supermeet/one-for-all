import pytest

from one_for_all.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.db")


def test_remember_and_search(store):
    store.remember("prefers the standard library over new dependencies", scope="procedural")
    store.remember("the deploy script lives in ops/deploy.sh", scope="semantic")

    hits = store.search("dependencies")
    assert any("standard library" in m.text for m in hits)


def test_search_survives_punctuation(store):
    """FTS5 reads bare punctuation as operators — this raised before quoting."""
    store.remember("don't use tabs")
    assert store.search("don't -- tabs?") is not None


def test_search_excludes_superseded(store):
    old = store.remember("uses Postgres")
    store.remember("uses SQLite", supersedes=old)

    texts = [m.text for m in store.search("Postgres SQLite")]
    assert "uses SQLite" in texts
    assert "uses Postgres" not in texts


def test_history_walks_the_chain(store):
    first = store.remember("uses Postgres")
    second = store.remember("uses SQLite", supersedes=first)
    store.remember("uses DuckDB", supersedes=second)

    chain = [m.text for m in store.history(first)]
    assert chain == ["uses Postgres", "uses SQLite", "uses DuckDB"]


def test_rejects_bad_input(store):
    with pytest.raises(ValueError):
        store.remember("   ")
    with pytest.raises(ValueError):
        store.remember("x", scope="nonsense")
    with pytest.raises(ValueError):
        store.remember("x" * 3000)


def test_usage_is_tracked(store):
    mid = store.remember("prefers tabs")
    store.search("tabs")
    row = store.db.execute(
        "SELECT used_count FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    assert row["used_count"] == 1


def test_forget(store):
    mid = store.remember("wrong fact")
    assert store.forget(mid) is True
    assert store.forget(mid) is False
    assert store.search("wrong fact") == []


def test_decision_log(store):
    did = store.log_decision("recall", "python style", "1,2")
    store.record_outcome(did, "used")
    row = store.db.execute(
        "SELECT outcome FROM decisions WHERE id = ?", (did,)
    ).fetchone()
    assert row["outcome"] == "used"


def test_stats(store):
    old = store.remember("a")
    store.remember("b", supersedes=old)
    store.log_decision("recall", "q", "1")

    assert store.stats() == {
        "live_memories": 1,
        "superseded": 1,
        "decisions_logged": 1,
    }
