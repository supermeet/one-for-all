"""Tests for the model layer.

Every test here runs offline against fake backends. That is a requirement, not
a convenience: a suite that needs a free-tier key is a suite that fails when the
free tier is rate-limited, and the routing and parsing logic is exactly the part
that must be verifiable without a network.
"""

from __future__ import annotations

import json

import pytest

from one_for_all import dataset, evals
from one_for_all.backends import BackendUnavailable, ModelInfo, RateLimited, parse_params_b
from one_for_all.config import BackendConfig, load as load_config
from one_for_all.critic import _parse_findings
from one_for_all.curator import Curator, parse_selection
from one_for_all.router import NoBackendAvailable, PrivacyRefusal, Router
from one_for_all.store import Memory, Store


class FakeBackend:
    """Duck-types Backend. Records what it was asked, so a test can assert on
    *which* model the router reached for, not merely that it succeeded."""

    def __init__(
        self, name, models, local=False, free_only=False, reply="", fail=None,
        non_text=(),
    ):
        self.cfg = BackendConfig(
            name=name,
            base_url="http://fake",
            api_key=None,
            local=local,
            free_only=free_only,
            enabled=True,
        )
        self.name = name
        self.local = local
        self._models = models
        self.reply = reply
        self.fail = fail
        self.non_text = set(non_text)
        self.calls: list[str] = []

    def catalog(self, refresh=False):
        if self.fail == "unavailable":
            raise BackendUnavailable(f"{self.name}: down")
        return [
            ModelInfo(
                id=mid,
                context_length=ctx,
                params_b=parse_params_b(mid),
                free=free,
                backend=self.name,
                text_only=mid not in self.non_text,
            )
            for mid, ctx, free in self._models
        ]

    def chat(self, messages, model, **kwargs):
        self.calls.append(model)
        if self.fail == "429":
            raise RateLimited(f"{self.name}: 429")
        return self.reply


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.db")


@pytest.fixture
def config():
    return load_config()


def mem(mid, text, scope="semantic"):
    return Memory(id=mid, scope=scope, text=text, source=None, created_at=0.0)


# -- model id parsing -----------------------------------------------------

@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("gemma3:4b", 4.0),
        ("qwen3-8b-instruct", 8.0),
        # The version number must not be mistaken for a parameter count.
        ("llama-3.3-8b-instruct:free", 8.0),
        ("phi-4-mini-3.8b", 3.8),
        ("gpt-4o-mini", None),
        ("some-model", None),
    ],
)
def test_parse_params(model_id, expected):
    assert parse_params_b(model_id) == expected


# -- routing --------------------------------------------------------------

def test_curate_prefers_the_smallest_model(config, store):
    backend = FakeBackend(
        "remote", [("big-70b", 8000, True), ("small-4b", 8000, True)], free_only=True
    )
    router = Router(config=config, store=store, backends=[backend])
    assert router.candidates("curate")[0][1].id == "small-4b"


def test_critique_prefers_the_most_capable(config, store):
    backend = FakeBackend(
        "remote", [("tiny-1b", 8000, True), ("big-70b", 8000, True)], free_only=True
    )
    router = Router(config=config, store=store, backends=[backend])
    assert router.candidates("critique")[0][1].id == "big-70b"


def test_curate_excludes_models_over_the_size_cap(config, store):
    backend = FakeBackend("remote", [("huge-70b", 8000, True)], free_only=True)
    router = Router(config=config, store=store, backends=[backend])
    assert router.candidates("curate") == []
    # The same model is fine for the critic, which has no cap.
    assert router.candidates("critique")


def test_free_only_backend_hides_paid_models(config, store):
    backend = FakeBackend(
        "remote", [("paid-4b", 8000, False), ("free-4b", 8000, True)], free_only=True
    )
    router = Router(config=config, store=store, backends=[backend])
    ids = [m.id for _, m in router.candidates("curate")]
    assert ids == ["free-4b"]


def test_unavailable_backend_is_skipped_not_fatal(config, store):
    """Ollama being absent is the normal case on the dev laptop (MODEL.md §5)."""
    dead = FakeBackend("ollama", [], local=True, fail="unavailable")
    alive = FakeBackend("remote", [("small-4b", 8000, True)], free_only=True)
    router = Router(config=config, store=store, backends=[dead, alive])
    assert [m.id for _, m in router.candidates("curate")] == ["small-4b"]


def test_non_text_models_are_never_candidates(config, store):
    """Regression. On the first live run against OpenRouter, the `generate`
    role picked a music model: it advertised a 1M context, which won a ranking
    that sorts by context length. A provider's model list is not a list of chat
    models."""
    backend = FakeBackend(
        "remote",
        [("music-model", 1_000_000, True), ("text-model-8b", 128_000, True)],
        free_only=True,
        non_text={"music-model"},
    )
    router = Router(config=config, store=store, backends=[backend])
    for role in ("curate", "critique", "generate"):
        assert [m.id for _, m in router.candidates(role)] == ["text-model-8b"], role


def test_text_only_detection_from_a_provider_payload():
    from one_for_all.backends import _is_text_only

    assert _is_text_only({"output_modalities": ["text"]})
    assert not _is_text_only({"output_modalities": ["text", "audio"]})
    assert not _is_text_only({"output_modalities": ["image", "text"]})
    # Multimodal *input* is harmless; only what comes back matters.
    assert _is_text_only({"input_modalities": ["text", "image"], "output_modalities": ["text"]})
    # Ollama and LM Studio report no architecture at all.
    assert _is_text_only({})


def test_local_is_preferred_for_curation(config, store):
    local = FakeBackend("ollama", [("gemma3:4b", 8000, True)], local=True)
    remote = FakeBackend("remote", [("gemma-3-4b", 8000, True)], free_only=True)
    router = Router(config=config, store=store, backends=[remote, local])
    assert router.candidates("curate")[0][0].name == "ollama"


# -- the privacy gate -----------------------------------------------------

def test_sensitive_content_is_detected(config, store):
    router = Router(config=config, store=store, backends=[])
    assert router.is_sensitive("the API_KEY is sk-abc123")
    assert router.is_sensitive("put it in .env")
    assert not router.is_sensitive("refactor the upload handler")


def test_sensitive_content_never_reaches_a_remote_backend(config, store):
    remote = FakeBackend("remote", [("small-4b", 8000, True)], free_only=True, reply="1")
    router = Router(config=config, store=store, backends=[remote])

    with pytest.raises(PrivacyRefusal):
        router.complete("curate", [{"role": "user", "content": "the password is hunter2"}])
    # Invariant 4: refusal happens before the request is formed, not after.
    assert remote.calls == []


def test_sensitive_content_routes_to_local(config, store):
    local = FakeBackend("ollama", [("gemma3:4b", 8000, True)], local=True, reply="1")
    remote = FakeBackend("remote", [("small-4b", 8000, True)], free_only=True, reply="9")
    router = Router(config=config, store=store, backends=[remote, local])

    result = router.complete("curate", [{"role": "user", "content": "my password is x"}])
    assert result.backend == "ollama"
    assert remote.calls == []


def test_rate_limit_spills_to_the_next_candidate(config, store):
    """429 is a normal condition on a free tier, not a failure."""
    limited = FakeBackend("remote", [("small-4b", 8000, True)], free_only=True, fail="429")
    local = FakeBackend("ollama", [("gemma3:4b", 8000, True)], local=True, reply="ok")
    router = Router(config=config, store=store, backends=[limited, local])

    result = router.complete("curate", [{"role": "user", "content": "hello"}])
    assert result.backend == "ollama"


def test_no_backend_raises_rather_than_guessing(config, store):
    router = Router(config=config, store=store, backends=[])
    with pytest.raises(NoBackendAvailable):
        router.complete("critique", [{"role": "user", "content": "hi"}])


def test_routing_is_logged(config, store):
    backend = FakeBackend("remote", [("small-4b", 8000, True)], free_only=True, reply="1")
    Router(config=config, store=store, backends=[backend]).complete(
        "curate", [{"role": "user", "content": "hello"}]
    )
    row = store.db.execute("SELECT * FROM decisions WHERE kind='route'").fetchone()
    assert "small-4b" in row["decision"]


# -- curator parsing ------------------------------------------------------

CANDIDATES = [mem(3, "alpha"), mem(7, "beta"), mem(12, "gamma")]


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("3,7", [3, 7]),
        ("3, 7, 12", [3, 7, 12]),
        ("ids: 3 and 12", [3, 12]),
        ("ALL", [3, 7, 12]),
        ("NONE", []),
        # Hallucinated ids are dropped, the good picks survive.
        ("3, 99", [3]),
        # Duplicates collapse rather than double-count.
        ("3,3,7", [3, 7]),
    ],
)
def test_parse_selection(reply, expected):
    picked = parse_selection(reply, CANDIDATES)
    assert [m.id for m in picked] == expected


def test_broken_format_is_distinguishable_from_an_empty_selection():
    """'nothing is relevant' and 'the model did not answer' are different
    events, and only the second is a format-compliance failure."""
    assert parse_selection("NONE", CANDIDATES) == []
    assert parse_selection("I'm not sure what you want", CANDIDATES) is None
    assert parse_selection("", CANDIDATES) is None


# -- curator behaviour ----------------------------------------------------

def _curator(store, config, reply, n_models=1):
    backend = FakeBackend("remote", [("small-4b", 8000, True)], free_only=True, reply=reply)
    router = Router(config=config, store=store, backends=[backend])
    return Curator(router, store, config), backend


def test_curator_skips_the_model_when_there_is_little_to_select_from(store, config):
    """Calling a model to filter six items costs more than it saves."""
    curator, backend = _curator(store, config, "3")
    result = curator.curate("some task", CANDIDATES)
    assert len(result.selected) == 3
    assert result.fallback_reason == "below min_candidates"
    assert backend.calls == []


def test_curator_narrows_when_there_is_enough_to_narrow(store, config):
    many = [mem(i, f"memory number {i}") for i in range(1, 13)]
    curator, backend = _curator(store, config, "1,2,3")
    result = curator.curate("a real task worth curating", many)
    assert [m.id for m in result.selected] == [1, 2, 3]
    assert backend.calls == ["small-4b"]
    assert result.compression_ratio > 1.0


def test_curator_fails_open_on_unparseable_output(store, config):
    """Recall over precision: dropping something needed is far worse than
    keeping something extra, so every failure path keeps everything."""
    many = [mem(i, f"memory number {i}") for i in range(1, 13)]
    curator, _ = _curator(store, config, "I don't understand the question")
    result = curator.curate("a real task worth curating", many)
    assert len(result.selected) == len(many)
    assert result.fallback_reason == "unparseable selection"


def test_curator_fails_open_when_no_backend_exists(store, config):
    many = [mem(i, f"memory number {i}") for i in range(1, 13)]
    curator = Curator(Router(config=config, store=store, backends=[]), store, config)
    result = curator.curate("a real task worth curating", many)
    assert len(result.selected) == len(many)
    assert "no backend" in result.fallback_reason


def test_curation_is_logged_replayably(store, config):
    """The log row must survive the memories it references being deleted, or it
    is not instrumentation."""
    many = [mem(i, f"memory number {i}") for i in range(1, 13)]
    curator, _ = _curator(store, config, "1,2")
    result = curator.curate("a real task worth curating", many)

    row = store.db.execute(
        "SELECT * FROM decisions WHERE id = ?", (result.decision_id,)
    ).fetchone()
    context = json.loads(row["context"])
    assert context["task"] == "a real task worth curating"
    assert len(context["candidates"]) == 12
    assert context["candidates"][0]["text"] == "memory number 1"


def test_recorded_miss_becomes_an_outcome(store, config):
    many = [mem(i, f"memory number {i}") for i in range(1, 13)]
    curator, _ = _curator(store, config, "1,2")
    result = curator.curate("a real task worth curating", many)
    curator.record_miss(result.decision_id, "needed the deploy path")

    row = store.db.execute(
        "SELECT outcome FROM decisions WHERE id = ?", (result.decision_id,)
    ).fetchone()
    assert row["outcome"].startswith("miss")


# -- critic ---------------------------------------------------------------

def test_critic_parses_fenced_json():
    reply = """Here's what I found:
```json
[{"kind": "bug", "confidence": 0.9, "summary": "off by one", "detail": "use <="}]
```"""
    findings = _parse_findings(reply)
    assert len(findings) == 1
    assert findings[0].kind == "bug"


def test_critic_silence_is_a_clean_result_not_a_failure():
    assert _parse_findings("[]") == []
    assert _parse_findings("No issues found.") == []
    # Genuinely broken output is still distinguishable from silence.
    assert _parse_findings("uhh") is None


def test_unparseable_confidence_lands_below_any_bar():
    """The failure direction is silence, never a false positive."""
    findings = _parse_findings('[{"summary": "x", "confidence": "high"}]')
    assert findings[0].confidence == 0.0


def test_critic_suppresses_below_the_confidence_bar(store, config):
    from one_for_all.critic import Critic

    reply = json.dumps(
        [
            {"kind": "bug", "confidence": 0.95, "summary": "real bug", "detail": "d"},
            {"kind": "style", "confidence": 0.2, "summary": "vague hunch", "detail": "d"},
        ]
    )
    backend = FakeBackend("remote", [("big-70b", 8000, True)], free_only=True, reply=reply)
    critic = Critic(Router(config=config, store=store, backends=[backend]), store, config)

    result = critic.review("def f(): pass")
    assert [f.summary for f in result.findings] == ["real bug"]
    # Suppressed findings are still logged — whether the bar is right is only
    # answerable if what it filtered was written down.
    assert [f.summary for f in result.suppressed] == ["vague hunch"]


def test_critic_stays_silent_when_no_backend_is_reachable(store, config):
    from one_for_all.critic import Critic

    critic = Critic(Router(config=config, store=store, backends=[]), store, config)
    result = critic.review("def f(): pass")
    assert result.findings == []
    assert result.render() == ""  # no opinion is not an error message


# -- eval harness ---------------------------------------------------------

def _case(cid, expected, n=4):
    return evals.EvalCase(
        id=cid,
        task="task",
        candidates=[mem(i, f"memory {i}") for i in range(1, n + 1)],
        expected=set(expected),
    )


def test_perfect_run_scores_clean():
    cases = [_case("a", [1, 2])]
    report = evals.run(cases, lambda case: "1,2", "perfect")
    assert report.miss_rate == 0.0
    assert report.noise_rate == 0.0
    assert report.format_compliance == 1.0


def test_miss_and_noise_are_measured_separately():
    cases = [_case("a", [1, 2])]
    report = evals.run(cases, lambda case: "1,3", "mixed")
    assert report.miss_rate == 0.5  # needed {1,2}, missed 2 of the 2 wanted
    assert report.noise_rate == 0.5  # picked {1,3}, and 3 was not wanted


def test_format_failure_is_counted():
    report = evals.run([_case("a", [1])], lambda case: "no idea", "broken")
    assert report.format_compliance == 0.0


def test_keep_all_baseline_never_misses():
    cases = [_case("a", [1, 2]), _case("b", [3])]
    report = evals.run(cases, evals.keep_all_runner(), "keep-all")
    assert report.miss_rate == 0.0
    assert report.compression_ratio == pytest.approx(1.0)


def test_miss_is_penalised_harder_than_noise():
    """MODEL.md §8: the miss is the failure that matters, and it is asymmetric."""
    cases = [_case("a", [1, 2])]
    misses = evals.run(cases, lambda case: "1", "misses")
    noisy = evals.run(cases, lambda case: "1,2,3", "noisy")
    assert misses.penalty > noisy.penalty


def test_backend_errors_are_not_scored_as_model_failures():
    """Regression. The first live run had no API key; every call 401'd, and the
    harness reported '100% miss rate, 0% format compliance' — a setup problem
    wearing a model verdict's clothes."""
    from one_for_all.backends import BackendError

    def failing(case):
        raise BackendError("openrouter: HTTP 401: no credentials")

    report = evals.run([_case("a", [1]), _case("b", [2])], failing, "no-key")
    assert report.error_rate == 1.0
    assert not report.usable
    assert "401" in report.first_error
    assert "NOT A MEASUREMENT" in report.render()
    # The misleading numbers must not be presented at all.
    assert "miss rate" not in report.render()


def test_unmeasurable_runs_are_never_ranked():
    from one_for_all.backends import BackendError

    def failing(case):
        raise BackendError("HTTP 401")

    cases = [_case("a", [1, 2])]
    good = evals.run(cases, lambda case: "1,2", "works")
    bad = evals.run(cases, failing, "no-key")

    table = evals.compare([bad, good])
    assert "best: works" in table
    assert "not measured" in table
    # A broken run must not be able to look like a second-place finish.
    assert table.index("works") < table.index("no-key")


def test_a_parse_failure_is_still_a_model_failure():
    """The distinction cuts both ways: bad output IS the model's fault."""
    report = evals.run([_case("a", [1])], lambda case: "no idea mate", "waffly")
    assert report.usable
    assert report.format_compliance == 0.0


def test_verdict_flags_thresholds_from_the_spec():
    cases = [_case("a", [1, 2, 3, 4])]
    report = evals.run(cases, evals.keep_all_runner(), "keep-all")
    assert any("below the 3x floor" in note for note in report.verdict())


# -- dataset export -------------------------------------------------------

def _log_curation(store, task, n_candidates, selected, outcome=None):
    context = json.dumps(
        {
            "task": task,
            "candidates": [
                {"id": i, "text": f"memory {i}"} for i in range(1, n_candidates + 1)
            ],
        }
    )
    did = store.log_decision("curate", context, ",".join(map(str, selected)))
    if outcome:
        store.record_outcome(did, outcome)
    return did


def test_export_reads_curations_back(store):
    _log_curation(store, "add retry logic to the upload path", 6, [1, 3])
    examples = dataset.load_curations(store)
    assert len(examples) == 1
    assert examples[0].selected == [1, 3]


def test_export_produces_the_training_shape(store, tmp_path):
    _log_curation(store, "add retry logic to the upload path", 6, [1, 3])
    kept, _ = dataset.curate_examples(dataset.load_curations(store))
    out = tmp_path / "train.jsonl"
    dataset.export_jsonl(kept, out)

    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert row["messages"][-1]["content"] == "1,3"
    assert "CANDIDATES:" in row["messages"][1]["content"]


def test_ambiguous_tasks_are_dropped(store):
    _log_curation(store, "help", 6, [1])
    _log_curation(store, "fix", 6, [1])
    kept, dropped = dataset.curate_examples(dataset.load_curations(store))
    assert kept == []
    assert dropped["ambiguous task"] == 2


def test_duplicates_are_dropped(store):
    for _ in range(3):
        _log_curation(store, "add retry logic to the upload path", 6, [1, 3])
    kept, dropped = dataset.curate_examples(dataset.load_curations(store))
    assert len(kept) == 1
    assert dropped["duplicate"] == 2


def test_keep_everything_rows_are_dropped_as_likely_fallbacks(store):
    _log_curation(store, "add retry logic to the upload path", 6, [1, 2, 3, 4, 5, 6])
    kept, dropped = dataset.curate_examples(dataset.load_curations(store))
    assert kept == []
    assert "kept everything (likely fallback)" in dropped


def test_every_recorded_miss_survives_every_filter(store):
    """MODEL.md §7: misses are the hard cases and carry the signal. They bypass
    the ambiguity, duplicate and balance filters."""
    for _ in range(3):
        _log_curation(store, "help", 2, [1], outcome="miss: needed the deploy path")
    kept, _ = dataset.curate_examples(dataset.load_curations(store))
    assert len(kept) == 3
    assert all(e.was_miss for e in kept)


def test_selection_sizes_are_balanced(store):
    """Unbalanced, the model learns the count instead of the criterion."""
    for i in range(40):
        _log_curation(store, f"a distinct task number {i} worth curating", 8, [1, 2, 3])
    for i in range(4):
        _log_curation(store, f"another distinct task number {i} here", 8, [1])

    kept, dropped = dataset.curate_examples(dataset.load_curations(store))
    sizes = {}
    for example in kept:
        sizes[len(example.selected)] = sizes.get(len(example.selected), 0) + 1
    assert sizes[3] < 40
    assert dropped["over-represented selection size"] > 0


def test_seed_eval_marks_itself_as_a_draft(store, tmp_path):
    _log_curation(store, "add retry logic to the upload path", 6, [1, 3])
    out = tmp_path / "eval.jsonl"
    dataset.seed_eval_cases(dataset.load_curations(store), out)

    text = out.read_text(encoding="utf-8")
    assert "NOT ground" in text
    cases = evals.load_cases(out)  # must round-trip into the eval harness
    assert cases[0].expected == {1, 3}


def test_seed_eval_puts_misses_first(store, tmp_path):
    """The prefilled answer is known-wrong on those, so they most need a human."""
    _log_curation(store, "a task that went fine and needs no attention", 6, [1])
    _log_curation(store, "a task where the curator dropped something", 6, [2], outcome="miss")

    out = tmp_path / "eval.jsonl"
    dataset.seed_eval_cases(dataset.load_curations(store), out)
    first = json.loads(
        [l for l in out.read_text(encoding="utf-8").splitlines() if not l.startswith("#")][0]
    )
    assert first["recorded_miss"] is True
