"""Curator — the volume tier.

Selects which memories go into a request. It runs on every single call, so its
cost is multiplied by everything, which is why MODEL.md §1.1 caps it at 1–4B and
a 300 ms latency budget. It selects; it does not reason, judge, or generate.

Three properties make this module worth more than the model behind it:

*Recall over precision.* Dropping a memory the model needed is far worse than
keeping one it did not. Every failure path here — no backend, malformed output,
an id that was never offered — falls back to keeping everything. A curator that
fails open costs tokens; one that fails closed costs a wrong answer.

*It knows when not to run.* Calling a model to filter six candidates costs more
than it saves. Below `min_candidates` the selection is skipped entirely.

*It is self-labelling.* Every call writes a decision-log row rich enough to
replay as a training example. When the downstream model later goes looking for
something the curator dropped, `record_miss` turns that row into a corrected
example — no labelling effort, just ordinary use. This is why the curator is the
one role worth fine-tuning (MODEL.md §7) and the critic is not.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import Config, load as load_config
from .router import NoBackendAvailable, Router
from .store import Memory, Store

# The output contract. Small models fail on format before they fail on
# judgment, so the instruction is short, the example is concrete, and there is
# an explicit escape hatch for "all of them" — without one, a model asked for
# ids will invent prose to explain that everything is relevant.
SYSTEM_PROMPT = """You select which stored memories are relevant to a task.

Reply with ONLY the ids of the relevant memories, comma-separated.
Reply with ALL if every memory is relevant, or NONE if none are.
No explanation, no other text.

Example reply: 3,7,12"""

_ID_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class Curation:
    """The result of one selection, plus everything needed to score it."""

    selected: list[Memory]
    candidates: list[Memory]
    decision_id: int | None
    model: str | None
    latency_s: float
    # Set when selection was skipped or a failure forced the keep-everything
    # fallback. A run with a reason attached spent no model call and should not
    # be read as evidence the curator is working.
    fallback_reason: str | None = None

    @property
    def compression_ratio(self) -> float:
        """Candidate chars ÷ selected chars. MODEL.md §2 puts the floor at ~3×;
        below that the curator is not earning its complexity."""
        kept = sum(len(m.text) for m in self.selected)
        offered = sum(len(m.text) for m in self.candidates)
        if not kept:
            return float(offered) if offered else 1.0
        return offered / kept

    def render(self) -> str:
        return "\n".join(m.render() for m in self.selected)


class Curator:
    def __init__(
        self,
        router: Router,
        store: Store,
        config: Config | None = None,
    ) -> None:
        self.router = router
        self.store = store
        self.config = config or load_config()
        self._settings = self.config.raw["curator"]

    def curate(self, task: str, candidates: list[Memory]) -> Curation:
        """Narrow `candidates` to those relevant to `task`."""
        if not self._settings.get("enabled", True):
            return self._keep_all(task, candidates, "curator disabled")
        if len(candidates) < int(self._settings.get("min_candidates", 8)):
            # Not enough to be worth a round trip. This is the common case for
            # a young store, and it is the correct behaviour, not a shortfall.
            return self._keep_all(task, candidates, "below min_candidates")

        prompt = _build_prompt(task, candidates)
        try:
            completion = self.router.complete(
                "curate",
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                # Memory contents are exactly what the privacy gate exists to
                # protect, so the curator is never allowed to assume otherwise.
                sensitive=self.router.is_sensitive(task, *(m.text for m in candidates)),
            )
        except NoBackendAvailable as exc:
            return self._keep_all(task, candidates, f"no backend: {exc}")

        selected = parse_selection(completion.text, candidates)
        if selected is None:
            # Format drift is the first symptom of a model that is too small or
            # too aggressively quantized (MODEL.md §6). Logged as a distinct
            # reason so the eval set can count it separately from a bad pick.
            return self._keep_all(
                task, candidates, "unparseable selection", model=completion.model
            )

        decision_id = self._log(task, candidates, selected, completion.model)
        return Curation(
            selected=selected,
            candidates=candidates,
            decision_id=decision_id,
            model=completion.model,
            latency_s=completion.latency_s,
        )

    def record_miss(self, decision_id: int, note: str = "") -> None:
        """Label a curation as having dropped something that was needed.

        This is the automatic training signal from MODEL.md §1.1: the label is
        generated by the product working, not by a labelling effort. Misses are
        the hard cases and §7 says to keep every one of them.
        """
        self.store.record_outcome(decision_id, f"miss{': ' + note if note else ''}")

    def record_hit(self, decision_id: int) -> None:
        """Label a curation as sufficient — the downstream model never went
        looking for anything else. The positive examples come from here."""
        self.store.record_outcome(decision_id, "hit")

    # -- internals --------------------------------------------------------

    def _keep_all(
        self,
        task: str,
        candidates: list[Memory],
        reason: str,
        model: str | None = None,
    ) -> Curation:
        decision_id = self._log(task, candidates, candidates, model, reason)
        return Curation(
            selected=list(candidates),
            candidates=candidates,
            decision_id=decision_id,
            model=model,
            latency_s=0.0,
            fallback_reason=reason,
        )

    def _log(
        self,
        task: str,
        candidates: list[Memory],
        selected: list[Memory],
        model: str | None,
        reason: str | None = None,
    ) -> int | None:
        """Write the row that a training example is later rebuilt from.

        The candidate *texts* are stored, not just their ids, because a memory
        can be superseded or forgotten between the decision and the day someone
        exports a dataset. A row that cannot be replayed is not instrumentation.
        """
        if not candidates:
            return None
        context = json.dumps(
            {
                "task": task,
                "candidates": [{"id": m.id, "text": m.text} for m in candidates],
                "model": model,
                "fallback": reason,
            },
            ensure_ascii=False,
        )
        return self.store.log_decision(
            kind="curate",
            context=context,
            decision=",".join(str(m.id) for m in selected) or "none",
        )


def _build_prompt(task: str, candidates: list[Memory]) -> str:
    lines = [f"TASK: {task}", "", "CANDIDATES:"]
    lines += [f"[{m.id}] {m.text}" for m in candidates]
    return "\n".join(lines)


def parse_selection(reply: str, candidates: list[Memory]) -> list[Memory] | None:
    """Turn a model reply into memories, or None if the contract was broken.

    Public because the eval harness must score the *production* parser. An eval
    with its own lenient copy would report a format-compliance number that the
    running system never achieves.

    Returning None rather than an empty list matters: "the model said none are
    relevant" and "the model did not answer in the requested format" are
    different events, and only the second is a format-compliance failure.
    """
    text = reply.strip()
    if not text:
        return None

    upper = text.upper()
    if "ALL" in upper and not _ID_RE.search(text):
        return list(candidates)
    if "NONE" in upper and not _ID_RE.search(text):
        return []

    by_id = {m.id: m for m in candidates}
    ids = [int(match) for match in _ID_RE.findall(text)]
    if not ids:
        return None

    # Ids the model invented are dropped rather than treated as a hard failure:
    # a reply of "3, 7, 99" against candidates 1–20 is a mostly-correct answer
    # with one hallucination, and discarding the good picks helps nobody.
    picked = [by_id[i] for i in dict.fromkeys(ids) if i in by_id]
    return picked if picked else None
