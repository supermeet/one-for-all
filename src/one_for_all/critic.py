"""Critic — the judgment tier, and the differentiator.

Not a prompt saying "be careful". A genuine second pass with a different job:
read the output cold and ask whether there is a bug, whether a library call
replaces this, and whether it matches how *this* user writes.

The design constraint that shapes everything here is that **the critic's fatal
failure mode is the false positive.** One confident wrong finding trains the
user to skim past the next one, and a critic that is skimmed past is a critic
that gets turned off. So:

- It is quiet by default. Findings below the confidence bar are logged and
  never shown. Saying nothing is a valid, common, correct output.
- It starts with one narrow job (bug / simpler-with-a-library) rather than a
  checklist. BUILD_PLAN §3 is explicit: resist adding dimensions until that one
  is calibrated.
- Its ignore-rate is a metric, not an opinion. Every finding is logged with an
  outcome slot so "is this thing actually useful?" is answerable from data.

It is also the role never to fine-tune. Judgment is what small models are worst
at, and "was this critique good?" has no cheap automatic label — so capability
is bought, not trained (MODEL.md §1.2).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import Config, load as load_config
from .router import NoBackendAvailable, Router
from .store import Memory, Store

FINDING_KINDS = ("bug", "simpler", "style")

SYSTEM_PROMPT = """You are a code critic. You read code cold and report only \
what is genuinely wrong or genuinely better.

Report a finding ONLY for:
  bug     - it is incorrect, will crash, or mishandles an edge case
  simpler - a standard-library or already-imported call replaces hand-written code
  style   - it contradicts a stated preference of this user, quoted below

Do NOT report: naming opinions, formatting, speculative refactors, missing
tests, or anything you are not confident about. Reporting nothing is the
correct answer for most code. An unsure finding is worse than silence.

Reply with a JSON array, and nothing else:
[{"kind": "bug", "confidence": 0.9, "summary": "one line", "detail": "why, and the fix"}]

confidence is 0.0-1.0: your probability that a competent reviewer agrees.
Reply with [] when you have nothing worth saying."""


@dataclass(frozen=True)
class Finding:
    kind: str
    confidence: float
    summary: str
    detail: str

    def render(self) -> str:
        return f"[{self.kind} {self.confidence:.0%}] {self.summary}\n    {self.detail}"

    def as_memory_text(self) -> str:
        """Every critic finding is candidate memory — that is how teaching
        becomes cumulative rather than per-session advice (ARCHITECTURE §3.2).
        Stored as the lesson, not the incident, so it is still useful in a file
        this code never touches."""
        return f"{self.summary} — {self.detail}"


@dataclass(frozen=True)
class Critique:
    findings: list[Finding]
    suppressed: list[Finding]
    decision_id: int | None
    model: str | None
    latency_s: float
    error: str | None = None

    @property
    def spoke(self) -> bool:
        return bool(self.findings)

    def render(self) -> str:
        if self.error:
            return ""
        if not self.findings:
            return ""
        return "\n".join(f.render() for f in self.findings)


class Critic:
    def __init__(
        self,
        router: Router,
        store: Store,
        config: Config | None = None,
    ) -> None:
        self.router = router
        self.store = store
        self.config = config or load_config()

    def review(
        self,
        code: str,
        context: str = "",
        memories: list[Memory] | None = None,
    ) -> Critique:
        """Read `code` cold and report only what clears the confidence bar.

        Never raises. The critic runs in the background and speaks only when it
        has something; a backend that is down means it has no opinion, which is
        not a failure the user needs to hear about (MODEL.md §9).
        """
        if not self.config.raw["critic"].get("enabled", True):
            return Critique([], [], None, None, 0.0, error="critic disabled")

        prompt = _build_prompt(code, context, memories or [])
        try:
            completion = self.router.complete(
                "critique",
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        except NoBackendAvailable as exc:
            return Critique([], [], None, None, 0.0, error=str(exc))

        parsed = _parse_findings(completion.text)
        if parsed is None:
            return Critique(
                [], [], None, completion.model, completion.latency_s,
                error="unparseable critique",
            )

        bar = self.config.confidence_bar
        findings = [f for f in parsed if f.confidence >= bar]
        suppressed = [f for f in parsed if f.confidence < bar]

        decision_id = self._log(code, findings, suppressed, completion.model)
        return Critique(
            findings=findings,
            suppressed=suppressed,
            decision_id=decision_id,
            model=completion.model,
            latency_s=completion.latency_s,
        )

    def record_acted(self, decision_id: int) -> None:
        """The user acted on the finding. A true positive."""
        self.store.record_outcome(decision_id, "acted")

    def record_ignored(self, decision_id: int) -> None:
        """The user ignored the finding — labelled a false positive.

        ARCHITECTURE §9 lists critic fatigue as a named risk with 'you turned it
        off' as its signal. This is the measurement that turns that from a
        feeling into an ignore-rate, which is what the confidence bar is
        eventually tuned against.
        """
        self.store.record_outcome(decision_id, "ignored")

    def teach(self, finding: Finding, source: str = "critic") -> int:
        """Promote a finding to a durable memory.

        Deliberately not automatic. Auto-writing every finding is the fastest
        route to the landfill that ARCHITECTURE §7 names as the #1 risk — the
        write policy is the product, and a finding the user never confirmed is
        exactly the kind of noise that poisons retrieval.
        """
        return self.store.remember(
            finding.as_memory_text()[:1900],
            scope="procedural" if finding.kind == "style" else "semantic",
            source=source,
        )

    def _log(
        self,
        code: str,
        findings: list[Finding],
        suppressed: list[Finding],
        model: str | None,
    ) -> int | None:
        # Suppressed findings are logged too. Whether the bar is set right is
        # answerable only if the things it filtered out were written down.
        payload = json.dumps(
            {
                "model": model,
                "shown": [f.summary for f in findings],
                "suppressed": [
                    {"summary": f.summary, "confidence": f.confidence}
                    for f in suppressed
                ],
            },
            ensure_ascii=False,
        )
        return self.store.log_decision(
            kind="critique",
            context=code[:4000],
            decision=payload,
        )


def _build_prompt(code: str, context: str, memories: list[Memory]) -> str:
    parts = []
    if context:
        parts.append(f"CONTEXT: {context}")
    if memories:
        # The critic needs the code *and* the relevant memories — the "does it
        # match how this user writes" question is unanswerable without them,
        # and it is the one question a generic linter cannot ask.
        parts.append(
            "STATED PREFERENCES OF THIS USER:\n"
            + "\n".join(f"- {m.text}" for m in memories)
        )
    parts.append(f"CODE:\n{code}")
    return "\n\n".join(parts)


def _parse_findings(reply: str) -> list[Finding] | None:
    """Extract findings from a reply, or None if it was not parseable.

    Models wrap JSON in prose and fences no matter how firmly asked not to, so
    the first bracketed array in the reply is taken rather than requiring the
    whole response to be valid JSON.
    """
    text = reply.strip()
    if not text:
        return None

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        # A model with nothing to say sometimes says so in words instead of
        # returning []. That is a clean result, not a format failure.
        if re.search(r"\bno (issues|findings|problems)\b", text, re.IGNORECASE):
            return []
        return None

    try:
        raw = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(raw, list):
        return None

    findings = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("summary"):
            continue
        kind = str(item.get("kind", "bug")).lower()
        findings.append(
            Finding(
                kind=kind if kind in FINDING_KINDS else "bug",
                # An unparseable confidence is treated as 0.0, so it lands below
                # any sane bar and stays silent. The failure direction is
                # silence, never a false positive.
                confidence=_clamp(item.get("confidence")),
                summary=str(item["summary"]).strip(),
                detail=str(item.get("detail", "")).strip(),
            )
        )
    return findings


def _clamp(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
