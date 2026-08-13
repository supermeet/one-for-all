"""The scoreboard — the prerequisite for every other decision in the model layer.

MODEL.md §8 states the rule this module exists to enforce: **any model change is
a hypothesis, tested against this set before adoption.** New model, new
quantization, new prompt, new adapter — no exceptions, including for changes
that are obviously better. Without this, model choice is taste, quantization is
superstition, and a fine-tune is a ritual with a GPU bill.

The eval set is ~50 hand-labelled cases drawn from your own logs. Fifty is
enough to detect a real difference and small enough to build in an afternoon.
`dataset.seed_eval_cases` writes the first draft from the decision log so the
job is editing rather than authoring.

Case format, one JSON object per line:

    {"id": "case-01",
     "task": "add retry logic to the upload path",
     "candidates": [{"id": 3, "text": "prefers stdlib over new deps"}, ...],
     "expected": [3, 12]}
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .backends import Backend, BackendError
from .curator import SYSTEM_PROMPT, parse_selection
from .store import Memory

# A single number is needed to rank candidates, and any single number encodes a
# judgement about which failure hurts more. MODEL.md §8 says the miss is the
# failure that matters and is asymmetric, so it is weighted 3× the noise rate.
# Format failures cost a retry, which erases the savings that justify curation
# at all, so they are weighted 2×. These weights are a choice, written down
# here rather than buried, and worth revisiting once real use says otherwise.
MISS_WEIGHT = 3.0
NOISE_WEIGHT = 1.0
FORMAT_WEIGHT = 2.0


@dataclass(frozen=True)
class EvalCase:
    id: str
    task: str
    candidates: list[Memory]
    expected: set[int]


@dataclass
class CaseResult:
    case_id: str
    selected: set[int] | None  # None = the output contract was broken
    expected: set[int]
    latency_s: float
    chars_in: int
    chars_out: int
    # Set when the backend itself failed — a 401, a timeout, an unreachable
    # host. Kept distinct from a parse failure because they demand opposite
    # responses: a format failure means change the model, an error means fix
    # your setup. Conflating them is how a missing API key gets misread as
    # "this model scores terribly". (Found exactly that way on the first run.)
    error: str | None = None

    @property
    def parsed(self) -> bool:
        return self.selected is not None

    @property
    def missed(self) -> set[int]:
        return self.expected - (self.selected or set())

    @property
    def noise(self) -> set[int]:
        return (self.selected or set()) - self.expected


@dataclass
class Report:
    """The five metrics from MODEL.md §8, plus enough detail to debug them."""

    label: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def miss_rate(self) -> float:
        """Needed memories that were not selected. The failure that matters."""
        needed = sum(len(r.expected) for r in self.results)
        missed = sum(len(r.missed) for r in self.results)
        return missed / needed if needed else 0.0

    @property
    def noise_rate(self) -> float:
        """Selected memories that were not needed. Costs tokens, dilutes
        attention, but never produces a wrong answer on its own."""
        picked = sum(len(r.selected or set()) for r in self.results)
        noise = sum(len(r.noise) for r in self.results)
        return noise / picked if picked else 0.0

    @property
    def error_rate(self) -> float:
        """Cases where the backend failed rather than answered."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.error) / len(self.results)

    @property
    def usable(self) -> bool:
        """Whether this run is a measurement at all.

        A run with backend errors in it is not a bad score, it is an absent
        one, and reporting it as a ranking is worse than reporting nothing —
        it invites a decision to be made on a number that means 'your key is
        missing'. The threshold is deliberately strict.
        """
        return self.error_rate <= 0.1

    @property
    def first_error(self) -> str | None:
        return next((r.error for r in self.results if r.error), None)

    @property
    def format_compliance(self) -> float:
        """Fraction parsed on the first attempt. Small models fail here first,
        and quantization damage shows up as format drift before it shows up as
        worse judgement — which is why this is measured, not assumed."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.parsed) / len(self.results)

    @property
    def p95_latency_s(self) -> float:
        """The tail, not the mean. The curator sits in the critical path of
        every request, so the slow case is the one the user feels."""
        if not self.results:
            return 0.0
        ordered = sorted(r.latency_s for r in self.results)
        index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return ordered[index]

    @property
    def compression_ratio(self) -> float:
        """Chars offered ÷ chars kept. Must clear ~3× to justify existing."""
        kept = sum(r.chars_out for r in self.results)
        offered = sum(r.chars_in for r in self.results)
        if not kept:
            return float(offered) if offered else 1.0
        return offered / kept

    @property
    def penalty(self) -> float:
        """Composite, lower is better. See the weights at the top of the file."""
        return (
            MISS_WEIGHT * self.miss_rate
            + NOISE_WEIGHT * self.noise_rate
            + FORMAT_WEIGHT * (1.0 - self.format_compliance)
        )

    def verdict(self) -> list[str]:
        """The thresholds from MODEL.md, checked rather than remembered."""
        notes = []
        if self.compression_ratio < 3.0:
            notes.append(
                f"compression {self.compression_ratio:.1f}x is below the 3x floor — "
                "curation is not earning its complexity here (MODEL.md §2)"
            )
        if self.p95_latency_s > 0.3:
            notes.append(
                f"p95 {self.p95_latency_s*1000:.0f}ms exceeds the 300ms budget — "
                "above that it costs more than it saves (MODEL.md §1.1)"
            )
        if self.format_compliance < 0.95:
            notes.append(
                f"format compliance {self.format_compliance:.0%} — a retry erases "
                "the savings; suspect the model is too small or over-quantized"
            )
        return notes

    def render(self) -> str:
        if not self.usable:
            return (
                f"=== {self.label} ({len(self.results)} cases) ===\n"
                f"  NOT A MEASUREMENT: {self.error_rate:.0%} of cases failed at the "
                f"backend.\n"
                f"  first error: {self.first_error}\n"
                f"  Fix the setup and re-run. The metrics below would describe your "
                f"configuration, not this model, so they are not shown."
            )
        lines = [
            f"=== {self.label} ({len(self.results)} cases) ===",
            f"  miss rate          {self.miss_rate:.1%}   (weighted heaviest)",
            f"  noise rate         {self.noise_rate:.1%}",
            f"  format compliance  {self.format_compliance:.1%}",
            f"  p95 latency        {self.p95_latency_s*1000:.0f} ms",
            f"  compression        {self.compression_ratio:.1f}x",
            f"  penalty            {self.penalty:.3f}  (lower is better)",
        ]
        lines += [f"  ! {note}" for note in self.verdict()]
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "cases": len(self.results),
            "usable": self.usable,
            "error_rate": self.error_rate,
            "first_error": self.first_error,
            "miss_rate": self.miss_rate,
            "noise_rate": self.noise_rate,
            "format_compliance": self.format_compliance,
            "p95_latency_s": self.p95_latency_s,
            "compression_ratio": self.compression_ratio,
            "penalty": self.penalty,
        }


class Runner(Protocol):
    """Anything that answers a curation case. Keeping this an interface is what
    lets the same eval score a hosted model, a local model, a LoRA adapter, or
    a non-model baseline without special-casing any of them."""

    def __call__(self, case: EvalCase) -> str: ...


def load_cases(path: Path) -> list[EvalCase]:
    cases = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: not valid JSON: {exc}") from exc
        cases.append(
            EvalCase(
                id=str(raw.get("id", f"case-{line_no}")),
                task=raw["task"],
                candidates=[
                    Memory(
                        id=int(c["id"]),
                        scope=c.get("scope", "semantic"),
                        text=c["text"],
                        source=None,
                        created_at=0.0,
                    )
                    for c in raw["candidates"]
                ],
                expected={int(i) for i in raw["expected"]},
            )
        )
    return cases


def run(cases: list[EvalCase], runner: Runner, label: str) -> Report:
    report = Report(label=label)
    for case in cases:
        started = time.perf_counter()
        error: str | None = None
        reply = ""
        try:
            reply = runner(case)
        except BackendError as exc:
            # The run continues rather than aborting on one flaky call, but the
            # failure is recorded as an error, never as a wrong answer. A model
            # that was never reached has not been measured.
            error = str(exc)
        latency = time.perf_counter() - started

        picked = None if error else parse_selection(reply, case.candidates)
        selected = {m.id for m in picked} if picked is not None else None
        by_id = {m.id: m for m in case.candidates}
        report.results.append(
            CaseResult(
                case_id=case.id,
                selected=selected,
                expected=case.expected,
                latency_s=latency,
                chars_in=sum(len(m.text) for m in case.candidates),
                chars_out=sum(len(by_id[i].text) for i in (selected or set()) if i in by_id),
                error=error,
            )
        )
    return report


def model_runner(
    backend: Backend,
    model_id: str,
    system_prompt: str = SYSTEM_PROMPT,
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> Runner:
    """Score one specific model, pinned. Bypasses the router deliberately — an
    eval that lets the router choose measures the router, not the model."""

    def _run(case: EvalCase) -> str:
        prompt = "\n".join(
            [f"TASK: {case.task}", "", "CANDIDATES:"]
            + [f"[{m.id}] {m.text}" for m in case.candidates]
        )
        return backend.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    return _run


def keep_all_runner() -> Runner:
    """The baseline that any curator must beat.

    It has a perfect miss rate and 1× compression, which is exactly the tradeoff
    curation is proposing to make. A model that does not clearly beat this on
    the composite is a model that is costing latency for nothing — and this
    baseline is free to run, so there is no excuse for not knowing.
    """

    def _run(case: EvalCase) -> str:
        return ",".join(str(m.id) for m in case.candidates)

    return _run


def compare(reports: list[Report]) -> str:
    """Rank runs by penalty. This is the artefact that turns 'the new model
    feels better' into a claim someone can check.

    Runs that failed at the backend are listed but never ranked. Sorting them
    in would put a missing API key at the bottom of a quality table, which
    reads as a verdict on the model and is not one.
    """
    usable = sorted([r for r in reports if r.usable], key=lambda r: r.penalty)
    broken = [r for r in reports if not r.usable]

    width = max((len(r.label) for r in reports), default=10)
    lines = [
        f"{'model':<{width}}  {'penalty':>8} {'miss':>7} {'noise':>7} "
        f"{'format':>7} {'p95':>8} {'compr':>7}"
    ]
    for report in usable:
        lines.append(
            f"{report.label:<{width}}  {report.penalty:>8.3f} "
            f"{report.miss_rate:>6.1%} {report.noise_rate:>6.1%} "
            f"{report.format_compliance:>6.1%} "
            f"{report.p95_latency_s*1000:>6.0f}ms {report.compression_ratio:>6.1f}x"
        )
    for report in broken:
        lines.append(f"{report.label:<{width}}  {'— not measured —':>8}")

    if usable:
        lines.append(f"\nbest: {usable[0].label}")
    if broken:
        lines.append(
            f"\n{len(broken)} run(s) failed at the backend and were not ranked:"
        )
        lines += [f"  {r.label}: {r.first_error}" for r in broken]
    return "\n".join(lines)
