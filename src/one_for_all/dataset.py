"""Decision log → training data.

This is the payoff for logging from the first commit while nothing read the log.
After some months of ordinary use the training set already exists; nobody ever
sat down to label anything.

MODEL.md §7 is blunt about the part that actually matters: **do not train on the
raw log.** 500 clean examples beat 5,000 noisy ones, and 1,000 hand-curated beat
100,000 noisy, so the highest-leverage hour of the whole fine-tuning exercise is
spent filtering. The filters here implement that section:

- drop rows where the task was too vague to have a right answer
- drop duplicates and near-duplicates — they teach frequency, not judgment
- keep every recorded miss; those are the hard cases and carry the signal
- balance selection sizes, so the model does not learn "always pick 3 to 5"

Reaching for this module before `evals.py` has a populated eval set is the
failure MODEL.md names: not a fine-tune, a ritual.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .curator import SYSTEM_PROMPT
from .store import Store

# A task string shorter than this cannot carry enough intent for "which
# memories are relevant" to have a defensible answer. Training on it teaches
# the model to guess confidently, which is the opposite of the goal.
MIN_TASK_CHARS = 12

# Selection is only a real decision when there is something to select *from*.
MIN_CANDIDATES = 4

# Above this share of one selection size, examples are downsampled. Left
# unbalanced, a log where most curations keep 3-5 items teaches the count
# rather than the criterion.
MAX_SIZE_SHARE = 0.4

_VAGUE = re.compile(
    r"^(help|fix|do it|continue|go on|thanks|ok|yes|no|hmm|what|why|test)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Example:
    """One curation decision, in the shape a chat fine-tune consumes."""

    task: str
    candidates: list[dict]
    selected: list[int]
    was_miss: bool

    def to_messages(self) -> dict:
        prompt = "\n".join(
            [f"TASK: {self.task}", "", "CANDIDATES:"]
            + [f"[{c['id']}] {c['text']}" for c in self.candidates]
        )
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ",".join(map(str, self.selected))},
            ]
        }

    @property
    def fingerprint(self) -> str:
        """Near-duplicate key: same task against the same candidate set is the
        same lesson however many times it was logged."""
        ids = ",".join(str(c["id"]) for c in sorted(self.candidates, key=lambda c: c["id"]))
        return f"{' '.join(self.task.lower().split())}|{ids}"


def load_curations(store: Store, limit: int | None = None) -> list[Example]:
    """Read curation decisions back out of the log.

    Rows written before a curation had an outcome are still usable — an
    unlabelled row is a decision that was never contradicted, which is weak
    positive evidence. Rows explicitly labelled `miss` are the valuable ones.
    """
    sql = "SELECT id, context, decision, outcome FROM decisions WHERE kind = 'curate' ORDER BY id DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"

    examples = []
    for row in store.db.execute(sql).fetchall():
        try:
            context = json.loads(row["context"])
        except (ValueError, TypeError):
            continue  # a row we cannot replay is not instrumentation
        candidates = context.get("candidates") or []
        task = (context.get("task") or "").strip()
        if not task or not candidates:
            continue

        outcome = (row["outcome"] or "").lower()
        was_miss = outcome.startswith("miss")

        selected = [int(i) for i in re.findall(r"\d+", row["decision"] or "")]
        valid = {c["id"] for c in candidates}
        selected = [i for i in selected if i in valid]

        examples.append(
            Example(
                task=task,
                candidates=candidates,
                selected=selected,
                was_miss=was_miss,
            )
        )
    return examples


def curate_examples(
    examples: list[Example],
    seed: int = 0,
    keep_fallbacks: bool = False,
) -> tuple[list[Example], dict[str, int]]:
    """Apply MODEL.md §7's filters. Returns the kept examples and a count of
    what each filter removed, because a filter you cannot see is a filter you
    cannot tune."""
    rng = random.Random(seed)
    dropped: Counter[str] = Counter()

    survivors: list[Example] = []
    seen: set[str] = set()

    for example in examples:
        # Misses are kept unconditionally and bypass every filter below. They
        # are the hard cases, they are rare, and they carry the signal — losing
        # one to a tidiness rule is a bad trade.
        if example.was_miss:
            survivors.append(example)
            continue

        if len(example.task) < MIN_TASK_CHARS or _VAGUE.match(example.task):
            dropped["ambiguous task"] += 1
            continue
        if len(example.candidates) < MIN_CANDIDATES:
            dropped["too few candidates"] += 1
            continue
        if not keep_fallbacks and len(example.selected) == len(example.candidates):
            # A keep-everything row is usually the curator's fallback path, not
            # a judgement. Training on it teaches the model to select all.
            dropped["kept everything (likely fallback)"] += 1
            continue
        if example.fingerprint in seen:
            dropped["duplicate"] += 1
            continue

        seen.add(example.fingerprint)
        survivors.append(example)

    balanced, capped = _balance_by_size(survivors, rng)
    if capped:
        dropped["over-represented selection size"] = capped
    return balanced, dict(dropped)


def _balance_by_size(
    examples: list[Example], rng: random.Random
) -> tuple[list[Example], int]:
    """Cap any one selection size at MAX_SIZE_SHARE of the set.

    If 90% of examples select 3-5 items, the model learns to always select 3-5
    items — a rule that looks like accuracy on a log drawn from the same
    distribution and fails the moment the right answer is 1 or 11.
    """
    if not examples:
        return [], 0

    by_size: dict[int, list[Example]] = {}
    for example in examples:
        by_size.setdefault(len(example.selected), []).append(example)

    # Misses are never dropped for balance, so the cap is computed against the
    # examples that are actually eligible for downsampling.
    cap = max(1, int(len(examples) * MAX_SIZE_SHARE))
    kept: list[Example] = []
    removed = 0
    for group in by_size.values():
        if len(group) <= cap:
            kept.extend(group)
            continue
        misses = [e for e in group if e.was_miss]
        rest = [e for e in group if not e.was_miss]
        rng.shuffle(rest)
        allowed = max(0, cap - len(misses))
        kept.extend(misses + rest[:allowed])
        removed += max(0, len(rest) - allowed)

    rng.shuffle(kept)
    return kept, removed


def export_jsonl(examples: list[Example], path: Path) -> int:
    """Write the training file. One `{"messages": [...]}` per line — the same
    shape as an API request, which is what every fine-tuning toolchain reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_messages(), ensure_ascii=False) + "\n")
    return len(examples)


def seed_eval_cases(examples: list[Example], path: Path, count: int = 50) -> int:
    """Write a *draft* eval set for a human to correct.

    `expected` is prefilled with what the curator actually chose, which is
    exactly what the eval is supposed to be judging — so this file is worthless
    until someone reads every line and fixes the wrong ones. That editing pass
    is the afternoon MODEL.md §8 budgets for, and it is not skippable: an eval
    set that agrees with the model by construction measures nothing.

    Recorded misses are placed first, since those are the cases where the
    prefilled answer is known to be wrong and most needs a human.
    """
    ordered = sorted(examples, key=lambda e: not e.was_miss)[:count]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "# DRAFT eval set. `expected` is what the curator chose, NOT ground\n"
            "# truth. Correct every line before trusting any number from it.\n"
            "# Lines marked recorded_miss are known-wrong: fix those first.\n"
        )
        for index, example in enumerate(ordered, 1):
            handle.write(
                json.dumps(
                    {
                        "id": f"case-{index:02d}",
                        "task": example.task,
                        "candidates": example.candidates,
                        "expected": example.selected,
                        "recorded_miss": example.was_miss,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(ordered)


def summarise(examples: list[Example], dropped: dict[str, int]) -> str:
    sizes = Counter(len(e.selected) for e in examples)
    misses = sum(1 for e in examples if e.was_miss)
    lines = [
        f"{len(examples)} examples kept ({misses} recorded misses)",
        "selection sizes: "
        + ", ".join(f"{size}→{n}" for size, n in sorted(sizes.items())),
    ]
    if dropped:
        lines.append("dropped:")
        lines += [f"  {reason}: {n}" for reason, n in sorted(dropped.items())]
    if len(examples) < 500:
        lines.append(
            f"\nNote: MODEL.md §7 wants 500-2,000 examples. At {len(examples)} this is "
            "not yet a training set — keep using the daemon, the log fills itself."
        )
    return "\n".join(lines)
