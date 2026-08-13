"""Command line for the model layer.

The MCP server is how the daemon is *used*; this is how it is inspected, scored,
and tuned. Everything here is deliberately outside the request path — running an
eval or exporting a dataset must never be something that happens while someone
is waiting for an answer.

    one-for-all-cli status                    what is reachable right now
    one-for-all-cli eval cases.jsonl          score the routed curator
    one-for-all-cli eval cases.jsonl --compare-all   rank every eligible model
    one-for-all-cli export train.jsonl        decision log -> training data
    one-for-all-cli seed-eval draft.jsonl     draft an eval set from the log
    one-for-all-cli config --write            materialise the config file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config as config_mod
from . import dataset, evals
from .backends import BackendUnavailable
from .router import Router
from .store import Store


def _router(store: Store) -> Router:
    return Router(config=config_mod.load(), store=store)


def cmd_status(args: argparse.Namespace) -> int:
    store = Store()
    print(_router(store).status())
    print()
    print("store:", json.dumps(store.stats()))
    print("config:", config_mod.config_path(), end="")
    print("" if config_mod.config_path().exists() else "  (not written — using defaults)")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cases = evals.load_cases(Path(args.cases))
    if not cases:
        print("no cases loaded", file=sys.stderr)
        return 1

    store = Store()
    router = _router(store)
    reports = []

    # The free baseline always runs. A curator that does not beat keep-all on
    # the composite is spending latency to achieve nothing, and that is the
    # single most useful thing an eval can tell you.
    reports.append(evals.run(cases, evals.keep_all_runner(), "baseline:keep-all"))

    pairs = router.candidates("curate")
    if not pairs:
        print("no curate-eligible model is reachable; baseline only\n", file=sys.stderr)
    elif args.compare_all:
        for backend, model in pairs[: args.limit]:
            print(f"running {backend.name}/{model.id} ...", file=sys.stderr)
            reports.append(
                evals.run(
                    cases,
                    evals.model_runner(backend, model.id),
                    f"{backend.name}/{model.id}",
                )
            )
    else:
        backend, model = pairs[0]
        print(f"running {backend.name}/{model.id} ...", file=sys.stderr)
        reports.append(
            evals.run(cases, evals.model_runner(backend, model.id), f"{backend.name}/{model.id}")
        )

    for report in reports:
        print(report.render())
        print()
    print(evals.compare(reports))

    if args.json:
        Path(args.json).write_text(
            json.dumps([r.as_dict() for r in reports], indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    store = Store()
    raw = dataset.load_curations(store, limit=args.limit)
    kept, dropped = dataset.curate_examples(raw, seed=args.seed)
    written = dataset.export_jsonl(kept, Path(args.out))
    print(f"read {len(raw)} logged curations")
    print(dataset.summarise(kept, dropped))
    print(f"\nwrote {written} examples to {args.out}")
    return 0


def cmd_seed_eval(args: argparse.Namespace) -> int:
    store = Store()
    raw = dataset.load_curations(store)
    # No filtering here: the eval set wants the messy real cases, especially the
    # ones the training filters would discard.
    written = dataset.seed_eval_cases(raw, Path(args.out), count=args.count)
    print(f"wrote {written} DRAFT cases to {args.out}")
    print("`expected` is what the curator chose, not ground truth.")
    print("Read every line and correct it before trusting a single number.")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    if args.write:
        path = config_mod.write_default()
        print(f"wrote {path}")
        return 0
    print(json.dumps(config_mod.load().raw, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="one-for-all-cli", description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)

    p = subs.add_parser("status", help="show reachable backends and role picks")
    p.set_defaults(func=cmd_status)

    p = subs.add_parser("eval", help="score the curator against an eval set")
    p.add_argument("cases", help="JSONL eval set")
    p.add_argument("--compare-all", action="store_true", help="rank every eligible model")
    p.add_argument("--limit", type=int, default=5, help="max models with --compare-all")
    p.add_argument("--json", help="also write the metrics to this file")
    p.set_defaults(func=cmd_eval)

    p = subs.add_parser("export", help="decision log -> training JSONL")
    p.add_argument("out")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_export)

    p = subs.add_parser("seed-eval", help="draft an eval set from the log")
    p.add_argument("out")
    p.add_argument("--count", type=int, default=50)
    p.set_defaults(func=cmd_seed_eval)

    p = subs.add_parser("config", help="show or write the config file")
    p.add_argument("--write", action="store_true")
    p.set_defaults(func=cmd_config)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (BackendUnavailable, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
