# one-for-all

**A private, local memory and judgment layer that any AI client can borrow, over MCP.**

Models are commodities. Access to them is nearly free and any one can be swapped
for another in a config change. What is *not* commodity is the accumulated
context of one person's work — how you think, what you have already decided, what
you keep getting wrong, what your projects assume.

Today that context is either absent (every session starts cold) or owned by a
vendor (locked in their product, unexportable, readable by them).

**one-for-all is a local daemon that owns that context and lends it to whatever
AI you already use.** It does not replace your tools; it makes them yours.

Nothing leaves your machine unless you allow that specific content, for that
specific request. Privacy here is structural, not a policy promise.

---

## How it works

You keep using Claude Code, Cursor, Copilot or Gemini CLI. They call the daemon.

```
┌──────────────────────────────────────────────────────────┐
│  YOUR CLIENT   Claude Code · Cursor · Copilot · Gemini    │
└─────────────────────────┬────────────────────────────────┘
                          │  MCP (stdio — no port, no server to start)
┌─────────────────────────▼────────────────────────────────┐
│  ONE-FOR-ALL                                             │
│                                                          │
│   MEMORY          CRITIC           ROUTER                │
│   what is true    is this good?    which model, and      │
│   about you                        may this leave?       │
│        └──────────────┬──────────────┘                   │
│                 DECISION LOG                             │
│         what it saw · chose · what happened              │
└─────────────────────────┬────────────────────────────────┘
                          │  OpenAI-compatible HTTP
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   OpenRouter          Ollama          any endpoint
   (free tier)      (local / LAN)
```

The client launches the daemon itself over stdio. There is no port to open, no
service to remember to start, and no UI of ours to learn.

---

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e .
```

Point your client at the entry point. For Claude Code, `.mcp.json` in the project
root:

```json
{
  "mcpServers": {
    "one-for-all": {
      "command": "C:/path/to/.venv/Scripts/one-for-all.exe",
      "args": []
    }
  }
}
```

Use the venv's entry point rather than bare `python`, so it resolves regardless
of what is on `PATH` when the client launches. Restart the client and the tools
appear.

**The memory tools need no key, no GPU and no network.** That is a design
constraint, not a coincidence — the first thing you try should not be blocked on
signing up for anything.

For the model layer (curation and the critic), add a free key:

```bash
setx OPENROUTER_API_KEY sk-or-...      # Windows
# export OPENROUTER_API_KEY=sk-or-...  # macOS / Linux
```

Check what it found:

```bash
one-for-all-cli status
```

---

## The tools your AI gets

| Tool | What it is for |
|---|---|
| `remember` | Store a preference, correction, decision or constraint |
| `recall` | Retrieve what is known, relevant to the task |
| `recall_was_incomplete` | Report that recall missed something — **the highest-value signal here** |
| `list_recent` | Audit what is stored |
| `memory_history` | See how a belief changed over time |
| `forget` | Delete something that should never have been stored |
| `critique` | Second opinion on code, with your preferences in hand |
| `model_status` | What is reachable, and what each role would pick right now |

The tool descriptions are load-bearing, not documentation. They are the only
thing the model reads when deciding whether to call a tool, so they state *when*
to call, not just what the tool does. Editing them changes behaviour.

---

## What makes it different

**Memories are never overwritten.** A memory that stops being true is
*superseded* by a newer one, and the old row survives with a pointer forward:

```
#2 "uses Postgres"  ──►  #7 "uses SQLite"  ──►  #23 "uses DuckDB"
     (retired)              (retired)              (live)
```

Search returns only live memories, but `memory_history` walks the chain. So "what
did I believe last month, and when did that change?" stays answerable, a wrong
correction is recoverable, and a fact corrected three times is visibly a fact
that was never captured well.

Memories are typed as **semantic** (something true), **procedural** (how you want
things done) or **episodic** (something that happened) — retrieved and aged
differently, so the distinction is structural rather than a label.

**Failures label themselves.** When the assistant calls `recall()` and then still
has to guess or go digging, it calls `recall_was_incomplete()`. That single call
turns an invisible retrieval failure into a labelled training example, generated
by ordinary use, for free. It is the mechanism the rest of the system compounds
on.

**The critic is quiet.** It runs automatically and speaks only above a confidence
bar, because a critic that cries wolf is a critic you turn off. An empty response
means nothing cleared the bar — that is success, not silence.

---

## The model layer

Three jobs with opposing requirements, so this is a portfolio with a router, not
a choice of model:

| Role | Wants | Runs | Fine-tuned? |
|---|---|---|---|
| **Curate** | small, fast, cheap | every request | ✅ the only target |
| **Critique** | judgment, precision | in the background | ❌ never |
| **Generate** | peak capability | usually your client's own model | — |

Model ids are fetched at runtime and never hardcoded, because free-tier lineups
change weekly. Rate limits are treated as a normal condition, not an error.
Content the privacy gate flags is forced to a local backend or refused outright.

See [MODEL.md](MODEL.md) for role specs, selection criteria, quantization,
serving and the fine-tuning procedure.

---

## The scoreboard gates everything

```bash
one-for-all-cli eval evalsets/starter.jsonl --compare-all
```

Reports miss rate, noise rate, format compliance, p95 latency and compression
ratio — against a free *keep-everything* baseline that always runs. A curator
that cannot beat keep-all is spending latency to achieve nothing, and that is the
most useful thing an eval can tell you.

Any model change — new model, new quantization, new prompt, new adapter — is a
hypothesis tested against this before adoption. No exceptions, including for
changes that are obviously better.

`evalsets/starter.jsonl` is a synthetic smoke test. Replace it with real cases
from your own logs:

```bash
one-for-all-cli seed-eval evalsets/mine.jsonl
```

That drafts cases from what the curator actually chose — which is *not* ground
truth. Read every line and correct it before trusting a single number.

---

## Fine-tuning

Only the curator, only after the scoreboard exists, and only once prompting is
exhausted. Selection is narrow, consistent and self-labelling; judgment is not,
which is why the critic is never tuned.

```bash
one-for-all-cli export train.jsonl
```

The training data comes from the decision log, labelled by ordinary use.
[`notebooks/curator_qlora_colab.ipynb`](notebooks/curator_qlora_colab.ipynb) runs
a QLoRA on a free Colab T4, evaluates before and after, and tells you to throw
the adapter away when the difference is noise.

Quality dominates volume here: a few hundred clean examples beat thousands of
noisy ones, so the highest-leverage hour is spent filtering, not collecting.

---

## CLI

| Command | Does |
|---|---|
| `status` | Reachable backends, per-role picks, store stats |
| `eval <cases>` | Score the curator; `--compare-all` ranks every eligible model |
| `export <out>` | Decision log → training JSONL |
| `seed-eval <out>` | Draft an eval set from the log |
| `config --write` | Materialise the config file |

Everything here is deliberately outside the request path — running an eval must
never happen while someone is waiting for an answer.

---

## Layout

```
src/one_for_all/
  store.py      memory + decision log (SQLite, FTS5)
  server.py     MCP surface — the 8 tools above
  router.py     backend and model selection, privacy gate
  backends.py   OpenAI-compatible client
  curator.py    context selection
  critic.py     the second pass
  evals.py      the scoreboard
  dataset.py    decision log → training / eval data
  config.py     defaults and config file
  cli.py        inspection, scoring, tuning
```

---

## Status

v0 (memory) and v1 (model layer) are in.

```bash
pytest -q      # 66 passed
```

The whole suite runs offline. A suite that needs a free-tier key is a suite that
fails when the free tier is busy.

**Known limitation:** candidate retrieval is keyword-based (BM25 via FTS5), so it
misses on vocabulary mismatch — *"what database do we use"* does not match
*"uses SQLite for the main store"*. The curator narrows candidates well but
cannot recover what retrieval never surfaced. Hybrid vector + BM25 retrieval is
the fix and is the next substantial piece of work.

---

## Docs

| | |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Thesis, components, invariants, where effort goes |
| [BUILD_PLAN.md](BUILD_PLAN.md) | Prerequisites, MCP wiring, phase sequence |
| [MODEL.md](MODEL.md) | Model roles, routing, evaluation, fine-tuning |

Apache-2.0.
