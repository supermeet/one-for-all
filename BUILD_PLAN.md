# one-for-all — Build Plan

Companion to [ARCHITECTURE.md](ARCHITECTURE.md). That doc says *what* and *why*.
This one says *how it starts, how it connects, and what each step requires*.

---

## 0. Prerequisites

Everything needed before writing a line. Total cost: nothing.

| Requirement | Status on this machine | Notes |
|---|---|---|
| Python ≥ 3.11 | ✅ 3.14.0 | 3.14 is new — no ML wheels yet, which is why v0 avoids ML entirely |
| git | ✅ 2.53 | |
| GitHub repo | ✅ `supermeet/one-for-all` | main, README + Apache-2.0 |
| An MCP-capable client | ✅ Claude Code | Cursor / Copilot / Gemini CLI also work |
| OpenRouter account | ⬜ **needed at v1, not v0** | Free, email signup, no card |
| Ollama on a second machine | ⬜ optional, v1+ | College PC or rented box; never required |

**Nothing is needed for v0 except Python and git.** v0 has no model calls at
all — it is a memory store and a protocol server. This is deliberate: the first
milestone should not be blocked on an account, a key, or hardware.

---

## 1. Repo bootstrap

The remote already has `README.md` and `LICENSE`. Do not overwrite them.

```bash
git remote add origin https://github.com/supermeet/one-for-all.git
git fetch origin
git branch -M main
git reset --soft origin/main     # adopt the remote's history, keep local files
```

Target layout:

```
one-for-all/
├── README.md              # exists — becomes the pitch + quickstart
├── LICENSE                # exists — Apache-2.0
├── ARCHITECTURE.md        # what and why
├── BUILD_PLAN.md          # this file
├── pyproject.toml
├── src/one_for_all/
│   ├── __init__.py
│   ├── store.py           # memory + decision log     ← v0
│   ├── server.py          # MCP surface               ← v0
│   ├── backends.py        # OpenAI-compatible client  ← v1
│   ├── router.py          # backend choice + privacy  ← v1
│   └── critic.py          # the second pass           ← v1
└── tests/
```

Branch per phase (`v0-memory`, `v1-critic`), PR into main. Not ceremony — it
makes "what changed when the critic got annoying" answerable later.

---

## 2. How it connects

This is the part that turns a Python script into something you actually use.

### The mechanism

MCP servers are **launched by the client, not run by you.** The client spawns the
process, talks JSON-RPC over stdin/stdout, and kills it on exit. There is no
port, no daemon to remember to start, no localhost URL.

```
Claude Code starts
   └─ reads its MCP config
       └─ spawns:  one-for-all           (our entry point)
           └─ handshake: server advertises its tools
               └─ tools appear to the model as remember / recall / forget
```

### The wiring

One config entry. Same shape across clients — only the file location differs.

```json
{
  "mcpServers": {
    "one-for-all": {
      "command": "C:/Users/meetv/Documents/ClaudeCode/One for all/.venv/Scripts/one-for-all.exe",
      "args": []
    }
  }
}
```

| Client | Where it goes |
|---|---|
| Claude Code | `.mcp.json` in the project root (per-project) or user settings (global) |
| Cursor | Settings → MCP → add server |
| VS Code / Copilot | `.vscode/mcp.json` |
| Gemini CLI | its settings file |

Point it at the venv's entry-point executable, not at bare `python` — that way
it works regardless of what is on `PATH` when the client launches.

### Verifying the connection

Three checks, in order. Each isolates a different failure.

1. **Server runs standalone.** `one-for-all` starts and waits on stdin without
   crashing. Failure here = a Python problem, not an MCP problem.
2. **Client lists the tools.** Restart the client; `remember` / `recall` /
   `forget` appear in its tool list. Failure here = config path or permissions.
3. **Round trip.** Tell it to remember something, start a *fresh* session, ask it
   to recall. Failure here = our logic, and now it is debuggable.

Do not proceed past step 3 until it passes. Everything later assumes this works.

---

## 3. How the work actually proceeds

### v0 — Memory (target: ~2 weeks)

**Build order, smallest useful thing first:**

| # | Step | Done when |
|---|---|---|
| 1 | `store.py` — SQLite schema, `remember` / `search` / `forget` | Unit tests pass, including punctuation in queries |
| 2 | Decision-log tables and writes | Every recall writes a row, even though nothing reads them |
| 3 | `server.py` — MCP tools wrapping the store | The three checks in §2 pass |
| 4 | Tool descriptions | See below — this is not a formality |
| 5 | **Use it, daily** | Two weeks of real use |

**Step 4 deserves its own note.** The tool `description` is the only thing that
decides whether the model ever calls it. It has to say *when* to call, not just
what it does — recent models are conservative about reaching for tools, and
prescriptive trigger conditions measurably raise the call rate. `"Save a memory"`
will be ignored. `"Call this when the user states a preference, corrects you, or
explains a constraint that will still be true tomorrow"` will not.

**Step 5 is the actual work.** Building it is a few days; the two weeks of use is
where you find out whether the write policy is right. Expect to rewrite the tool
descriptions several times. That rewriting *is* the project — not a delay before it.

**Exit criterion:** turning it off is annoying.

### v1 — Judgment (target: ~3 weeks)

Order matters here, because each step de-risks the next.

1. **`backends.py`** — one `chat()` function over an OpenAI-compatible endpoint.
   Fetch the model list at startup; never hardcode an id. ~100 lines. Do not
   gold-plate this; it is plumbing (ARCHITECTURE §7).
2. **`router.py`** — pick a backend, enforce the privacy gate, handle 429 as a
   normal condition with backoff.
3. **`critic.py`** — the second pass. Start with one narrow job: *given this
   code, is there a bug or a stdlib/library call that replaces it?* Resist adding
   dimensions until that one is calibrated.
4. **Wire critic findings into memory** — this closes the first loop and is where
   "teaches you" becomes real rather than aspirational.

**Exit criterion:** over a week of use, the critic is right more often than it is
noise, and you have not turned it off.

### v2 — Measurement (target: ~2 weeks)

The unglamorous phase that everything after depends on.

- Replay harness: take logged decisions, re-run them under a changed prompt or
  threshold, diff the outcomes.
- A handful of metrics that mean something: critic ignore-rate, recall
  precision (surfaced memories that mattered), curation miss-rate.
- ~50 hand-labelled cases from your own logs as a fixed benchmark.

**Exit criterion:** you can change a prompt and *show* whether it helped.

### v3 — Self-improvement

Only now. Nightly job: propose variants, replay against v2's harness, keep
winners. Gated on v2 existing — see ARCHITECTURE §6.

---

## 4. Fine-tuning — what it actually requires

Taking this seriously, since it keeps coming up.

### Do the cheaper things first

Fine-tuning is the *fourth* option, not the first. In order of cost:

1. **Better prompt** — free, minutes, usually most of the gain.
2. **Few-shot examples selected from your own log** — free, and it adapts as your
   log grows. Often indistinguishable from tuning on narrow tasks.
3. **A better base model** — free, a config change.
4. **LoRA fine-tune** — everything below.

Reach for (4) only when (1)–(3) are exhausted **and the v2 scoreboard shows a
consistent gap**. Without the scoreboard you cannot tell whether tuning helped,
which makes the whole exercise unfalsifiable.

### What a LoRA run needs

| Requirement | Reality |
|---|---|
| **Data** | ~500–2,000 input/output pairs for a narrow task like curation. Fewer works if the task is tight and consistent. |
| **Data format** | JSONL, one `{"messages": [...]}` per line. Same shape as an API request. |
| **Where the data comes from** | The decision log. This is exactly why §3 logs from day one — after some months of use, the training set already exists. |
| **Hardware** | ~16–24 GB VRAM for a 7–8B LoRA. **Not this laptop** (7.8 GB, integrated Iris Xe). |
| **Realistic option** | Rented GPU by the hour — an A100 or 4090 class box, a few dollars for a full run. Or the college PC if it has a discrete card with enough VRAM. |
| **Time** | Hours, not days, for a LoRA on a few thousand examples. |
| **Output** | An adapter file (tens of MB), merged into the base model or loaded alongside. Ollama can serve the result. |

### What to tune, and what not to

**Good candidate: the curator.** Selection and extraction is a narrow,
high-volume, consistent task with automatically labelled failures — the model
asked for something the curator dropped. That is a clean training signal and it
accumulates without any effort from you.

**Bad candidate: the critic.** Judgment is exactly what small models are worst
at, and "was this critique good?" has no cheap automatic label. Keep the critic
on a capable model.

### The realistic timeline

Fine-tuning is a **v3+ activity, at the earliest a few months into daily use**,
and there is a real chance the base model plus a good prompt is simply enough.
Treat it as a possible optimization the architecture leaves room for — not a
milestone to plan around.

---

## 5. What could stall this

| Stall | Signal | Response |
|---|---|---|
| Building instead of using | Week 3 of v0 with lots of code and no daily use | Stop. Ship what exists and use it. |
| Backend plumbing rabbit hole | Elaborate provider abstraction before any memory exists | It is an HTTP client. Move on. |
| Memory landfill | Recall returns noise | Tighten the write policy; delete aggressively; this is the #1 effort item |
| Critic fatigue | You turned it off | Raise the bar. Ignore-rate is a metric, not an opinion. |
| Chasing self-improvement early | Tuning before a scoreboard | Build v2 first. Non-negotiable. |

---

## 6. Immediate next actions

1. Wire the local directory to `origin/main` without clobbering README/LICENSE.
2. Commit ARCHITECTURE.md + BUILD_PLAN.md — design before code, on the record.
3. Build v0 steps 1–3.
4. Wire into Claude Code; pass the three connection checks.
5. Use it for two weeks. Change nothing structural during that window; take notes.
