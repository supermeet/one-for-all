# one-for-all — Architecture

**Status:** design draft, pre-v0
**Last updated:** 2026-08-12

---

## 1. Thesis

Models are commodities. Access to them is nearly free and getting freer, and any
model can be swapped for another in a config change. What is *not* commodity is
the accumulated context of one person's work — how they think, what they have
decided, what they keep getting wrong, what their projects assume.

Today that context is either absent (every session starts cold) or owned by a
vendor (locked in their product, unexportable, and readable by them).

**one-for-all is a local daemon that owns that context and lends it to whatever
AI you are already using.** It does not replace your tools. It makes them yours.

Three capabilities, in dependency order:

1. **Memory** — a private, local, durable record of you.
2. **Judgment** — a critic pass that asks whether the work was done *well*, and
   turns the answer into teaching.
3. **Self-improvement** — the daemon measures its own decisions and tunes itself.

### Non-goals

Naming these matters as much as the goals, because each is a plausible-sounding
direction that would sink the project.

- **Not a chat app.** No UI of our own. We are a server; the client is whatever
  the user already has open.
- **Not a hosted service.** No server we operate, no account, no user data
  leaving the machine. Privacy is structural, not a policy promise.
- **Not a model provider.** We do not train foundation models or claim any model
  is ours.
- **Not an "all the free models" aggregator.** Free tiers are an implementation
  detail and they churn weekly. They are never the value proposition.

---

## 2. System shape

```
┌──────────────────────────────────────────────────────────────┐
│  CLIENTS  (not ours — we plug into them)                     │
│  Claude Code · Cursor · VS Code/Copilot · Gemini CLI · …      │
└───────────────────────────┬──────────────────────────────────┘
                            │  MCP  (stdio)
┌───────────────────────────▼──────────────────────────────────┐
│  DAEMON  (Python, local, single process)                     │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐              │
│  │  MEMORY    │  │  CRITIC    │  │  ROUTER    │              │
│  │            │  │            │  │            │              │
│  │ store +    │  │ 2nd pass:  │  │ picks a    │              │
│  │ retrieval  │  │ bugs,      │  │ backend,   │              │
│  │            │  │ simpler,   │  │ enforces   │              │
│  │            │  │ teach      │  │ privacy    │              │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘              │
│        └───────────┬───┴───────────────┘                     │
│              ┌─────▼──────┐                                  │
│              │ DECISION   │  every choice + what happened     │
│              │    LOG     │  next. feeds §6.                  │
│              └────────────┘                                  │
└───────────────────────────┬──────────────────────────────────┘
                            │  Backend interface (OpenAI-compatible HTTP)
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   OpenRouter          Ollama              (any OpenAI-
   free tier        localhost / LAN         compatible URL)
```

Everything inside the daemon box is ours and is the moat. Everything outside is
replaceable by design.

---

## 3. Components

### 3.1 Memory

Owns the durable record. The only component with no model dependency — it needs
retrieval, not reasoning, which is why it can ship first and alone.

| Concern | Decision |
|---|---|
| Storage | One SQLite file under the OS user-data dir. Single file = trivially backed up, inspected, deleted. |
| Retrieval | Behind a single `search(query) -> [Memory]` function. FTS5 first (zero ML deps); embeddings later without touching callers. |
| Encryption | Deferred past v0 — SQLite is local and OS file permissions apply. Revisit before anyone else installs this. |
| What gets stored | Preferences, corrections, project constraints, working style. **Not** things the repo already records. |

The hard problem here is not storage, it is **write policy** — deciding what is
worth remembering. Over-eager writing produces a landfill that poisons retrieval.
See §7.

### 3.2 Critic

The differentiator. Not a prompt saying "be careful" — a genuine second pass with
a different job: read the output cold and ask *is there a bug, does a library
call replace this, does it match how this user writes, what is the one thing
worth teaching here*.

- Runs **automatically but quietly**: fires on outputs, speaks only above a
  confidence bar. Bar is config, tunable per user.
- Criticism is cheap — it is a small, bounded task, so it runs on the free tier
  or a local model, not the expensive path.
- **Every critic finding is candidate memory.** This is how teaching becomes
  cumulative rather than per-session advice.

### 3.3 Router

Chooses a backend per request and enforces the privacy boundary.

- **Backend interface**: everything is an OpenAI-compatible HTTP endpoint —
  OpenRouter, Ollama, LM Studio, vLLM, rented GPU. One client, config-selected.
- **Never hardcode a model id.** Fetch the live model list at startup, filter,
  pick by capability. Free-tier lineups change weekly.
- **429 is normal, not exceptional.** Free tiers cap around 20 req/min; queue,
  back off, spill to a local backend where one exists.
- **Privacy gate**: content classified sensitive is forced local or refused.
  Memory contents never ride on a free endpoint (free tiers carry the loosest
  data terms — that is part of why they are free).

### 3.4 Decision log

Not a feature — instrumentation, and the prerequisite for §6.

Every decision is written as a triple: **what it saw, what it decided, what
happened next.**

| Decision | Outcome signal |
|---|---|
| Critic flagged an issue | Did the user act on it, or ignore it? *(ignored = labelled false positive)* |
| Curator dropped context | Did the model then go looking for it? *(labelled miss)* |
| Memory was surfaced | Did it influence the answer? |

These labels are generated by ordinary use, for free. That is the whole point:
the training set is a byproduct of the product working.

---

## 4. Invariants

Rules that are cheap now and expensive to retrofit. Violating one of these is a
bug even if the tests pass.

1. **The stable prefix is byte-stable.** System prompt, tool definitions, and
   durable preferences must never be rewritten per-request. Prompt caching is a
   *prefix match* — one changed byte invalidates everything after it, and cached
   reads cost ~10% of fresh ones. Curation and compression apply only *after*
   the cached prefix, never across it.
2. **Model ids are data, never code.** Fetched at runtime, never literals.
3. **Every decision is logged**, even while nothing reads the log.
4. **Memory never leaves the machine** unless the privacy gate explicitly allows
   that specific content, for that specific request.
5. **One retrieval path.** All recall goes through `search()`. No component gets
   its own private query.
6. **The daemon is useful with zero configuration.** Default path must work on a
   laptop with no GPU and no paid key.

---

## 5. Data model

```
memories(id, kind, text, source, created_at, used_count, last_used)
    kind ∈ preference | fact | project | correction

decisions(id, ts, kind, context, decision, outcome)
    kind ∈ recall | critique | curate
    outcome nullable — written later, when known
```

`used_count` / `last_used` are not bookkeeping; they are the decay signal that
eventually distinguishes a live memory from landfill.

---

## 6. Self-improvement

The loops in §3 currently *end in a log*. Closing them is the endgame.

Once the decision log has history, improvement is an offline job: propose a
variant (a critic prompt, a confidence bar, a retrieval weighting), replay it
against logged history, keep it if it scores better.

**This is gated on measurement, absolutely.** Self-improvement without a
scoreboard is drift with good branding — the system changes, everyone assumes it
improved, nobody can tell. The scoreboard is the hard part and it is not
optional.

Order is therefore: **log (v0) → measure (v1) → tune (v2+)**. Any attempt to
start at "tune" produces a random number generator.

---

## 7. Where the effort actually goes

The honest ranking, because most of this project is easy and a small part is
hard. Effort spent in the wrong place here is how it dies.

### High effort — this is the product

1. **Memory write policy.** *What deserves to be remembered?* Get this wrong and
   the store fills with noise, retrieval degrades, and the daemon becomes worse
   than nothing. This is a judgment problem, not an engineering one, and it will
   need many iterations against real use. **Most of your thinking goes here.**
2. **The scoreboard.** How do we know a change made it better? Without this,
   every later decision is vibes, and §6 is impossible. Unglamorous, load-bearing.
3. **The critic's prompt and confidence bar.** The difference between a valued
   second opinion and an annoying linter is entirely calibration.

### Medium effort

4. Retrieval quality — FTS5 → embeddings → reranking, once there is enough
   stored to tell the difference.
5. Privacy classification — what counts as sensitive, and the cost of being
   wrong in each direction.

### Low effort — resist over-investing

6. Backend plumbing. It is an HTTP client and a config file. It *feels*
   productive because it is visible and easy. It is not the product.
7. Voice. Genuinely fun, genuinely last. Voice on top of something that knows
   you is Jarvis; voice on top of a blank model is a worse keyboard.
8. Any UI of our own.

---

## 8. Roadmap

Each phase is independently useful; none is a prerequisite that pays off only later.

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **v0** | Memory store + MCP server + decision log | Used daily from a real client for two weeks, and its absence is missed |
| **v1** | Router (backend abstraction, privacy gate) + critic | Critic surfaces something genuinely useful more often than it annoys |
| **v2** | Scoreboard + context curation | A prompt/threshold change can be shown better, not just asserted |
| **v3** | Self-improvement job | Measured improvement on the v2 scoreboard without human tuning |
| **v4** | Voice shell | — |

**v0 exit criterion is deliberately subjective.** The honest test of a memory
layer is whether removing it hurts. No amount of green tests substitutes.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Memory becomes a landfill; retrieval degrades | Write policy is the #1 effort item; decay via `used_count`; aggressive `forget` |
| Free tiers vanish or tighten | Backend is pluggable by design; never a stated feature |
| Critic is annoying, gets turned off | Quiet by default, high bar, tunable; measure ignore-rate as a first-class metric |
| Self-improvement becomes unfalsifiable | Hard-gated behind the scoreboard |
| Scope creep into a chat app / hosted service | Listed as explicit non-goals in §1 |
| Python 3.14 lacks ML wheels | v0 has no ML dependencies at all; embeddings deferred until they are needed |

---

## 10. Environment

Development machine: Windows 11, Python 3.14, 7.8 GB RAM, Intel Iris Xe
(integrated). **Local model inference is not viable here** — a 4-bit 8B model
needs roughly the whole of that RAM. Implications:

- Default backend is OpenRouter's free tier: no hardware requirement.
- Ollama support still ships early — it is a URL and a config key — and points
  at a college PC or a rented box when one is available.
- Nothing in the design may assume a local model exists.
