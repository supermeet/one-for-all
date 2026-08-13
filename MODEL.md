# one-for-all — Model Layer

**Status:** specification, pre-implementation (v1)
**Last updated:** 2026-08-13
**Companions:** [ARCHITECTURE.md](ARCHITECTURE.md) (what & why) · [BUILD_PLAN.md](BUILD_PLAN.md) (sequence & wiring)

---

## 0. The premise, stated plainly

There is no single "best and most efficient" model for this project, and chasing
one is the most expensive mistake available here.

This system asks models to do three jobs, and the jobs have **opposing**
requirements:

| Job | Needs | Runs | Tolerates |
|---|---|---|---|
| **Curate** | Speed, low cost, high volume | Constantly, on every request | Being small and dumb |
| **Critique** | Judgment, correctness | Occasionally, in background | Being slow |
| **Generate** | Peak capability | Interactively | Being expensive |

A model optimised for one is wrong for the others. A 3B model that curates
beautifully cannot judge whether code is well written. A frontier model that
judges beautifully is absurd to run on every context assembly.

**So the model layer is a portfolio with a router, not a choice.** This document
specifies the portfolio: what each role requires, how to pick for it, how to
serve it, how to tune it, and how to know it is working.

> **On specific model names.** Every named model below is a *candidate as of
> August 2026*, not a commitment. This field churns monthly and the router
> fetches the live model list at runtime precisely so no name is load-bearing.
> Re-verify before adopting; treat §3's *criteria* as the durable part and §4's
> *names* as disposable.

---

## 1. Role specifications

### 1.1 Curator — the volume tier

Assembles context: selects which memories, file chunks, and history go into a
request. Runs on every single call, so its cost is multiplied by everything.

| Property | Requirement |
|---|---|
| Task type | Selection, extraction, ranking, classification |
| **Not** | Reasoning, judgment, generation |
| Size | 1–4B, quantized |
| Latency budget | < 300 ms — above that it costs more than it saves |
| Context | 8K minimum; 32K preferred |
| Quality bar | Recall over precision — dropping something needed is far worse than keeping something extra |
| Privacy | **Must be local-capable.** It sees raw memory contents. |

This is the one role worth eventually fine-tuning (§7). Selection is narrow,
consistent, high-volume, and — crucially — **self-labelling**: when the
downstream model asks for something the curator dropped, that is an automatically
generated training example.

### 1.2 Critic — the judgment tier

Reads output cold and asks: is there a bug, does a library call replace this, does
it match how this user writes, what is worth teaching here.

| Property | Requirement |
|---|---|
| Task type | Analysis, judgment, correctness |
| Size | 7B+ minimum; frontier meaningfully better |
| Latency budget | Seconds. It runs in the background and speaks only when confident. |
| Context | 32K+ — it needs the code *and* the relevant memories |
| Quality bar | **Precision over recall.** A false positive trains the user to ignore it; that is fatal. |
| Privacy | Sees code. Gate-controlled. |

**Do not fine-tune this role.** Judgment is what small models are worst at, and
"was this critique good?" has no cheap automatic label. Buy capability here.

### 1.3 Generator — the capability tier

The model the user is actually talking to. **Mostly not ours** — in the primary
flow this is Claude Code or Cursor, and one-for-all is the MCP server it calls.
We only need a generator for headless work (a nightly job, an autonomous run).

| Property | Requirement |
|---|---|
| Size | Best available |
| Context | 128K+ |
| Privacy | Gate-controlled; may be remote |
| Note | Usually supplied by the client, not by us |

---

## 2. What "efficient" actually means

Tokens-per-second is the metric everyone quotes and it is close to the least
useful one. The metrics that matter for this system:

| Metric | Why it matters | Target (curator) |
|---|---|---|
| **Quality per GB of RAM** | RAM is the binding constraint on consumer hardware | Fit in ≤ 6 GB |
| **Time to first token** | Curation sits in the critical path of every request | < 300 ms |
| **Cost per useful decision** | Not per token — a cheap model that curates wrongly costs a re-query | — |
| **Tokens saved ÷ tokens spent** | Curation must earn its keep; below ~3× it is not worth the complexity | > 3× |
| Tokens/sec | Only matters for long generation, which the curator never does | Ignore |

**The efficiency trap to avoid:** compressing context to save tokens while
breaking prompt caching. Cached input costs roughly 10% of fresh input, and
caching is a prefix match — rewriting the prefix each turn turns an 80%
reduction into a 2× *increase*. See ARCHITECTURE §4, invariant 1. Curation
applies only after the cached prefix.

---

## 3. Selection criteria

Apply in order. Stop at the first that disqualifies.

1. **Licence.** Must permit local commercial use. Apache-2.0 or MIT preferred;
   read custom licences (some "open" weights restrict deployment).
2. **Fits the hardware target** at Q4_K_M with the required context length.
   Compute it, do not eyeball it: params × bits/8 × 1.2 overhead + KV cache.
3. **Serves over an OpenAI-compatible endpoint.** Non-negotiable — the whole
   backend abstraction depends on it (ARCHITECTURE §3.3).
4. **Instruction-following ability**, measured on *our* task, not a leaderboard.
   For the curator specifically: does it reliably return only the selected ids,
   in the requested format, with no commentary?
5. **Fine-tunable** if it is the curator — weights available, LoRA-supported
   architecture, works with Unsloth/Axolotl.
6. Only then: benchmark scores.

**Benchmarks are a filter, never a decision.** IFEval and HumanEval say nothing
about whether a model can pick the right five memories out of forty. Build the
50-case eval set (§8) and rank candidates on that.

---

## 4. Candidate models (August 2026 — verify before adopting)

### Curator tier (1–4B)

| Model | Notes |
|---|---|
| **Gemma 3 4B** | ~4.2 GB RAM — best-in-class memory efficiency; strong default for constrained boxes |
| **Phi-4-mini (3.8B)** | Runs on CPU with no GPU; the fallback when there is no card at all |
| Qwen3 ~4B | Strong instruction following in the small class |

**Starting recommendation: Gemma 3 4B at Q4_K_M**, on RAM efficiency plus
licence. Re-rank against the §8 eval before committing.

### Critic tier (7B+, or remote)

| Model | Notes |
|---|---|
| **Qwen3 8B** | Leads the 7–8B class on code generation (HumanEval and similar) — the closest match to the critic's actual job |
| **Llama 3.3 8B** | Best instruction following in class (~92% IFEval); most predictable output shape |
| Phi-4 | Strongest math/reasoning per GB |
| Mistral Small 3 7B | Fastest — ~50 tok/s on mid-range 16 GB at Q4_K_M |
| DeepSeek R1 | Best chain-of-thought where deliberation is worth the latency |

**Starting recommendation: a frontier model via OpenRouter's free tier**, with
Qwen3 8B as the local/offline fallback. The critic's failure mode is false
positives, and capability buys precision. Run it locally only when the privacy
gate demands it.

### Generator tier

Supplied by the client (Claude Code, Cursor). For headless work: whatever the
router's live free-tier list offers with the largest context.

---

## 5. Hardware reality

| Machine | Spec | Verdict |
|---|---|---|
| **This laptop** | 7.8 GB RAM, Intel Iris Xe (integrated) | **No local inference.** A 4-bit 8B needs nearly all of it; integrated graphics offer no useful acceleration. Curator on CPU at 3B *might* be viable but will be slow. |
| **College PC** | TBD — needs a discrete GPU with ≥ 8 GB VRAM | Run Ollama here, expose on the LAN, point the router at it |
| **Rented GPU** | 12–24 GB class, hourly | The fine-tuning box (§7); a few dollars per run |

**Rules of thumb:** 8 GB VRAM covers 7–8B models; 24 GB is the practical floor
for 30B; 40 GB+ for 70B without aggressive quantization.

**Design consequence, and it is absolute:** nothing in the system may assume a
local model exists. Default path is remote; local is an optimisation the router
uses when available.

---

## 6. Quantization and serving

### Quantization

| Format | Use |
|---|---|
| **Q4_K_M** | The default. Best quality-per-byte tradeoff; ~4.5 bits/param effective |
| Q5_K_M | When quality regression is measurable and RAM allows |
| Q8_0 | Reference runs when validating that quantization is what hurt |
| Q3 and below | Avoid — degradation becomes obvious on instruction following |

Always validate a quantized model against the §8 eval set. Quantization damage
shows up first as format drift — the model stops respecting the output
contract — which is exactly what the curator cannot tolerate.

### Serving

| Option | When |
|---|---|
| **Ollama** | Default. One command, OpenAI-compatible at `/v1`, trivial model management |
| llama.cpp | When you need granular control over threads, layers, KV cache |
| vLLM | Only if serving concurrent users — irrelevant for a single-user daemon |
| LM Studio | GUI for exploration, not for the daemon |

All speak OpenAI-compatible HTTP, so all are the same to our router — a URL and a
config key. Serving choice is genuinely reversible; do not agonise over it.

---

## 7. Fine-tuning the curator

Only the curator. Only after §8 exists. Only when prompting is exhausted.

### The ladder — climb it in order

| Step | Cost | Typical gain |
|---|---|---|
| 1. Better prompt | Minutes | Most of it |
| 2. Few-shot examples pulled from the decision log | Free, self-improving | Often closes the gap entirely |
| 3. Better base model | Config change | Sometimes large |
| 4. **QLoRA fine-tune** | Below | The remainder |

Reaching step 4 without a scoreboard means you cannot tell whether it worked.
That is not a fine-tune, it is a ritual.

### Requirements

| Item | Spec |
|---|---|
| **Method** | QLoRA (4-bit base + LoRA adapters) |
| **VRAM** | 8–12 GB for a 7–8B model. Under 10 GB achievable at seq ≤ 512, batch 1, gradient checkpointing. An RTX 4070 Ti (12 GB) suffices. |
| **Data volume** | 500–2,000 examples. **Quality dominates:** 500 clean beat 5,000 noisy; 1,000 hand-curated beat 100,000 noisy. |
| **Data source** | The decision log — this is why it ships in v0 unused |
| **Format** | JSONL, one `{"messages": [...]}` per line |
| **LoRA rank** | r=16 for format/style adherence; r=32 for general SFT. r=64 is for complex multi-turn/coding — overkill here. |
| **Toolchain** | Unsloth (fast on consumer GPUs) → Axolotl (YAML pipelines) → TRL (advanced objectives) |
| **Duration** | Hours, not days |
| **Output** | A LoRA adapter (tens of MB), merged or served alongside the base. Ollama can serve the result. |

### Training data shape

Each example is one curation decision that the log already labelled:

```json
{"messages": [
  {"role": "system", "content": "Select the memories relevant to the task. Return ids only."},
  {"role": "user", "content": "TASK: <task>\n\nCANDIDATES:\n[1] ...\n[2] ...\n[17] ..."},
  {"role": "assistant", "content": "3,7,12"}
]}
```

**Labels come free.** The log records what the curator chose and whether the
downstream model then went looking for something dropped. A miss is a corrected
example; a clean run is a positive one. Months of ordinary use produce the
dataset without a labelling effort.

### Curation before training

Do not train on the raw log. Given that 500 clean examples beat 5,000 noisy ones,
the highest-leverage hour is spent filtering:

- Drop examples where the task itself was ambiguous
- Drop duplicates and near-duplicates (they teach frequency, not judgment)
- **Keep every recorded miss** — those are the hard cases and carry the signal
- Balance: if 90% of examples select 3–5 items, the model learns to always
  select 3–5 items

---

## 8. Evaluation — the prerequisite for all of the above

Without this, every model choice, quantization decision, and fine-tune is
unfalsifiable. Build it before any of them.

### The eval set

~50 hand-labelled cases drawn from your own logs. For each: the task, the
candidate memories, and the ids that *should* have been selected. Fifty is
enough to detect real differences and small enough to build in an afternoon.

### Metrics

| Metric | Definition | Why |
|---|---|---|
| **Miss rate** | Needed memory not selected | The failure that matters — asymmetric, weight it heavily |
| **Noise rate** | Irrelevant memory selected | Costs tokens, dilutes attention |
| **Format compliance** | Output parsed on first attempt | Small models fail here first; a retry erases the savings |
| **p95 latency** | Tail, not mean | The curator is in the critical path |
| **Compression ratio** | Tokens in ÷ tokens out | Must clear ~3× to justify existing |

### The rule

**Any model change is a hypothesis, tested against this set before adoption** —
new model, new quantization, new prompt, new adapter. No exceptions, including
for changes that are obviously better.

---

## 9. Router defaults

Encoding the above as the behaviour the router should implement:

```
request arrives
  ├─ privacy gate: does this touch flagged content?
  │    yes → local backend only; refuse if none available
  │    no  → continue
  ├─ role?
  │    curate    → smallest capable model; local if present, else free tier
  │    critique  → most capable available; free tier by default
  │    generate  → normally the client's own model; else largest free-tier context
  └─ on 429 → back off, then spill to local; never fail loudly on a background task
```

Plus the standing rules from ARCHITECTURE §4: model ids fetched at runtime and
never hardcoded; 429 treated as normal; the cached prefix never rewritten.

---

## 10. Anti-patterns

| Don't | Because |
|---|---|
| Search for one model that does everything | The three roles have opposing requirements (§0) |
| Pick from leaderboards | They do not measure our task (§3) |
| Fine-tune before a scoreboard | Unfalsifiable (§8) |
| Fine-tune the critic | Judgment is what small models are worst at, and there is no cheap label |
| Chase tokens/sec | Wrong metric for a selection workload (§2) |
| Compress across the cached prefix | Turns a saving into a 2× cost (§2) |
| Hardcode a model id | Free tiers churn weekly |
| Assume a local model exists | The primary dev machine cannot run one (§5) |
| Over-build the backend abstraction | It is an HTTP client and a config file |
| Rank a model that errored | A 401 scores like a terrible model; fix setup, then measure (§8) |
| Assume a provider's model list is chat models | It also lists image and audio generators, some with huge contexts |

---

## 11. Open questions

Unresolved, listed so they are not silently decided by accident:

1. **Embeddings on Python 3.14.** Hybrid retrieval needs vectors, and torch has
   no 3.14 wheels yet. ONNX Runtime with a small embedding model? A remote
   embedding endpoint (privacy cost)? Pin to 3.12 for the daemon? — **blocks the
   retrieval fix, decide first.**
2. ~~**Curator on CPU.**~~ **Answered — no.** Measured 2026-08-13 against
   `evalsets/starter.jsonl` on this laptop, with the Ollama that turns out to be
   installed here (`phi3:latest`, 3.8B):

   | | phi3 on CPU | budget | baseline: keep-all |
   |---|---|---|---|
   | p95 latency | **51 s** | 300 ms | 0 ms |
   | miss rate | 53.7% | — | 0% |
   | compression | 2.5× | > 3× | 1.0× |
   | penalty | 2.176 | — | **0.487** |

   170× over the latency budget and comfortably worse than not curating at all.
   The curator is remote-only until there is a GPU. Note this measures *phi3 on
   this CPU*, not the curator tier — Gemma 3 4B on a discrete card is untested
   and §4's recommendation stands.
3. **College PC spec.** Determines whether local inference is real or theoretical.
   Now the blocking question for the curator, given (2).
4. **Critic confidence bar.** Needs calibration against real use; cannot be
   chosen in advance.
5. **Privacy classification.** What counts as sensitive, and which direction of
   error is cheaper.

---

## 12. Summary

- **Three roles, three models, one router.** Not one model.
- **Curator:** small, local-capable, fast, self-labelling — the only fine-tune target.
- **Critic:** capable, precision-first, never fine-tuned.
- **Generator:** usually the client's, not ours.
- **Efficiency** is quality-per-GB and cost-per-decision, not tokens/sec.
- **The eval set gates everything.** Build it before choosing, quantizing, or tuning.
- **Names are disposable; criteria are durable.** The router fetches models at
  runtime so no name in §4 is load-bearing.
