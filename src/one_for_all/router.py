"""Router — picks a backend per request and enforces the privacy boundary.

This is the implementation of MODEL.md §9. Two things distinguish it from a
plain "call the model" helper:

*It routes by role, not by preference.* The three roles have opposing
requirements (MODEL.md §0) — the curator wants small and fast, the critic wants
capable and tolerates slowness — so there is no single right model and no
setting that would make one appear. The router encodes the tradeoff.

*It refuses.* Invariant 4 says memory contents never leave the machine unless
the gate allows that specific content for that specific request. When content
is sensitive and no local backend answers, this raises rather than quietly
falling back to a free endpoint. A privacy boundary that yields under load is
not a boundary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .backends import Backend, BackendError, BackendUnavailable, ModelInfo, RateLimited
from .config import Config, RoleConfig, load as load_config
from .store import Store


class NoBackendAvailable(RuntimeError):
    """Nothing could serve this request. On a background task, callers should
    swallow this — a critic that cannot reach a model has no opinion, which is
    not an error the user needs to hear about."""


class PrivacyRefusal(NoBackendAvailable):
    """Content was classified sensitive and no local backend was reachable."""


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    backend: str
    latency_s: float
    sensitive: bool


class Router:
    def __init__(
        self,
        config: Config | None = None,
        store: Store | None = None,
        backends: list[Backend] | None = None,
    ) -> None:
        self.config = config or load_config()
        self.store = store
        self.backends = backends if backends is not None else [
            Backend(cfg) for cfg in self.config.backends if cfg.enabled
        ]

    # -- the privacy gate -------------------------------------------------

    def is_sensitive(self, *texts: str) -> bool:
        """Crude substring classification, deliberately. MODEL.md §11.5 leaves
        the real policy open; until it is decided, the cheap error is a false
        positive — being forced local costs latency, whereas a false negative
        puts a credential on an endpoint whose data terms are loose enough to
        be free."""
        blob = " ".join(texts).lower()
        return any(pattern in blob for pattern in self.config.privacy_patterns)

    # -- selection --------------------------------------------------------

    def candidates(self, role: str, sensitive: bool = False) -> list[tuple[Backend, ModelInfo]]:
        """Every viable (backend, model) pair for this role, best first.

        Returning the whole ordered list rather than one winner is what makes
        spill-on-429 a one-line loop instead of a retry subsystem.
        """
        spec = self.config.role(role)
        pairs: list[tuple[Backend, ModelInfo]] = []

        for backend in self.backends:
            if sensitive and not backend.local:
                continue  # invariant 4, enforced before any request is formed
            try:
                models = backend.catalog()
            except BackendUnavailable:
                continue  # Ollama being absent is the normal case, not a fault
            for model in models:
                if not model.text_only:
                    continue  # image and audio generators are not chat models
                if backend.cfg.free_only and not model.free:
                    continue
                if spec.max_params_b is not None and model.params_b is not None:
                    if model.params_b > spec.max_params_b:
                        continue
                pairs.append((backend, model))

        pairs.sort(key=lambda pair: self._rank(role, spec, pair[0], pair[1]))
        return pairs

    def _rank(
        self, role: str, spec: RoleConfig, backend: Backend, model: ModelInfo
    ) -> tuple:
        """Sort key; lower is better.

        The preference hints from config come first, so a name that is known to
        work on our task beats a generic heuristic. Everything after them is
        the durable criteria from MODEL.md §3, which is what keeps routing
        working after every hint in the config has been renamed out of
        existence.
        """
        hint = _hint_rank(model.id, spec.prefer)
        local_first = 0 if (spec.prefer_local and backend.local) else 1

        if role == "curate":
            # Smallest capable. Latency budget is < 300 ms and it runs on every
            # request, so size is the criterion, not benchmark scores.
            return (local_first, hint, model.size_rank, model.id)

        if role == "critique":
            # Most capable available. Capability buys precision, and the
            # critic's fatal failure mode is the false positive. Parameter
            # count and context length are weak proxies for capability — they
            # are what a model list actually exposes.
            return (
                local_first,
                hint,
                -(model.params_b or 0.0),
                -(model.context_length or 0),
                model.id,
            )

        # generate: the client normally supplies its own model. When we need one
        # headless, take the largest context on offer.
        return (local_first, hint, -(model.context_length or 0), model.id)

    # -- inference --------------------------------------------------------

    def complete(
        self,
        role: str,
        messages: list[dict[str, str]],
        *,
        sensitive: bool | None = None,
        max_attempts: int = 3,
    ) -> Completion:
        """Run `messages` under the policy for `role`.

        Walks the candidate list: a rate limit or an unreachable backend moves
        to the next pair, which naturally spills from the free tier to a local
        model when one exists. Raises `NoBackendAvailable` only when every
        candidate has been tried.
        """
        spec = self.config.role(role)
        if sensitive is None:
            sensitive = self.is_sensitive(*(m.get("content", "") for m in messages))

        pairs = self.candidates(role, sensitive=sensitive)
        if not pairs:
            if sensitive and self.config.refuse_without_local:
                raise PrivacyRefusal(
                    "content classified sensitive and no local backend is reachable; "
                    "refusing to send it to a remote endpoint"
                )
            raise NoBackendAvailable(f"no backend can serve role {role!r}")

        errors: list[str] = []
        for backend, model in pairs[:max_attempts]:
            started = time.perf_counter()
            try:
                text = backend.chat(
                    messages,
                    model=model.id,
                    max_tokens=spec.max_tokens,
                    temperature=spec.temperature,
                    timeout_s=spec.timeout_s,
                )
            except (RateLimited, BackendUnavailable) as exc:
                errors.append(str(exc))
                continue  # spill to the next candidate
            except BackendError as exc:
                errors.append(str(exc))
                continue

            latency = time.perf_counter() - started
            self._log(role, model, backend, latency, sensitive)
            return Completion(
                text=text,
                model=model.id,
                backend=backend.name,
                latency_s=latency,
                sensitive=sensitive,
            )

        raise NoBackendAvailable(
            f"every candidate for role {role!r} failed: " + "; ".join(errors)
        )

    def _log(
        self,
        role: str,
        model: ModelInfo,
        backend: Backend,
        latency: float,
        sensitive: bool,
    ) -> None:
        """Invariant 3: every decision is logged, even while nothing reads the
        log. Routing history is what makes 'did switching models help?'
        answerable later instead of a matter of recollection."""
        if self.store is None:
            return
        self.store.log_decision(
            kind="route",
            context=f"role={role} sensitive={sensitive}",
            decision=f"{backend.name}/{model.id} in {latency:.2f}s",
        )

    def status(self) -> str:
        """Human-readable summary of what is reachable right now. The first
        thing to look at when routing behaves unexpectedly."""
        lines = []
        for backend in self.backends:
            try:
                models = backend.catalog(refresh=True)
            except BackendUnavailable as exc:
                lines.append(f"{backend.name}: unavailable ({exc})")
                continue
            free = sum(1 for m in models if m.free)
            kind = "local" if backend.local else "remote"
            lines.append(f"{backend.name}: {kind}, {len(models)} models ({free} free)")

        for role in self.config.roles:
            pairs = self.candidates(role)
            pick = f"{pairs[0][0].name}/{pairs[0][1].id}" if pairs else "none available"
            lines.append(f"role {role}: {pick}")
        return "\n".join(lines)


def _hint_rank(model_id: str, prefer: list[str]) -> int:
    """Position of the first preference hint this id matches, or a large number
    if none do. Hints are substrings so they survive a provider re-tagging
    `qwen3-8b` as `qwen3-8b-instruct:free`."""
    lowered = model_id.lower()
    for index, hint in enumerate(prefer):
        if hint.lower() in lowered:
            return index
    return len(prefer) + 1
