"""Backend client — an HTTP client and a config file, and nothing more.

ARCHITECTURE §7 ranks this as low-effort work that *feels* productive because it
is visible and easy. It is not the product. The only two ideas here that are
load-bearing:

*Model ids are fetched, never written.* `catalog()` asks the provider what it
currently serves. Free-tier lineups change weekly; a hardcoded id is a bug with
a delayed fuse.

*429 is a normal condition, not an exception.* Free tiers cap around 20 req/min.
A rate limit means wait, then try elsewhere — it does not mean the request
failed.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass

import httpx

from .config import BackendConfig

# Free tiers cap around 20 req/min, so a burst hitting 429 is expected traffic
# rather than a fault. Retry a few times with jitter before giving up on this
# backend and letting the router spill elsewhere.
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.5

# A model id like "qwen3-8b" or "gemma3:4b" carries its parameter count; one
# like "llama-3.3-8b" carries a version number too, so the digits only count
# when followed by a `b` that ends the token.
_PARAMS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b(?![a-z0-9])", re.IGNORECASE)


class BackendError(RuntimeError):
    """Base for anything that means 'this backend did not answer'."""


class BackendUnavailable(BackendError):
    """Endpoint unreachable — Ollama not running, no network, bad URL."""


class RateLimited(BackendError):
    """Exhausted retries against a 429. The caller should spill, not fail."""


@dataclass(frozen=True)
class ModelInfo:
    id: str
    context_length: int | None
    params_b: float | None
    free: bool
    backend: str
    # Whether this model does text-in, text-out. A provider's model list is not
    # a list of chat models: it also carries image and audio generators, and
    # some of them advertise enormous context windows, so any ranking by
    # context length will float them to the top. Found the hard way — the
    # `generate` role picked a music model on the first live run.
    text_only: bool = True

    @property
    def size_rank(self) -> float:
        """For 'smallest capable' ordering. Unknown size sorts large, because
        assuming a model is small when it is not puts the curator's latency
        budget in the hands of a guess."""
        return self.params_b if self.params_b is not None else 1e6


def parse_params_b(model_id: str) -> float | None:
    """Extract a parameter count from a model id, or None if it does not
    advertise one. Best-effort by nature — ids are a provider's naming habit,
    not an API."""
    matches = _PARAMS_RE.findall(model_id)
    if not matches:
        return None
    # "llama-3.3-8b-instruct" yields one match; where a name carries several,
    # the largest is the parameter count and the rest are versions.
    return max(float(m) for m in matches)


class Backend:
    """One OpenAI-compatible endpoint. OpenRouter, Ollama, LM Studio, vLLM and
    a rented box are all the same object with a different URL."""

    def __init__(self, cfg: BackendConfig, client: httpx.Client | None = None) -> None:
        self.cfg = cfg
        self.name = cfg.name
        self.local = cfg.local
        self._client = client or httpx.Client(timeout=30.0)
        self._catalog: list[ModelInfo] | None = None

    # -- discovery --------------------------------------------------------

    def catalog(self, refresh: bool = False) -> list[ModelInfo]:
        """What this backend serves right now. Cached for the process lifetime;
        the daemon is short-lived enough that staleness is not a concern."""
        if self._catalog is not None and not refresh:
            return self._catalog
        try:
            resp = self._client.get(
                f"{self.cfg.base_url}/models", headers=self._headers()
            )
            resp.raise_for_status()
            payload = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            # A backend that cannot be listed is a backend that is not there.
            # Ollama being absent is the normal case on the dev laptop.
            raise BackendUnavailable(f"{self.name}: {exc}") from exc

        self._catalog = [self._to_model(item) for item in payload.get("data", [])]
        return self._catalog

    def available(self) -> bool:
        try:
            return bool(self.catalog())
        except BackendUnavailable:
            return False

    def _to_model(self, item: dict) -> ModelInfo:
        model_id = item.get("id", "")
        pricing = item.get("pricing") or {}
        # OpenRouter marks free models both by a `:free` id suffix and by zero
        # prompt pricing. Ollama reports no pricing at all, and a model you are
        # already hosting is free by any definition.
        priced = _as_float(pricing.get("prompt"))
        free = (
            model_id.endswith(":free")
            or (priced is not None and priced == 0.0)
            or (not pricing and self.local)
        )
        ctx = item.get("context_length") or (item.get("top_provider") or {}).get(
            "context_length"
        )
        return ModelInfo(
            id=model_id,
            context_length=int(ctx) if ctx else None,
            params_b=parse_params_b(model_id),
            free=free,
            backend=self.name,
            text_only=_is_text_only(item.get("architecture") or {}),
        )

    # -- inference --------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
    ) -> str:
        """One completion. Returns the assistant's text, raises on failure."""
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.post(
                    f"{self.cfg.base_url}/chat/completions",
                    json=body,
                    headers=self._headers(),
                    timeout=timeout_s,
                )
            except httpx.HTTPError as exc:
                raise BackendUnavailable(f"{self.name}: {exc}") from exc

            if resp.status_code == 429:
                # Expected traffic on a free tier. Honour Retry-After when the
                # provider sends one; otherwise exponential backoff with jitter
                # so parallel callers do not resynchronise on the same second.
                last_error = RateLimited(f"{self.name}: 429")
                if attempt == MAX_RETRIES - 1:
                    break
                time.sleep(_retry_delay(resp.headers.get("Retry-After"), attempt))
                continue

            if resp.status_code >= 400:
                raise BackendError(
                    f"{self.name}: HTTP {resp.status_code}: {resp.text[:300]}"
                )

            try:
                return resp.json()["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, ValueError) as exc:
                raise BackendError(f"{self.name}: malformed response: {exc}") from exc

        raise last_error or BackendError(f"{self.name}: exhausted retries")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        return headers

    def close(self) -> None:
        self._client.close()


def _retry_delay(retry_after: str | None, attempt: int) -> float:
    if retry_after:
        try:
            return min(float(retry_after), 30.0)
        except ValueError:
            pass
    return BACKOFF_BASE_S * (2**attempt) + random.uniform(0, 0.5)


def _is_text_only(architecture: dict) -> bool:
    """Whether a model both accepts and returns plain text.

    Output modalities are what matter: a model that emits audio or images is a
    generator of something we cannot use, however large its context window.
    Accepting images as *input* is harmless, so multimodal input is allowed.

    A provider that reports no architecture at all — Ollama, LM Studio — is
    assumed to be serving text, which is what those tools are for.
    """
    outputs = architecture.get("output_modalities")
    if not outputs:
        return True
    return set(outputs) == {"text"}


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
