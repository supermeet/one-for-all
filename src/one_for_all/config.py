"""Configuration — where every disposable name is allowed to live.

MODEL.md is emphatic that model ids are data, never code (ARCHITECTURE §4,
invariant 2), because free-tier lineups churn weekly. This module is the one
place a name may appear, and even here a name is only a *hint*: the router
fetches the live model list at runtime and treats these strings as ranking
preferences, matched as substrings. If every preferred name has vanished from
the provider, routing still succeeds by falling back to the generic criteria in
§3 — size for the curator, capability for the critic.

Zero configuration must work (invariant 6). The defaults below target a laptop
with no GPU and no paid key: OpenRouter's free tier for everything, with a local
Ollama backend that is simply absent until someone starts one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

# Roles from MODEL.md §1. Their requirements are opposing, which is the whole
# reason the model layer is a portfolio with a router rather than a choice.
ROLES = ("curate", "critique", "generate")

DEFAULTS: dict[str, Any] = {
    "backends": {
        # Remote default. Free tier, no card, no hardware requirement.
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "local": False,
            # Free tiers carry the loosest data terms — that is part of why they
            # are free — so this backend is never eligible for sensitive content.
            "free_only": True,
            "enabled": True,
        },
        # Local default. Absent on this laptop; present on the college PC or a
        # rented box. Nothing may assume it exists (MODEL.md §5).
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "api_key_env": None,
            "local": True,
            "free_only": False,
            "enabled": True,
        },
    },
    "roles": {
        # Volume tier: selection only, in the critical path of every request.
        "curate": {
            "prefer": ["gemma-3-4b", "gemma-3n", "phi-4-mini", "qwen3-4b", "qwen-2.5-3b"],
            "max_params_b": 8.0,
            "prefer_local": True,
            "timeout_s": 20.0,
            "max_tokens": 128,
            # Selection is deterministic work; sampling only adds format drift.
            "temperature": 0.0,
        },
        # Judgment tier: precision-first, tolerates latency, buy capability.
        "critique": {
            "prefer": ["qwen3", "llama-3.3", "deepseek", "phi-4", "mistral-small"],
            "max_params_b": None,
            "prefer_local": False,
            "timeout_s": 90.0,
            "max_tokens": 800,
            "temperature": 0.1,
        },
        # Usually the client's own model. We only need this headless.
        "generate": {
            "prefer": [],
            "max_params_b": None,
            "prefer_local": False,
            "timeout_s": 120.0,
            "max_tokens": 2048,
            "temperature": 0.3,
        },
    },
    "privacy": {
        # Substring triggers, lowercased. Deliberately crude: MODEL.md §11.5
        # lists the classification policy as an open question, and the cheap
        # error direction is a false positive — being forced local costs
        # latency, while a false negative puts a secret on a free endpoint.
        "patterns": [
            "api_key", "api key", "apikey", "secret", "password", "passwd",
            "token", "credential", "private key", "-----begin", "authorization:",
            ".env", "ssh-rsa", "bearer ",
        ],
        # If content is sensitive and no local backend is reachable, refuse
        # rather than degrade. Invariant 4 has no "unless it was inconvenient".
        "refuse_without_local": True,
    },
    "critic": {
        # Precision over recall. A false positive trains the user to ignore the
        # critic, which is fatal (MODEL.md §1.2). The bar starts high; §11.4
        # notes it can only be calibrated against real use.
        "confidence_bar": 0.7,
        "enabled": True,
    },
    "curator": {
        # Below ~3x, curation is not earning its complexity (MODEL.md §2).
        "min_compression_ratio": 3.0,
        # Skip the model entirely when there is little to select from — calling
        # a model to filter six items costs more than it saves.
        "min_candidates": 8,
        "enabled": True,
    },
}


def config_path() -> Path:
    return Path(user_config_dir("one-for-all", appauthor=False)) / "config.json"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge, so a user config naming one key does not silently drop the
    sibling defaults around it."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class BackendConfig:
    name: str
    base_url: str
    api_key: str | None
    local: bool
    free_only: bool
    enabled: bool


@dataclass(frozen=True)
class RoleConfig:
    name: str
    prefer: list[str]
    max_params_b: float | None
    prefer_local: bool
    timeout_s: float
    max_tokens: int
    temperature: float


@dataclass(frozen=True)
class Config:
    backends: list[BackendConfig]
    roles: dict[str, RoleConfig]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    def role(self, name: str) -> RoleConfig:
        if name not in self.roles:
            raise ValueError(f"unknown role {name!r}; expected one of {ROLES}")
        return self.roles[name]

    def backend(self, name: str) -> BackendConfig | None:
        return next((b for b in self.backends if b.name == name), None)

    @property
    def privacy_patterns(self) -> list[str]:
        return [p.lower() for p in self.raw["privacy"]["patterns"]]

    @property
    def refuse_without_local(self) -> bool:
        return bool(self.raw["privacy"]["refuse_without_local"])

    @property
    def confidence_bar(self) -> float:
        return float(self.raw["critic"]["confidence_bar"])


def load(path: Path | None = None) -> Config:
    """Load config, merging a user file over the defaults. A missing file is
    the normal case, not an error."""
    path = path or config_path()
    raw = DEFAULTS
    if path.exists():
        raw = _merge(DEFAULTS, json.loads(path.read_text(encoding="utf-8")))

    backends = []
    for name, spec in raw["backends"].items():
        env = spec.get("api_key_env")
        backends.append(
            BackendConfig(
                name=name,
                base_url=spec["base_url"].rstrip("/"),
                # Read at load time, never written to disk. Keys belong in the
                # environment; the config file is checked into nothing but is
                # still a file on disk with ordinary permissions.
                api_key=os.environ.get(env) if env else None,
                local=bool(spec.get("local", False)),
                free_only=bool(spec.get("free_only", False)),
                enabled=bool(spec.get("enabled", True)),
            )
        )

    roles = {
        name: RoleConfig(
            name=name,
            prefer=list(spec.get("prefer", [])),
            max_params_b=spec.get("max_params_b"),
            prefer_local=bool(spec.get("prefer_local", False)),
            timeout_s=float(spec.get("timeout_s", 60.0)),
            max_tokens=int(spec.get("max_tokens", 512)),
            temperature=float(spec.get("temperature", 0.0)),
        )
        for name, spec in raw["roles"].items()
    }
    return Config(backends=backends, roles=roles, raw=raw)


def write_default(path: Path | None = None) -> Path:
    """Materialise the defaults so there is something to edit. Never called
    automatically — a config file that appears on its own is a file nobody
    knows they own."""
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULTS, indent=2), encoding="utf-8")
    return path
