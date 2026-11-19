"""YAML configs with single inheritance (`base_config: other.yaml`, child keys win)."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    base = cfg.pop("base_config", None)
    if base:
        parent = load_config(path.parent / base)
        parent.update(cfg)
        cfg = parent

    # `--set key=value` from the command line (value parsed as YAML: 3 -> int, [a,b] -> list)
    for item in overrides or []:
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"override must look like key=value, got {item!r}")
        cfg[key.strip()] = _parse(value)
    return cfg


def _parse(value: str):
    parsed = yaml.safe_load(value)
    if isinstance(parsed, str):   # YAML 1.1 reads "1e-3" as a string
        try:
            return float(parsed)
        except ValueError:
            pass
    return parsed
