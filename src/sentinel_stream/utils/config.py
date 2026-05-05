"""Configuration loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config from disk.

    Resolution order:
        1. The ``path`` argument, if provided.
        2. The ``SENTINEL_CONFIG_PATH`` environment variable.
        3. The default ``config/config.yaml`` shipped with the package.
    """
    candidate = path or os.environ.get("SENTINEL_CONFIG_PATH") or _DEFAULT_CONFIG_PATH
    with open(candidate) as f:
        return yaml.safe_load(f)
