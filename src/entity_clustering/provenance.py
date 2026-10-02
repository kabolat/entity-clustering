"""Small, deterministic provenance helpers for scientific runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def canonical_json_hash(value: Any) -> str:
    """Return a stable SHA-256 hash of JSON-compatible data."""

    encoded = json.dumps(value, default=str, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def git_tag() -> str | None:
    result = subprocess.run(["git", "describe", "--tags", "--exact-match"], check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def utc_run_id() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")


def environment_metadata() -> dict[str, Any]:
    packages = ("matplotlib", "numpy", "pandas", "pydantic", "scikit-learn", "scipy")
    installed: dict[str, str] = {}
    for package in packages:
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            installed[package] = "not-installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": installed,
        "git_commit": git_commit(),
        "git_tag": git_tag(),
    }
