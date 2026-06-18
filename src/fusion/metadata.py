"""Run-metadata helpers.

Captures the bits a future user needs to reproduce a run: package
version, Python / platform info, and a content hash for any input
file the caller wants to pin.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import TypedDict


class EnvironmentInfo(TypedDict):
    """Minimal environment fingerprint recorded in ``run_metadata.json``."""

    python: str
    platform: str


def file_hash(path: str | Path) -> str:
    """SHA-256 of a file's bytes, as hex."""
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def environment_info() -> EnvironmentInfo:
    """Minimum environment fingerprint for ``run_metadata.json``."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
    }
