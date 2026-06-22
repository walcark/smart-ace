"""Stable short hashing helper, shared by the cacheable `.key()` methods."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(payload: Any, length: int = 12) -> str:
    """Deterministic short hex digest of a JSON-serialisable payload.

    ``sort_keys`` makes the digest order-independent; ``default=str`` lets
    non-JSON values (e.g. a truncation object) fall back to their ``str``.
    """
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:length]
