"""Internal clock helpers.

Isolated in one module so tests can monkeypatch ``now_ms`` / ``now_iso``
deterministically (the analog of stubbing ``Date.now`` / ``Date.toISOString``).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone


def now_ms() -> int:
    """Current unix time in milliseconds (mirrors JS ``Date.now()``)."""
    return int(time.time() * 1000)


def now_iso() -> str:
    """Current UTC time as ISO-8601 with millisecond precision and a ``Z`` suffix.

    Matches JavaScript ``new Date().toISOString()`` byte-for-byte
    (e.g. ``"2024-06-22T13:45:30.123Z"``), which the strict replay parser
    in :mod:`memex.replay` accepts.
    """
    dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
