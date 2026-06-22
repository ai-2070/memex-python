"""UUIDv7 generation and timestamp extraction.

Mirrors the single runtime dependency of the TypeScript library (``uuidv7``)
with a tiny internal implementation so the only third-party dependency is
``pydantic``. UUIDv7 (RFC 9562) encodes a 48-bit big-endian millisecond
timestamp in its first six bytes; we decode exactly those bytes the same way
the TS ``safeExtractTimestamp`` does.
"""

from __future__ import annotations

import os
from uuid import UUID

from . import _time

__all__ = ["uuid7", "safe_extract_timestamp"]


def uuid7(ms: int | None = None) -> str:
    """Generate a UUIDv7 string for ``ms`` (defaults to the current time)."""
    if ms is None:
        ms = _time.now_ms()
    ts = ms & ((1 << 48) - 1)
    rand = os.urandom(10)

    b = bytearray(16)
    b[0] = (ts >> 40) & 0xFF
    b[1] = (ts >> 32) & 0xFF
    b[2] = (ts >> 24) & 0xFF
    b[3] = (ts >> 16) & 0xFF
    b[4] = (ts >> 8) & 0xFF
    b[5] = ts & 0xFF
    b[6] = 0x70 | (rand[0] & 0x0F)  # version 7 in the high nibble
    b[7] = rand[1]
    b[8] = 0x80 | (rand[2] & 0x3F)  # RFC 4122 variant (0b10) in the top bits
    b[9:16] = rand[3:10]
    return str(UUID(bytes=bytes(b)))


def safe_extract_timestamp(value: str) -> int | None:
    """Decode the millisecond timestamp from a UUIDv7 id.

    Returns ``None`` for anything that is not a valid version-7 UUID, or whose
    encoded timestamp is non-positive — matching the TS helper's tolerance.
    """
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError, TypeError):
        return None
    if parsed.version != 7:
        return None
    b = parsed.bytes
    ts = (
        (b[0] << 40)
        | (b[1] << 32)
        | (b[2] << 24)
        | (b[3] << 16)
        | (b[4] << 8)
        | b[5]
    )
    return ts if ts > 0 else None
