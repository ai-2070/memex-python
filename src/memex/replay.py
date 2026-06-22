"""Event-log replay. Integrity-tolerant: per-item failures are collected in
``skipped`` rather than thrown. Includes a strict ISO-8601 parser ported
verbatim from the TS library (rejects sub-ms precision, validates calendar
fields, requires ``Z`` or an explicit offset) so replay ordering is deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, NamedTuple, cast

from .errors import InvalidTimestampError
from .graph import GraphState, create_graph_state
from .models import MemoryLifecycleEvent
from .reducer import apply_command

__all__ = ["ReplayFailure", "ReplayResult", "replay_commands", "replay_from_envelopes"]


@dataclass
class ReplayFailure:
    # dataclass (not NamedTuple) so the `index` field does not clash with
    # tuple.index under strict typing.
    index: int
    error: Exception
    command: Any = None
    envelope: Any = None


class ReplayResult(NamedTuple):
    state: GraphState
    events: list[MemoryLifecycleEvent]
    skipped: list[ReplayFailure]


def replay_commands(commands: list[Any]) -> ReplayResult:
    state = create_graph_state()
    all_events: list[MemoryLifecycleEvent] = []
    skipped: list[ReplayFailure] = []

    for i, cmd in enumerate(commands):
        try:
            result = apply_command(state, cmd)
            state = result.state
            all_events.extend(result.events)
        except Exception as err:  # noqa: BLE001 - integrity-tolerant by design
            skipped.append(ReplayFailure(index=i, command=cmd, error=err))

    return ReplayResult(state, all_events, skipped)


# Strict ISO 8601, milliseconds-only precision, explicit offset or Z.
_ISO_8601_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?(?:Z|([+-])(\d{2}):(\d{2}))$"
)

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _is_leap_year(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0


def _days_in_month(year: int, month: int) -> int:
    if month == 2:
        return 29 if _is_leap_year(year) else 28
    if month in (4, 6, 9, 11):
        return 30
    return 31


def parse_iso_ts(ts: str) -> int:
    m = _ISO_8601_RE.match(ts)
    if not m:
        raise InvalidTimestampError(f'Invalid envelope timestamp: "{ts}" (expected ISO 8601)')

    year, month, day = int(m[1]), int(m[2]), int(m[3])
    hour, minute, second = int(m[4]), int(m[5]), int(m[6])
    ms = int(m[7].ljust(3, "0")) if m[7] else 0

    if (
        month < 1 or month > 12
        or day < 1 or day > _days_in_month(year, month)
        or hour > 23 or minute > 59 or second > 59
    ):
        raise InvalidTimestampError(f'Invalid envelope timestamp: "{ts}" (calendar fields out of range)')

    try:
        dt = datetime(year, month, day, hour, minute, second, ms * 1000, tzinfo=timezone.utc)
    except ValueError as err:
        raise InvalidTimestampError(f'Invalid envelope timestamp: "{ts}" ({err})') from err

    delta = dt - _EPOCH
    epoch = delta.days * 86_400_000 + delta.seconds * 1000 + delta.microseconds // 1000

    if m[8]:
        off_h, off_m = int(m[9]), int(m[10])
        if off_h > 23 or off_m > 59:
            raise InvalidTimestampError(f'Invalid envelope timestamp: "{ts}" (bad offset)')
        sign = 1 if m[8] == "-" else -1
        epoch += sign * (off_h * 60 + off_m) * 60 * 1000

    return epoch


def _env_ts(env: Any) -> str:
    # An envelope is a dict (e.g. from JSON) or an EventEnvelope model; its `ts`
    # is always an ISO string.
    return cast(str, env["ts"] if isinstance(env, dict) else env.ts)


def _env_payload(env: Any) -> Any:
    # The payload is genuinely heterogeneous — a command model or a raw dict —
    # so Any is the honest type; apply_command re-validates it.
    return env["payload"] if isinstance(env, dict) else env.payload


def replay_from_envelopes(envelopes: list[Any]) -> ReplayResult:
    skipped: list[ReplayFailure] = []
    sortable: list[tuple[Any, int, int]] = []  # (env, ts, original index)

    for i, env in enumerate(envelopes):
        try:
            ts = parse_iso_ts(_env_ts(env))
            sortable.append((env, ts, i))
        except Exception as err:  # noqa: BLE001 - integrity-tolerant by design
            skipped.append(ReplayFailure(index=i, envelope=env, error=err))

    sortable.sort(key=lambda x: x[1])

    state = create_graph_state()
    all_events: list[MemoryLifecycleEvent] = []

    for env, _ts, index in sortable:
        try:
            result = apply_command(state, _env_payload(env))
            state = result.state
            all_events.extend(result.events)
        except Exception as err:  # noqa: BLE001 - integrity-tolerant by design
            skipped.append(ReplayFailure(index=index, envelope=env, error=err))

    return ReplayResult(state, all_events, skipped)
