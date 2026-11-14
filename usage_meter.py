"""
usage_meter.py - Counts the AI tokens used during one app action.

The AI modules call add() after every model reply. The app calls reset()
before an action and total() after it, then records the total in the database.
Each Streamlit session runs in its own thread, so the counter is thread-local
and two users can never mix up their numbers. No Streamlit dependency.
"""

from __future__ import annotations

import threading

_local = threading.local()


def reset() -> None:
    """Start counting from zero."""
    _local.tokens = 0


def add(tokens) -> None:
    """Add tokens used by one model reply."""
    try:
        amount = int(tokens or 0)
    except (TypeError, ValueError):
        amount = 0
    _local.tokens = getattr(_local, "tokens", 0) + max(0, amount)


def total() -> int:
    """Tokens counted since the last reset()."""
    return getattr(_local, "tokens", 0)