"""
database.py - Supabase persistence layer for PUGU.

* user_settings : one row of preferences per Google user.
* ai_usage      : AI tokens used per user (and by everyone together) per UTC day.

All credentials are read from st.secrets["supabase"].
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import streamlit as st
from supabase import Client, create_client

logger = logging.getLogger(__name__)

TABLE_NAME = "user_settings"
USAGE_TABLE = "ai_usage"
GLOBAL_USAGE_ID = "__all__"  # the usage row that counts everyone together

# Choices shared with the UI so the database never stores unknown values.
FONT_CHOICES = ("OpenDyslexic", "Arial")
BG_COLOR_CHOICES = ("White", "Soft Yellow", "Pale Blue", "Mint Green", "Peach")
CHUNK_MINUTES_MIN = 5
CHUNK_MINUTES_MAX = 60
CHUNK_MINUTES_STEP = 5

DEFAULT_SETTINGS: dict[str, Any] = {
    "chunk_minutes": 25,
    "font_choice": "OpenDyslexic",
    "tts_enabled": False,
    "bg_color": "White",
}


@st.cache_resource(show_spinner=False)
def get_client() -> Client:
    """Create (once per server process) and return the Supabase client."""
    config = st.secrets["supabase"]
    return create_client(config["url"], config["key"])


# --------------------------------------------------------------------------- #
# User settings
# --------------------------------------------------------------------------- #
def normalise_settings(raw: dict[str, Any] | None) -> dict[str, Any]:
    """
    Merge raw values over the defaults and coerce them into valid ranges/choices.

    Any missing or invalid field silently falls back to its default.
    """
    settings = dict(DEFAULT_SETTINGS)
    if not raw:
        return settings

    try:
        minutes = int(raw.get("chunk_minutes", settings["chunk_minutes"]))
    except (TypeError, ValueError):
        minutes = settings["chunk_minutes"]
    minutes = CHUNK_MINUTES_STEP * round(minutes / CHUNK_MINUTES_STEP)
    settings["chunk_minutes"] = max(CHUNK_MINUTES_MIN, min(CHUNK_MINUTES_MAX, minutes))

    if raw.get("font_choice") in FONT_CHOICES:
        settings["font_choice"] = raw["font_choice"]
    if raw.get("bg_color") in BG_COLOR_CHOICES:
        settings["bg_color"] = raw["bg_color"]
    if isinstance(raw.get("tts_enabled"), bool):
        settings["tts_enabled"] = raw["tts_enabled"]

    return settings


def get_user_settings(google_id: str) -> dict[str, Any]:
    """
    Return the saved settings for a Google user.

    New users (no row yet) and any database error return the default settings.
    """
    try:
        response = (
            get_client()
            .table(TABLE_NAME)
            .select("chunk_minutes, font_choice, tts_enabled, bg_color")
            .eq("google_id", google_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:  # network, auth, or missing-table problems
        st.warning(f"Could not load your saved settings, using defaults. ({exc})")
        return dict(DEFAULT_SETTINGS)

    rows = response.data or []
    return normalise_settings(rows[0] if rows else None)


def upsert_user_settings(
    google_id: str, email: str | None, settings: dict[str, Any]
) -> bool:
    """
    Insert or update the settings row for a Google user.

    Returns True on success, False (after showing a warning) on failure.
    """
    row = {
        "google_id": google_id,
        "email": email,
        **normalise_settings(settings),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        get_client().table(TABLE_NAME).upsert(row, on_conflict="google_id").execute()
    except Exception as exc:
        st.warning(f"Could not save your settings. ({exc})")
        return False
    return True


# --------------------------------------------------------------------------- #
# AI usage (used by the limits in app.py)
# --------------------------------------------------------------------------- #
def _today_utc() -> str:
    """Today's date in UTC, the same day boundary Groq uses for daily limits."""
    return datetime.now(timezone.utc).date().isoformat()


def get_usage_today(google_id: str) -> tuple[int, int] | None:
    """
    Return (tokens this user used today, tokens everyone used today).

    Returns None if usage cannot be read, so the caller can refuse AI use
    instead of running without protection.
    """
    try:
        rows = (
            get_client()
            .table(USAGE_TABLE)
            .select("google_id, tokens")
            .eq("day", _today_utc())
            .in_("google_id", [google_id, GLOBAL_USAGE_ID])
            .execute()
            .data
            or []
        )
    except Exception:
        logger.warning("Could not read AI usage.", exc_info=True)
        return None

    user_tokens = sum(r["tokens"] for r in rows if r["google_id"] == google_id)
    global_tokens = sum(r["tokens"] for r in rows if r["google_id"] == GLOBAL_USAGE_ID)
    return user_tokens, global_tokens


def record_ai_usage(google_id: str, tokens: int) -> None:
    """Add `tokens` to this user's and the global counter for today (atomic, in SQL)."""
    if tokens <= 0:
        return
    try:
        get_client().rpc(
            "record_ai_usage", {"p_google_id": google_id, "p_tokens": int(tokens)}
        ).execute()
    except Exception:
        logger.warning("Could not record AI usage.", exc_info=True)


def get_admin_report() -> dict[str, Any] | None:
    """Today's usage per user plus the most recent users, for the admin tab."""
    try:
        client = get_client()
        usage_rows = (
            client.table(USAGE_TABLE)
            .select("google_id, tokens, requests")
            .eq("day", _today_utc())
            .execute()
            .data
            or []
        )
        user_rows = (
            client.table(TABLE_NAME)
            .select("google_id, email, updated_at")
            .order("updated_at", desc=True)
            .limit(200)
            .execute()
            .data
            or []
        )
    except Exception:
        logger.warning("Could not build the admin report.", exc_info=True)
        return None

    emails = {row["google_id"]: row.get("email") for row in user_rows}
    per_user = [
        {
            "user": emails.get(row["google_id"]) or row["google_id"],
            "tokens": row["tokens"],
            "requests": row["requests"],
        }
        for row in usage_rows
        if row["google_id"] != GLOBAL_USAGE_ID
    ]
    per_user.sort(key=lambda item: item["tokens"], reverse=True)

    everyone = next((r for r in usage_rows if r["google_id"] == GLOBAL_USAGE_ID), None)
    return {
        "global_tokens": everyone["tokens"] if everyone else 0,
        "global_requests": everyone["requests"] if everyone else 0,
        "per_user": per_user,
        "users": [
            {"email": row.get("email"), "last_saved": row.get("updated_at")}
            for row in user_rows
        ],
    }