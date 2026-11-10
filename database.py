"""
database.py - Supabase persistence layer for PUGU.

Stores one row of preferences per Google user in the `user_settings` table.
All credentials are read from st.secrets["supabase"].
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import streamlit as st
from supabase import Client, create_client

TABLE_NAME = "user_settings"

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