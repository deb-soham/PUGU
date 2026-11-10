"""
app.py - PUGU (Personalized Understanding and Guidance Unit)

Entry point: configuration check, Google login (blocks the app), per-user
settings, sidebar, and tab routing for the ADHD and Dyslexia tools.
"""

from __future__ import annotations
try:
    import truststore

    truststore.inject_into_ssl()  # trust the Windows certificate store (needed on networks that inspect HTTPS)
except Exception:
    pass
import streamlit as st
import streamlit.components.v1 as components
from groq import Groq

import adhd_module
import database
import dyslexia_module

# Last-resort model name, used only if the model list can't be fetched.
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Best first. Plain chat models come before "reasoning" models, because reasoning
# models spend part of their output budget on hidden thinking.
PREFERRED_MODELS = (
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
)

# Model names containing these words can't do chat (speech, safety filters, etc.).
EXCLUDED_KEYWORDS = (
    "whisper", "tts", "orpheus", "playai", "guard", "safeguard", "embed", "compound",
)

st.set_page_config(page_title="PUGU", page_icon="🧠", layout="wide")


# --------------------------------------------------------------------------- #
# Configuration, login and session helpers
# --------------------------------------------------------------------------- #
def secrets_section_exists(name: str) -> bool:
    """True if st.secrets has a section called `name` (safe if no secrets file)."""
    try:
        return name in st.secrets
    except Exception:
        return False


def check_configuration() -> None:
    """Stop with a helpful message if required secrets are missing."""
    missing = [n for n in ("auth", "supabase") if not secrets_section_exists(n)]
    if missing:
        st.error(
            "Missing secrets section(s): "
            + ", ".join(f"[{name}]" for name in missing)
            + ". Add them to .streamlit/secrets.toml (see the setup instructions)."
        )
        st.stop()


@st.cache_data(ttl=3600, show_spinner=False)
def list_groq_models(api_key: str) -> list[str]:
    """Return the model IDs this Groq key can use (cached for one hour)."""
    client = Groq(api_key=api_key, timeout=15.0, max_retries=1)
    return sorted(model.id for model in client.models.list().data)


def pick_model(available: list[str]) -> str | None:
    """Choose the best usable chat model from the list Groq returned."""
    usable = [
        name for name in available
        if not any(word in name.lower() for word in EXCLUDED_KEYWORDS)
    ]
    for preferred in PREFERRED_MODELS:
        if preferred in usable:
            return preferred
    for name in usable:  # any other Llama chat model
        if "llama" in name.lower():
            return name
    return usable[0] if usable else None


def get_ai_config() -> tuple[str | None, str]:
    """
    Return (api_key, model) for Groq.

    A `model` line in secrets always wins. Otherwise the best available model
    is chosen automatically from what the key can use.
    """
    try:
        section = st.secrets["groq"]
        api_key = section.get("api_key") or None
        pinned_model = section.get("model")
    except Exception:
        return None, DEFAULT_MODEL

    if not api_key:
        return None, DEFAULT_MODEL
    if pinned_model:
        return api_key, pinned_model

    try:
        return api_key, pick_model(list_groq_models(api_key)) or DEFAULT_MODEL
    except Exception:
        return api_key, DEFAULT_MODEL


def require_login() -> dict:
    """Block the app until the user signs in with Google; return their identity."""
    if not st.user.is_logged_in:
        st.title("🧠 PUGU")
        st.subheader("Personalized Understanding and Guidance Unit")
        st.write(
            "Calm, step-by-step support for focus and reading. "
            "Please sign in with Google to continue."
        )
        if st.button("🔐 Sign in with Google", type="primary"):
            st.login("google")
        st.stop()

    email = st.user.get("email")
    google_id = st.user.get("sub") or email
    if not google_id:
        st.error("Google did not return an account ID. Please log out and try again.")
        st.button("Log out", on_click=logout)
        st.stop()

    return {
        "google_id": str(google_id),
        "email": email,
        "name": st.user.get("name") or email or "friend",
    }


def logout() -> None:
    """Clear this session's data and log out of Google."""
    st.session_state.clear()
    st.logout()


def init_session_state(user: dict) -> None:
    """Load the user's saved settings into session state (once per login)."""
    if st.session_state.get("owner_id") == user["google_id"]:
        return

    settings = database.get_user_settings(user["google_id"])
    st.session_state.update(
        {
            "owner_id": user["google_id"],
            "saved_settings": settings,
            # Widget values are pre-set here so widgets start from saved settings.
            "w_chunk_minutes": settings["chunk_minutes"],
            "w_font": settings["font_choice"],
            "w_bg": settings["bg_color"],
            "w_tts": settings["tts_enabled"],
            # ADHD tool state.
            "steps": [],
            "step_index": 0,
            "skipped": [],
            "used_ai": False,
            "active_task": "",
            "active_minutes": settings["chunk_minutes"],
            "parked_thoughts": [],
            "celebrate": False,
            # Dyslexia tool state.
            "overlay_text": "",
            "overlay_wanted_ai": False,
            "overlay_used_ai": False,
        }
    )


def current_settings() -> dict:
    """Collect the current widget values into a settings dict."""
    return {
        "chunk_minutes": st.session_state.w_chunk_minutes,
        "font_choice": st.session_state.w_font,
        "tts_enabled": st.session_state.w_tts,
        "bg_color": st.session_state.w_bg,
    }


def persist_settings(user: dict) -> None:
    """Save the current settings to Supabase and remember them in session state."""
    settings = current_settings()
    if database.upsert_user_settings(user["google_id"], user["email"], settings):
        st.session_state.saved_settings = settings
        st.toast("Preferences saved", icon="💾")


# --------------------------------------------------------------------------- #
# ADHD tab callbacks (run before the next rerun, so balloons are not cut off)
# --------------------------------------------------------------------------- #
def on_done() -> None:
    st.session_state.step_index += 1
    st.session_state.celebrate = True


def on_skip() -> None:
    steps = st.session_state.steps
    index = st.session_state.step_index
    if index < len(steps):
        st.session_state.skipped.append(steps[index])
    st.session_state.step_index = index + 1


def on_park_thought() -> None:
    thought = st.session_state.brain_dump_input.strip()
    if thought:
        st.session_state.parked_thoughts.append(thought)
    st.session_state.brain_dump_input = ""


def on_restart() -> None:
    st.session_state.update(
        steps=[], step_index=0, skipped=[], active_task="", celebrate=False
    )
    st.session_state.task_input = ""


# --------------------------------------------------------------------------- #
# ADHD tab
# --------------------------------------------------------------------------- #
def render_brain_dump() -> None:
    """The 'I'm Distracted' expander for parking off-topic thoughts."""
    with st.expander("🌀 I'm Distracted - park a thought"):
        st.text_area(
            "Brain Dump: write it down, then get back to your step.",
            key="brain_dump_input",
            height=90,
        )
        st.button("🅿️ Park it", on_click=on_park_thought)
        if st.session_state.parked_thoughts:
            st.markdown("**Parked for later:**")
            for thought in st.session_state.parked_thoughts:
                st.markdown(f"- {thought}")
        st.caption("Parked thoughts stay in this session only.")


def render_blinkers() -> None:
    """
    Blinkers UI: show exactly ONE step at a time.

    Shows a clear warning (with the reason) when the AI step-maker failed and
    generic fallback steps are being shown instead.
    """
    steps = st.session_state.steps
    if not steps:
        return

    st.divider()

    # Micro-reward: the callback sets this flag, and balloons are shown on the
    # rerun that follows so they are not cut off.
    if st.session_state.celebrate:
        st.balloons()
        st.session_state.celebrate = False

    index = st.session_state.step_index
    total = len(steps)

    st.caption(f"Task: {st.session_state.active_task}")

    if not st.session_state.used_ai:
        st.warning(
            "The AI step-maker is not working, so these are generic steps. "
            f"Reason: {st.session_state.get('ai_error') or 'unknown'}"
        )

    if index >= total:
        # Finished (or skipped) every step.
        st.success("🎉 All steps finished. Great work!")
        if st.session_state.skipped:
            st.markdown("**You skipped these. Come back to them when you're ready:**")
            for step in st.session_state.skipped:
                st.markdown(f"- {step}")
        st.button("Start a new task", on_click=on_restart)
    else:
        # Current step only: no list of upcoming steps, to reduce overwhelm.
        st.progress(index / total, text=f"Step {index + 1} of {total}")
        with st.container(border=True):
            st.markdown(f"## {steps[index]}")
            st.caption(
                f"Give it about {st.session_state.active_minutes} minutes. "
                "Nothing else matters right now."
            )

        col_done, col_skip = st.columns([2, 1])
        col_done.button("✅ Done, next step", type="primary", on_click=on_done)
        col_skip.button("⏭️ Skip step", on_click=on_skip)

    render_brain_dump()


def render_adhd_tab(user: dict) -> None:
    st.subheader("🎯 Break a big task into tiny steps")
    st.text_area(
        "What do you need to get done?",
        key="task_input",
        height=110,
        max_chars=500,
        placeholder="e.g. Write the introduction for my history essay",
    )
    st.slider(
        "Minutes per step",
        min_value=database.CHUNK_MINUTES_MIN,
        max_value=database.CHUNK_MINUTES_MAX,
        step=database.CHUNK_MINUTES_STEP,
        key="w_chunk_minutes",
    )

    if st.button("🧩 Break it down", type="primary"):
        task = st.session_state.task_input.strip()
        if not task:
            st.warning("Please type a task first.")
        else:
            persist_settings(user)
            api_key, model = get_ai_config()
            minutes = st.session_state.w_chunk_minutes
            with st.spinner("Breaking your task into small steps..."):
                steps, used_ai, ai_error = adhd_module.chunk_task(task, minutes, api_key, model)
            st.session_state.update(
                steps=steps,
                step_index=0,
                skipped=[],
                used_ai=used_ai,
                ai_error=ai_error,
                active_task=task,
                active_minutes=minutes,
            )

    render_blinkers()


# --------------------------------------------------------------------------- #
# Dyslexia tab
# --------------------------------------------------------------------------- #
def render_dyslexia_tab(user: dict) -> None:
    st.subheader("📖 Reading overlay")
    st.text_area(
        "Paste the text you want to read",
        key="dyslexia_input",
        height=200,
        max_chars=dyslexia_module.MAX_INPUT_CHARS,
    )

    col_font, col_bg, col_tts = st.columns(3)
    col_font.selectbox("Typography", database.FONT_CHOICES, key="w_font")
    col_bg.selectbox("Background color", database.BG_COLOR_CHOICES, key="w_bg")
    col_tts.toggle("🔊 Read aloud (TTS)", key="w_tts")

    simplify = st.checkbox("✨ Simplify difficult words with AI", value=True)

    if st.button("📄 Render text", type="primary"):
        raw = st.session_state.dyslexia_input.strip()
        if not raw:
            st.warning("Please paste some text first.")
        else:
            persist_settings(user)
            final_text, used_ai = raw, False
            if simplify:
                api_key, model = get_ai_config()
                with st.spinner("Simplifying the wording..."):
                    final_text, used_ai = dyslexia_module.simplify_text(raw, api_key, model)
            st.session_state.update(
                overlay_text=final_text,
                overlay_wanted_ai=simplify,
                overlay_used_ai=used_ai,
            )

    text = st.session_state.overlay_text
    if text:
        if st.session_state.overlay_wanted_ai and not st.session_state.overlay_used_ai:
            st.info("Couldn't simplify the text right now, so your original text is shown.")
        # Fonts, colors and TTS follow the dropdowns instantly (no new AI call).
        overlay = dyslexia_module.build_overlay_html(
            text,
            st.session_state.w_font,
            st.session_state.w_bg,
            st.session_state.w_tts,
        )
        components.html(
            overlay,
            height=dyslexia_module.estimate_height(text, st.session_state.w_tts),
            scrolling=True,
        )


# --------------------------------------------------------------------------- #
# Sidebar and main
# --------------------------------------------------------------------------- #
def render_sidebar(user: dict) -> None:
    saved = st.session_state.saved_settings
    with st.sidebar:
        st.header("🧠 PUGU")
        st.write(f"Signed in as **{user['name']}**")
        if user["email"]:
            st.caption(user["email"])
        st.button("Log out", on_click=logout)
        st.divider()
        st.subheader("Saved preferences")
        st.markdown(
            f"- Minutes per step: **{saved['chunk_minutes']}**\n"
            f"- Typography: **{saved['font_choice']}**\n"
            f"- Background: **{saved['bg_color']}**\n"
            f"- Read aloud: **{'On' if saved['tts_enabled'] else 'Off'}**"
        )
        st.caption(f"AI model: {get_ai_config()[1]}")
        st.caption(
            "Saved when you click 'Break it down' or 'Render text', "
            "and loaded automatically next time you sign in."
        )


def main() -> None:
    check_configuration()
    user = require_login()
    init_session_state(user)

    st.title("🧠 PUGU")
    st.caption("Personalized Understanding and Guidance Unit")

    tab_adhd, tab_dyslexia = st.tabs(["🎯 Focus (ADHD)", "📖 Read (Dyslexia)"])
    with tab_adhd:
        render_adhd_tab(user)
    with tab_dyslexia:
        render_dyslexia_tab(user)

    # Rendered last so it reflects settings saved during this run.
    render_sidebar(user)


main()