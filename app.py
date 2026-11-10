"""
app.py - PUGU (Personalized Understanding and Guidance Unit)

Entry point: Google login (blocks the app), per-user settings, the ADHD
"focus mode" (one step at a time, one work session at a time) and the Dyslexia
reading overlay.
"""

from __future__ import annotations

try:
    import truststore

    truststore.inject_into_ssl()  # trust the Windows certificate store (networks that inspect HTTPS)
except Exception:
    pass

import html

import streamlit as st
import streamlit.components.v1 as components
from groq import Groq

import adhd_module
import database
import dyslexia_module

st.set_page_config(page_title="PUGU", page_icon="🧠", layout="wide")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
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

# Widgets that must keep their value even in runs where they are not displayed
# (for example while focus mode hides the planning screen).
KEEP_ALIVE_KEYS = (
    "w_chunk_minutes", "w_font", "w_bg", "w_tts", "task_input", "dyslexia_input",
)

FOCUS_CSS = """
<style>
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
.focus-step {
  font-size: clamp(1.8rem, 4.5vw, 3rem); line-height: 1.35; font-weight: 600;
  text-align: center; padding: 8vh 0.5rem; min-height: 38vh;
  display: flex; align-items: center; justify-content: center;
}
.focus-sub { text-align: center; opacity: 0.7; margin: 0; }
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
  min-height: 3rem; font-size: 1.1rem;
}
</style>
"""


# --------------------------------------------------------------------------- #
# Configuration, AI model and login helpers
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


# --------------------------------------------------------------------------- #
# Session state and saved settings
# --------------------------------------------------------------------------- #
def empty_task_state() -> dict:
    """Fresh state for the ADHD tool (no task running)."""
    return {
        "focus_active": False,   # True while a task is in progress
        "task": "",
        "step_minutes": database.DEFAULT_SETTINGS["chunk_minutes"],
        "queue": [],             # [{"text": str, "depth": int}]; queue[0] is the current step
        "planning_done": False,  # True once the whole task is known to be complete
        "done_steps": [],
        "skipped_steps": [],
        "done_count": 0,
        "explanation": None,     # {"step": str, "text": str} for the current step
        "pending_action": None,  # "split", "explain" or "next_session", handled on the next run
        "notice": None,
        "ai_error": None,
        "plan_error": None,
        "celebrate": False,
    }


def init_session_state(user: dict) -> None:
    """Load the user's saved settings into session state (once per login)."""
    if (
        st.session_state.get("owner_id") == user["google_id"]
        and "focus_active" in st.session_state
    ):
        return

    settings = database.get_user_settings(user["google_id"])
    st.session_state.update(
        {
            "owner_id": user["google_id"],
            "saved_settings": settings,
            # Widget values are pre-set so widgets start from the saved settings.
            "w_chunk_minutes": settings["chunk_minutes"],
            "w_font": settings["font_choice"],
            "w_bg": settings["bg_color"],
            "w_tts": settings["tts_enabled"],
            "task_input": "",
            "dyslexia_input": "",
            "parked_thoughts": [],
            "overlay_text": "",
            "overlay_wanted_ai": False,
            "overlay_used_ai": False,
        }
    )
    st.session_state.update(empty_task_state())


def keep_widget_state_alive() -> None:
    """Stop Streamlit from deleting widget values while their widgets are hidden."""
    for key in KEEP_ALIVE_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


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
# ADHD tool: callbacks (they run before the next rerun)
# --------------------------------------------------------------------------- #
def on_done() -> None:
    s = st.session_state
    if not s.queue:
        return
    step = s.queue.pop(0)
    s.done_steps.append(step["text"])
    s.done_count += 1
    s.explanation = None
    s.celebrate = True  # balloons are shown on the rerun that follows


def on_skip() -> None:
    s = st.session_state
    if not s.queue:
        return
    step = s.queue.pop(0)
    s.skipped_steps.append(step["text"])
    s.explanation = None


def request_action(action: str) -> None:
    """Ask for a slow action (needs the AI); it is handled on the next run."""
    st.session_state.pending_action = action


def reset_task() -> None:
    """End the current task and return to the planning screen."""
    st.session_state.update(empty_task_state())
    st.session_state.task_input = ""


def retry_planning() -> None:
    st.session_state.plan_error = None


def finish_planning() -> None:
    st.session_state.plan_error = None
    st.session_state.planning_done = True


def clear_parked() -> None:
    st.session_state.parked_thoughts = []


# --------------------------------------------------------------------------- #
# ADHD tool: logic that may call the AI
# --------------------------------------------------------------------------- #
def start_task(user: dict, task: str) -> None:
    """Save settings, plan the first work session and switch to focus mode."""
    persist_settings(user)
    api_key, model = get_ai_config()
    minutes = st.session_state.w_chunk_minutes

    with st.spinner("Getting your first step ready..."):
        steps, complete, _used_ai, error = adhd_module.start_plan(task, minutes, api_key, model)

    st.session_state.update(empty_task_state())
    st.session_state.update(
        focus_active=True,
        task=task,
        step_minutes=minutes,
        queue=[{"text": step, "depth": 0} for step in steps],
        planning_done=complete,
        ai_error=error,
    )


def plan_next_session() -> None:
    """Plan the next work session. Only runs when the person asks for it."""
    s = st.session_state
    api_key, model = get_ai_config()
    with st.spinner("Planning your next session..."):
        steps, complete, error = adhd_module.continue_plan(
            s.task, s.step_minutes, s.done_steps, s.skipped_steps, api_key, model
        )
    if error:
        s.plan_error = error
        return
    s.queue = [{"text": step, "depth": 0} for step in steps]
    s.planning_done = complete


def handle_pending_action() -> None:
    """Run a requested 'next session', 'too big' or 'explain' action."""
    s = st.session_state
    action = s.pending_action
    s.pending_action = None
    if not action:
        return

    if action == "next_session":
        plan_next_session()
        return

    if not s.queue:
        return

    api_key, model = get_ai_config()
    current = s.queue[0]

    if action == "explain":
        with st.spinner("Finding a simpler way to say it..."):
            text, _used_ai, error = adhd_module.explain_step(
                current["text"], s.task, api_key, model
            )
        s.explanation = {"step": current["text"], "text": text}
        s.ai_error = error

    elif action == "split":
        if current["depth"] > adhd_module.MAX_SPLIT_DEPTH:
            s.notice = "This is already as small as it gets. Take a breath and just begin for two minutes."
            return
        if current["depth"] == adhd_module.MAX_SPLIT_DEPTH:
            pieces, error = [adhd_module.starter_step(current["text"])], None
        else:
            with st.spinner("Making it smaller..."):
                pieces, _used_ai, error = adhd_module.break_down_step(
                    current["text"], s.task, api_key, model
                )
        depth = current["depth"] + 1
        s.queue = [{"text": piece, "depth": depth} for piece in pieces] + s.queue[1:]
        s.explanation = None
        s.ai_error = error


# --------------------------------------------------------------------------- #
# ADHD tool: screens
# --------------------------------------------------------------------------- #
def render_planning_screen(user: dict) -> None:
    """The calm starting screen: describe the task, choose a step size."""
    st.subheader("🎯 What do you want to get done?")
    st.caption(
        "PUGU gives you one small step at a time, and every task has a clear finish line."
    )
    st.text_area(
        "Your task",
        key="task_input",
        height=110,
        max_chars=500,
        placeholder="e.g. Write the introduction for my history essay",
    )
    st.slider(
        "How big should each step be? (minutes)",
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
            start_task(user, task)
            st.rerun()


def render_current_step() -> None:
    """Show the current step in large type, with only a few choices."""
    s = st.session_state
    current = s.queue[0]

    if s.done_count:
        st.markdown(f'<p class="focus-sub">✅ {s.done_count} done</p>', unsafe_allow_html=True)
    if current["depth"] > 0:
        st.markdown('<p class="focus-sub">A smaller piece</p>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="focus-step">{html.escape(current["text"])}</div>',
        unsafe_allow_html=True,
    )

    explanation = s.explanation
    if explanation and explanation["step"] == current["text"]:
        st.info(explanation["text"], icon="💡")
    if s.notice:
        st.caption(s.notice)
        s.notice = None
    if s.ai_error:
        st.caption(f"⚠️ The AI is unavailable, so PUGU used simple built-in help. Reason: {s.ai_error}")

    st.button("✅ Done, next step", type="primary", on_click=on_done)

    col_big, col_explain, col_more = st.columns(3)
    col_big.button("🧩 This is too big", on_click=request_action, args=("split",))
    col_explain.button("💡 Explain this", on_click=request_action, args=("explain",))
    with col_more.popover("⋯ More"):
        st.button("⏭️ Skip step", on_click=on_skip)
        st.button("⏹ Stop this task", on_click=reset_task)


def render_plan_error() -> None:
    """Shown when the AI could not plan the next session."""
    s = st.session_state
    st.warning(f"I couldn't plan the next session. Reason: {s.plan_error}")
    col_retry, col_finish = st.columns(2)
    col_retry.button("🔄 Try again", on_click=retry_planning)
    col_finish.button("✅ I'm finished", on_click=finish_planning)


def render_completion() -> None:
    """
    Shown when the planned steps are used up.

    Either the whole task is complete, or this was one session of a bigger task:
    then it is a natural place to pause, and the person decides what happens next.
    """
    s = st.session_state
    more_to_do = not s.planning_done
    plural = "" if s.done_count == 1 else "s"

    if more_to_do:
        st.markdown(
            '<div class="focus-step">🌟 Great session. This is a good place to pause.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"You finished {s.done_count} step{plural}. Stretch, drink some water, rest your eyes. "
            "When you feel ready, PUGU can plan the next session."
        )
    else:
        st.markdown('<div class="focus-step">🎉 You did it.</div>', unsafe_allow_html=True)
        st.markdown(f"You finished {s.done_count} step{plural}. That took real effort.")

    if s.skipped_steps:
        st.markdown("**You skipped these. Come back to them when you're ready:**")
        for step in s.skipped_steps:
            st.markdown(f"- {step}")
    if s.parked_thoughts:
        st.markdown("You parked some thoughts. Open the list below when you're ready.")

    if more_to_do:
        col_next, col_stop = st.columns(2)
        col_next.button(
            "▶ Plan the next session",
            type="primary",
            on_click=request_action,
            args=("next_session",),
        )
        col_stop.button("✅ I'm finished for now", on_click=reset_task)
    else:
        st.button("Start a new task", type="primary", on_click=reset_task)


def render_parking_box() -> None:
    """Always-visible place to park off-topic thoughts without leaving the task."""
    s = st.session_state
    with st.form("park_form", clear_on_submit=True):
        thought = st.text_input(
            "Park a thought",
            placeholder="Something else on your mind? Write it here. It will wait for you.",
        )
        parked = st.form_submit_button("🅿️ Park it")
    if parked and thought.strip():
        s.parked_thoughts.append(thought.strip())
        st.toast("Parked. Back to your step.", icon="🅿️")

    if s.parked_thoughts:
        with st.expander(f"Parked thoughts ({len(s.parked_thoughts)})"):
            for item in s.parked_thoughts:
                st.markdown(f"- {item}")
            st.button("Clear parked thoughts", on_click=clear_parked)


def render_focus_mode() -> None:
    """Full-page focus mode: no title, tabs or sidebar, just the current step."""
    s = st.session_state
    st.markdown(FOCUS_CSS, unsafe_allow_html=True)

    if s.celebrate:
        st.balloons()
        s.celebrate = False

    handle_pending_action()

    _, center, _ = st.columns([1, 6, 1])
    with center:
        if s.plan_error:
            render_plan_error()
        elif not s.queue:
            render_completion()
        else:
            render_current_step()
        st.divider()
        render_parking_box()


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
            f"- Step size: **{saved['chunk_minutes']} min**\n"
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
    keep_widget_state_alive()

    if st.session_state.focus_active:
        render_focus_mode()
        return

    st.title("🧠 PUGU")
    st.caption("Personalized Understanding and Guidance Unit")

    tab_adhd, tab_dyslexia = st.tabs(["🎯 Focus (ADHD)", "📖 Read (Dyslexia)"])
    with tab_adhd:
        render_planning_screen(user)
    with tab_dyslexia:
        render_dyslexia_tab(user)

    # Rendered last so it reflects settings saved during this run.
    render_sidebar(user)


main()