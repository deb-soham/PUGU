"""
adhd_module.py - Step planning logic for PUGU's ADHD tools.

The app shows ONE step at a time and never reveals how many steps there are, but
every task still has a finish line. The AI plans one work SESSION at a time
(about SESSION_MINUTES of work). The number of steps in a session is derived
from the person's chosen step size (a session holds as many steps of that size
as fit in it), so there is no fixed step count. When a session is done, the
person decides whether to plan another one. PUGU never keeps handing out steps
on its own.

A person can also ask for a step to be split into smaller pieces or explained
in simpler words. Every AI function has a built-in fallback so the app works
without the AI. This module has no Streamlit dependency.
"""

from __future__ import annotations

import json
import logging
import math
import re

from groq import Groq

import usage_meter

logger = logging.getLogger(__name__)

# Tunable settings. None of these is shown to the user.
SESSION_MINUTES = 75    # one work session; bigger tasks are planned session by session
SAFETY_MAX_STEPS = 25   # absolute upper limit for one reply (guards against runaway output)
MAX_SPLIT_DEPTH = 2     # how many times one step may be split into smaller ones
_CONTEXT_STEPS = 12     # how many recent steps are sent back to the AI as context


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _ask(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float = 0.3,
) -> str:
    """Send one chat request to Groq and return the reply text. Raises on failure."""
    client = Groq(api_key=api_key, timeout=30.0, max_retries=1)

    # "gpt-oss" models think before answering and that hidden thinking uses up the
    # token budget, so they get a bigger budget and a request for low effort.
    options: dict = {"max_tokens": max_tokens}
    if "gpt-oss" in model.lower():
        options = {"max_tokens": max_tokens * 3, "extra_body": {"reasoning_effort": "low"}}

    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **options,
    )
    usage_meter.add(getattr(getattr(completion, "usage", None), "total_tokens", 0))
    text = (completion.choices[0].message.content or "").strip()
    if not text:
        raise ValueError(
            "The model returned an empty reply (its token budget may have been "
            "used up by reasoning)."
        )
    return text


def _extract_json(raw: str):
    """Find and parse the first JSON object or array in the model's reply."""
    match = re.search(r"\{.*\}|\[.*\]", raw, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON found in the model response.")
    return json.loads(match.group(0))


def _clean_step(step: str) -> str:
    """Remove leading numbering or bullet characters from a step."""
    return re.sub(r"^\s*(?:\d+[.)]|[-*\u2022])\s*", "", step).strip()


def _clean_steps(items) -> list[str]:
    """Clean steps, drop empty and duplicate ones, and apply the safety cap."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        step = _clean_step(str(item))
        key = step.lower()
        if step and key not in seen:
            seen.add(key)
            cleaned.append(step)
    return cleaned[:SAFETY_MAX_STEPS]


def _parse_plan(raw: str, allow_empty: bool = False) -> tuple[list[str], bool]:
    """
    Parse the model's reply into (steps, task_complete).

    If the reply does not say whether the task is complete, it is treated as
    complete, so a plan can never run on forever.
    """
    data = _extract_json(raw)
    if isinstance(data, list):  # a bare list means "this is the whole plan"
        items, complete = data, True
    elif isinstance(data, dict):
        items = data.get("steps", [])
        complete = bool(data.get("task_complete", data.get("finished", True)))
    else:
        raise ValueError("Unexpected JSON shape in the model response.")

    if not isinstance(items, list):
        raise ValueError("Expected a list of steps.")
    steps = _clean_steps(items)
    if not steps and not allow_empty:
        raise ValueError("The model returned no usable steps.")
    return steps, complete


def step_budget(step_minutes: int) -> int:
    """How many steps of the chosen size fit in one work session."""
    return max(1, min(SAFETY_MAX_STEPS, math.ceil(SESSION_MINUTES / max(1, step_minutes))))


def _apply_budget(
    steps: list[str], complete: bool, budget: int
) -> tuple[list[str], bool]:
    """Keep a session within its budget; anything cut off belongs to a later session."""
    if len(steps) > budget:
        return steps[:budget], False
    return steps, complete


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def _plan_system_prompt(step_minutes: int, budget: int) -> str:
    return (
        "You are a calm, encouraging executive-function coach for people with ADHD. "
        "You plan ONE work session of a task. "
        f"A session lasts about {SESSION_MINUTES} minutes and has at most {budget} steps. "
        f"Each step is ONE concrete piece of work that takes roughly {step_minutes} minutes "
        "of focus. Combine tiny actions into a single step instead of listing them "
        "separately. The first step must be easy to start. Start every step with a verb, "
        "keep it under 25 words, use no numbering and no emojis, and never mention how "
        "many steps there are. Use fewer steps if the task is small. "
        'Respond with ONLY a JSON object like {"steps": ["..."], "task_complete": true}. '
        'Set "task_complete" to true if these steps finish the WHOLE task. Set it to false '
        "only if the task is too big to finish in one session and more work will remain."
    )


_SPLIT_SYSTEM = (
    "A person with ADHD finds the following step too big to start. Split it into "
    "smaller concrete steps. Rules: each smaller step is ONE action that starts with a "
    "verb and takes only a couple of minutes; the first one must be almost effortless; "
    "keep each under 15 words; no numbering, no emojis; use only as many steps as are "
    'truly needed. Respond with ONLY a JSON object like {"steps": ["..."]}.'
)

_EXPLAIN_SYSTEM = (
    "You help a person with ADHD understand one step of their task. Explain the step in "
    "simple, friendly words in two or three short sentences, and give one concrete "
    "example. Plain text only: no markdown, no lists, and do not add new steps."
)


# --------------------------------------------------------------------------- #
# Fallbacks (used when the AI is unavailable)
# --------------------------------------------------------------------------- #
def _short(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def fallback_plan(task: str, step_minutes: int) -> list[str]:
    """Generic, complete plan used when the AI cannot plan the task."""
    label = _short(task)
    return [
        f"Get ready: gather everything you need for \"{label}\" and clear your workspace.",
        "Write one sentence that describes what \"finished\" looks like.",
        f"Start tiny: work on the very first small piece for {step_minutes} minutes.",
        f"Keep going: do the next small piece for {step_minutes} minutes.",
        "Wrap up: quickly check your work and write down anything still left to do.",
    ]


def fallback_split(step: str) -> list[str]:
    """Simple split used when the AI cannot split a step."""
    label = _short(step)
    return [
        f"Get just the first thing you need ready for: {label}",
        f"Do only the first two minutes of: {label}",
    ]


def fallback_explain(step: str) -> str:
    """Generic help used when the AI cannot explain a step."""
    return (
        "Read the step once, then do only the very first small action. "
        "You do not need to finish it yet. "
        "If it still feels like too much, press \"This is too big\"."
    )


def starter_step(step: str) -> str:
    """The smallest possible version of a step (used when splitting has gone far enough)."""
    return f"Set a timer for 2 minutes and just begin: {step}"


# --------------------------------------------------------------------------- #
# Public functions
# --------------------------------------------------------------------------- #
def start_plan(
    task: str, step_minutes: int, api_key: str | None, model: str
) -> tuple[list[str], bool, bool, str | None]:
    """
    Plan the first work session of a task.

    Returns (steps, task_complete, used_ai, error). `task_complete` is True when
    these steps finish the whole task. `error` explains why the fallback was used.
    """
    if not api_key:
        return (
            fallback_plan(task, step_minutes),
            True,
            False,
            "No Groq API key found. Check the [groq] section in .streamlit/secrets.toml.",
        )
    budget = step_budget(step_minutes)
    try:
        raw = _ask(
            api_key, model, _plan_system_prompt(step_minutes, budget), f"Task: {task}", 800
        )
        steps, complete = _parse_plan(raw)
        steps, complete = _apply_budget(steps, complete, budget)
        return steps, complete, True, None
    except Exception as exc:
        logger.warning("AI planning failed; using fallback.", exc_info=True)
        return fallback_plan(task, step_minutes), True, False, f"{type(exc).__name__}: {exc}"


def continue_plan(
    task: str,
    step_minutes: int,
    done_steps: list[str],
    skipped_steps: list[str],
    api_key: str | None,
    model: str,
) -> tuple[list[str], bool, str | None]:
    """
    Plan the NEXT work session. Only called when the person asks for it.

    Returns (steps, task_complete, error). With no API key the task simply ends.
    """
    if not api_key:
        return [], True, None

    lines = [f"Task: {task}", "Steps already done:"]
    lines += [f"- {item}" for item in done_steps[-_CONTEXT_STEPS:]] or ["- (none)"]
    if skipped_steps:
        lines.append("Steps the person skipped:")
        lines += [f"- {item}" for item in skipped_steps[-_CONTEXT_STEPS:]]
    lines.append(
        "Plan the next work session. Do not repeat steps that are already done. "
        "If nothing meaningful remains, return an empty list with task_complete true."
    )

    budget = step_budget(step_minutes)
    try:
        raw = _ask(
            api_key, model, _plan_system_prompt(step_minutes, budget), "\n".join(lines), 800
        )
        steps, complete = _parse_plan(raw, allow_empty=True)
    except Exception as exc:
        logger.warning("AI planning (next session) failed.", exc_info=True)
        return [], False, f"{type(exc).__name__}: {exc}"

    already = {item.lower() for item in done_steps + skipped_steps}
    steps = [item for item in steps if item.lower() not in already]
    steps, complete = _apply_budget(steps, complete, budget)
    return steps, complete or not steps, None


def break_down_step(
    step: str, task: str, api_key: str | None, model: str
) -> tuple[list[str], bool, str | None]:
    """
    Split one step into smaller steps.

    Returns (smaller_steps, used_ai, error).
    """
    if not api_key:
        return (
            fallback_split(step),
            False,
            "No Groq API key found. Check the [groq] section in .streamlit/secrets.toml.",
        )
    try:
        raw = _ask(
            api_key,
            model,
            _SPLIT_SYSTEM,
            f"Overall task: {task}\nStep that feels too big: {step}",
            500,
        )
        steps, _ = _parse_plan(raw)
        steps = [item for item in steps if item.lower() != step.lower()]
        if not steps:
            raise ValueError("The model did not make the step any smaller.")
        return steps, True, None
    except Exception as exc:
        logger.warning("AI step splitting failed; using fallback.", exc_info=True)
        return fallback_split(step), False, f"{type(exc).__name__}: {exc}"


def explain_step(
    step: str, task: str, api_key: str | None, model: str
) -> tuple[str, bool, str | None]:
    """
    Explain one step in simpler words.

    Returns (explanation, used_ai, error).
    """
    if not api_key:
        return (
            fallback_explain(step),
            False,
            "No Groq API key found. Check the [groq] section in .streamlit/secrets.toml.",
        )
    try:
        text = _ask(
            api_key,
            model,
            _EXPLAIN_SYSTEM,
            f"Overall task: {task}\nStep to explain: {step}",
            300,
        )
        return text, True, None
    except Exception as exc:
        logger.warning("AI explanation failed; using fallback.", exc_info=True)
        return fallback_explain(step), False, f"{type(exc).__name__}: {exc}"