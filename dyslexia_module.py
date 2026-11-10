"""
dyslexia_module.py - Text simplification and the reading overlay for PUGU.

* simplify_text()      : lexical simplification through the Groq API (free tier).
* build_overlay_html() : self-contained HTML/JS block (one <span> per sentence)
                         with Web Speech API read-aloud, "Smart Pacing" and a
                         "Reading Ruler" highlight. This module has no Streamlit
                         dependency; embed the HTML with components.html().
"""

from __future__ import annotations

import html
import logging
import re

from groq import Groq

logger = logging.getLogger(__name__)

MAX_INPUT_CHARS = 5000

FONT_STACKS = {
    "OpenDyslexic": "'OpenDyslexic', 'Comic Sans MS', Verdana, sans-serif",
    "Arial": "Arial, Helvetica, sans-serif",
}

BACKGROUND_HEX = {
    "White": "#FFFFFF",
    "Soft Yellow": "#FFF9C4",
    "Pale Blue": "#E3F2FD",
    "Mint Green": "#E8F5E9",
    "Peach": "#FFE5D0",
}

_SYSTEM_PROMPT = (
    "You rewrite text so it is easier to read for people with dyslexia. Rules: "
    "1) Replace uncommon, long or complex words with simple everyday words. "
    "2) Keep sentences short, ideally under 15 words; split long sentences. "
    "3) Keep the original meaning, facts, names and numbers. Do not add or remove information. "
    "4) Use plain text only: no markdown, no bullet points, no headings, no commentary. "
    "5) Keep paragraph breaks as blank lines. "
    "6) End every sentence with a period, exclamation mark or question mark. "
    "Output ONLY the rewritten text, with no introduction such as 'Here is the text'."
)


def _strip_preamble(text: str) -> str:
    """Remove a leading 'Here is the rewritten text:' line that smaller models add."""
    return re.sub(r"^\s*here(?:'s| is)[^\n]*:\s*\n+", "", text, flags=re.IGNORECASE).strip()


def _request_options(model: str, normal_tokens: int, reasoning_tokens: int) -> dict:
    """
    Model-specific request settings.

    "gpt-oss" models think before answering, and that hidden thinking uses up
    the token budget. They get a bigger budget and a request for low reasoning
    effort; other models use the normal budget.
    """
    if "gpt-oss" in model.lower():
        return {"max_tokens": reasoning_tokens, "extra_body": {"reasoning_effort": "low"}}
    return {"max_tokens": normal_tokens}


def simplify_text_ai(text: str, api_key: str, model: str) -> str:
    """Simplify `text` with a Groq-hosted model. Raises on any failure or empty reply."""
    client = Groq(api_key=api_key, timeout=45.0, max_retries=1)
    completion = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"<text>\n{text}\n</text>"},
        ],
        **_request_options(model, normal_tokens=1800, reasoning_tokens=3500),
    )
    simplified = _strip_preamble(completion.choices[0].message.content or "")
    if not simplified:
        raise ValueError(
            "The model returned an empty reply (its token budget may have been "
            "used up by reasoning)."
        )
    return simplified


def simplify_text(text: str, api_key: str | None, model: str) -> tuple[str, bool]:
    """
    Simplify text when possible.

    Returns (text, used_ai). If the API key is missing or the call fails (for
    example the free limit is reached), the original text is returned with
    used_ai=False.
    """
    if api_key:
        try:
            return simplify_text_ai(text, api_key, model), True
        except Exception:
            logger.warning("AI text simplification failed; using original.", exc_info=True)
    return text, False

def split_paragraphs(text: str) -> list[list[str]]:
    """Split text into paragraphs, each a list of sentences."""
    paragraphs: list[list[str]] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        flat = re.sub(r"\s+", " ", block).strip()
        if not flat:
            continue
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if s.strip()]
        if sentences:
            paragraphs.append(sentences)
    return paragraphs


def estimate_height(text: str, tts_enabled: bool) -> int:
    """Rough pixel height for the embedded overlay (it scrolls if longer)."""
    lines = max(1, len(text) // 55 + text.count("\n\n") + 1)
    height = 60 + lines * 46 + (90 if tts_enabled else 0)
    return max(260, min(height, 760))


_CONTROLS_HTML = """
<div class="toolbar" role="toolbar" aria-label="Read aloud controls">
  <button id="playBtn" type="button">▶ Read aloud</button>
  <button id="pauseBtn" type="button" disabled>⏸ Pause</button>
  <button id="stopBtn" type="button" disabled>⏹ Stop</button>
  <label class="rate">Speed
    <input id="rate" type="range" min="0.6" max="1.4" step="0.1" value="0.9">
    <span id="rateValue">0.9×</span>
  </label>
  <span id="status" role="status">Ready. You can also click any sentence to start there.</span>
</div>
"""

# Raw string: the JavaScript contains backslashes that Python must not touch.
_OVERLAY_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  @font-face {
    font-family: 'OpenDyslexic';
    font-style: normal;
    font-weight: 400;
    font-display: swap;
    src: local('OpenDyslexic'),
         url('https://cdn.jsdelivr.net/fontsource/fonts/opendyslexic@5.3.0/latin-400-normal.woff2') format('woff2'),
         url('https://cdnjs.cloudflare.com/ajax/libs/fontsource-opendyslexic/5.3.0/files/opendyslexic-latin-400-normal.woff2') format('woff2');
  }
  html, body { margin: 0; padding: 0; background: __BG__; color: #1B1B1B; }
  .page { padding: 8px 20px 24px; }
  .toolbar {
    position: sticky; top: 0; z-index: 5; display: flex; flex-wrap: wrap;
    align-items: center; gap: 10px; padding: 10px 0; background: __BG__;
    font-family: Arial, Helvetica, sans-serif; font-size: 0.95rem;
    border-bottom: 1px solid rgba(0, 0, 0, 0.15);
  }
  .toolbar button {
    padding: 10px 16px; font-size: 1rem; border: 0; border-radius: 8px;
    background: #2B5BD7; color: #FFFFFF; cursor: pointer;
  }
  .toolbar button:disabled { background: #9AA5B8; cursor: not-allowed; }
  .toolbar button:focus-visible, .toolbar input:focus-visible { outline: 3px solid #1B1B1B; outline-offset: 2px; }
  .rate { display: flex; align-items: center; gap: 6px; }
  #status { flex-basis: 100%; font-size: 0.9rem; }
  #reader {
    font-family: __FONT_STACK__; font-size: 1.35rem; line-height: 2.1;
    letter-spacing: 0.05em; word-spacing: 0.16em; max-width: 72ch; padding-top: 12px;
  }
  #reader p { margin: 0 0 1.1em; }
  .sentence {
    transition: opacity 0.2s ease, background-color 0.2s ease;
    border-radius: 4px; cursor: pointer;
    -webkit-box-decoration-break: clone; box-decoration-break: clone;
  }
  .sentence.unread { opacity: 0.25; }
  .sentence.active { opacity: 1; background: #FFD54F; box-shadow: 0 0 0 4px #FFD54F; }
</style>
</head>
<body>
<div class="page">
__CONTROLS__
<main id="reader">__CONTENT__</main>
</div>
<script>
(function () {
  var TTS_ENABLED = __TTS_ENABLED__;
  if (!TTS_ENABLED) { return; }

  var PAUSE_MS = 800;                                   // Smart Pacing pause
  var END_PUNCT = /[.!?]["'\u201D\u2019)\]]*\s*$/;      // sentence ends with . ! ?

  var sentences = Array.prototype.slice.call(document.querySelectorAll('.sentence'));
  var playBtn = document.getElementById('playBtn');
  var pauseBtn = document.getElementById('pauseBtn');
  var stopBtn = document.getElementById('stopBtn');
  var rateInput = document.getElementById('rate');
  var rateValue = document.getElementById('rateValue');
  var statusEl = document.getElementById('status');

  if (!('speechSynthesis' in window) || typeof SpeechSynthesisUtterance === 'undefined') {
    statusEl.textContent = 'Read-aloud is not supported in this browser. Try Chrome, Edge or Safari.';
    playBtn.disabled = true;
    return;
  }

  var synth = window.speechSynthesis;
  var state = 'idle';          // 'idle' | 'playing' | 'paused'
  var index = 0;               // next sentence to speak (or current when paused)
  var session = 0;             // bumped on every start/pause/stop to ignore stale callbacks
  var timer = null;
  var currentUtterance = null; // kept in a variable so the browser does not garbage-collect it

  // Reading Ruler: highlight the current sentence, dim every unread one.
  function setHighlight(active) {
    sentences.forEach(function (el, k) {
      el.classList.remove('active', 'unread');
      if (active < 0) { return; }
      if (k === active) { el.classList.add('active'); }
      else if (k > active) { el.classList.add('unread'); }
    });
  }

  function updateUi() {
    playBtn.textContent = state === 'playing' ? '▶ Restart' : '▶ Read aloud';
    pauseBtn.textContent = state === 'paused' ? '⏯ Resume' : '⏸ Pause';
    pauseBtn.disabled = state === 'idle';
    stopBtn.disabled = state === 'idle';
    if (state === 'idle') { statusEl.textContent = 'Ready. You can also click any sentence to start there.'; }
    if (state === 'paused') { statusEl.textContent = 'Paused.'; }
  }

  function keepInView(el) {
    var rect = el.getBoundingClientRect();
    if (rect.top < 90 || rect.bottom > window.innerHeight - 20) {
      window.scrollTo({ top: window.scrollY + rect.top - window.innerHeight / 3, behavior: 'smooth' });
    }
  }

  function finish() {
    state = 'idle';
    index = 0;
    setHighlight(-1);
    updateUi();
    statusEl.textContent = 'Finished reading ✓';
  }

  function stopAll() {
    session += 1;
    clearTimeout(timer);
    synth.cancel();
    state = 'idle';
    index = 0;
    setHighlight(-1);
    updateUi();
  }

  function speak(i, token) {
    if (token !== session) { return; }
    if (i >= sentences.length) { finish(); return; }

    index = i;
    setHighlight(i);
    keepInView(sentences[i]);
    statusEl.textContent = 'Reading sentence ' + (i + 1) + ' of ' + sentences.length;

    var text = sentences[i].textContent.trim();
    var utter = new SpeechSynthesisUtterance(text);
    utter.rate = parseFloat(rateInput.value);

    utter.onend = function () {
      if (token !== session) { return; }
      index = i + 1;
      // Smart Pacing: 800 ms breathing room after . ! ? to protect working memory.
      var delay = END_PUNCT.test(text) ? PAUSE_MS : 0;
      timer = setTimeout(function () { speak(i + 1, token); }, delay);
    };
    utter.onerror = function (ev) {
      if (token !== session) { return; }
      if (ev.error === 'canceled' || ev.error === 'interrupted') { return; }
      stopAll();
      statusEl.textContent = 'Read-aloud problem: ' + ev.error;
    };

    currentUtterance = utter;
    synth.speak(utter);
  }

  function begin(i) {
    session += 1;
    var token = session;
    clearTimeout(timer);
    synth.cancel();
    state = 'playing';
    updateUi();
    // Short delay: some browsers drop speech queued right after cancel().
    timer = setTimeout(function () { speak(i, token); }, 60);
  }

  playBtn.addEventListener('click', function () { begin(0); });
  stopBtn.addEventListener('click', stopAll);
  pauseBtn.addEventListener('click', function () {
    if (state === 'playing') {
      session += 1;
      clearTimeout(timer);
      synth.cancel();
      state = 'paused';
      updateUi();
    } else if (state === 'paused') {
      begin(index);
    }
  });
  rateInput.addEventListener('input', function () {
    rateValue.textContent = parseFloat(rateInput.value).toFixed(1) + '×';
  });
  sentences.forEach(function (el, k) {
    el.addEventListener('click', function () { begin(k); });
  });
  window.addEventListener('pagehide', function () { synth.cancel(); });
})();
</script>
</body>
</html>
"""


def build_overlay_html(
    text: str, font_choice: str, bg_color: str, tts_enabled: bool
) -> str:
    """
    Build the HTML/JS reading overlay for `text`.

    Every sentence is wrapped in its own <span class="sentence"> so the script
    can highlight the sentence being read and dim the unread ones.
    """
    paragraph_blocks: list[str] = []
    for sentences in split_paragraphs(text):
        spans = [
            '<span class="sentence">' + html.escape(sentence) + "</span>"
            for sentence in sentences
        ]
        paragraph_blocks.append("<p>" + " ".join(spans) + "</p>")
    content = "\n".join(paragraph_blocks)

    page = _OVERLAY_TEMPLATE
    page = page.replace("__FONT_STACK__", FONT_STACKS.get(font_choice, FONT_STACKS["Arial"]))
    page = page.replace("__BG__", BACKGROUND_HEX.get(bg_color, "#FFFFFF"))
    page = page.replace("__TTS_ENABLED__", "true" if tts_enabled else "false")
    page = page.replace("__CONTROLS__", _CONTROLS_HTML if tts_enabled else "")
    # Content goes in last so user text can never be mistaken for a template token.
    return page.replace("__CONTENT__", content)