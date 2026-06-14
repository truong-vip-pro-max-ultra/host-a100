"""
Tools / Gen video — turn each narration chunk into a cinematic image prompt.

Two layers, in order of quality:

  1. **LLM rewrite via the API farm** (preferred): the host already runs an
     OpenAI-compatible LLM server (``serve_service`` — Qwen3-Coder). We call it
     INTERNALLY (login node → compute node, NOT through the Cloudflare tunnel, no
     API key needed — exactly like ``voice_pipeline`` talks to the GPU server) to
     rewrite a Vietnamese narration line into ONE concrete, depictable English
     scene. This both translates and "directs" the shot so scenes differ.
  2. **Rule-based fallback** (always works, offline): a port of the desktop
     gen-video ``prompt_generator`` — detect the script's dominant mood + a
     recurring subject anchor, then assemble ``{subject}, {anchor}, {mood},
     {style}, {base}``. If ``deep_translator`` is importable (login node has
     internet) the raw narration is translated VN→EN first; otherwise it's used
     as-is (SDXL is English-centric, so translation helps a lot).

Pure-CPU + stdlib (+ optional deep_translator); safe to unit-test without a GPU.
"""
from __future__ import annotations

import functools
import hashlib
import json
import re
import time
import http.client

try:
    from services import serve_service
except Exception:  # noqa: BLE001 — keep importable in a bare test env
    serve_service = None  # type: ignore


# --------------------------------------------------------------------------- #
# Look / style
# --------------------------------------------------------------------------- #
_BASE_STYLE = "cinematic still, dramatic lighting, highly detailed, sharp focus"

# One art "look" applied to every shot of a video so the whole thing feels like
# one piece. Key = the value sent by the form's "Phong cách ảnh" dropdown.
_STYLE_PRESETS = {
    "cinematic": "",  # _BASE_STYLE already covers the house cinematic look
    "realistic": "photorealistic, ultra realistic, lifelike, natural skin texture",
    "documentary": "documentary photography, photojournalism, candid, natural realistic light",
    "3d": "3D render, pixar style, octane render, soft global illumination, subsurface scattering",
    "anime": "anime style, cel shaded, studio anime key visual, clean lineart",
    "comic": "comic book art, bold ink outlines, halftone shading, dynamic paneling, vivid",
    "oil_painting": "oil painting, visible textured brushstrokes, classical fine art, painterly",
    "watercolor": "watercolor painting, soft washes, paper texture, delicate",
    "pencil_sketch": "pencil sketch, hand-drawn graphite drawing, fine hatching, "
                     "sketchbook, monochrome, grayscale, white paper background",
    "colored_pencil": "colored pencil drawing, hand-drawn, vibrant colored pencils, "
                      "soft layered strokes, sketchbook, white paper background",
    "ink_drawing": "pen and ink line art, clean bold outlines, hand-drawn illustration, "
                   "cross-hatching, black and white, white background",
    "cartoon_doodle": "cute cartoon illustration, flat vector style, clean bold black "
                      "outlines, simple rounded shapes, soft flat pastel colors, "
                      "friendly characters, minimalist explainer doodle, plain white background",
    "stick_figure": "hand-drawn stick figure doodle, thin black marker lines, round "
                    "heads with simple smiley faces, stick arms and legs, plain white "
                    "background, black and white line art, monochrome, no color, "
                    "minimalist, very simple",
}

# Drawing presets must NOT inherit the photoreal house look (it fights them).
_DRAWING_PRESETS = frozenset(
    {"pencil_sketch", "colored_pencil", "ink_drawing", "cartoon_doodle", "stick_figure"})
# Presets where the mood's colour/atmosphere tokens leak colour into a monochrome
# sketch or muddy the flat cartoon look — drop them (mood name still detected).
_NO_MOOD_PRESETS = frozenset(
    {"pencil_sketch", "ink_drawing", "cartoon_doodle", "stick_figure"})
# Playful presets that should NOT lock the recurring subject anchor — the flat
# look already keeps the video cohesive; the anchor only made shots look alike.
_NO_ANCHOR_PRESETS = frozenset({"cartoon_doodle", "stick_figure"})
# Presets whose look is so distinctive it must LEAD the prompt (CLIP weights the
# first tokens most). For these the style goes first + the subject stays short.
_STYLE_FIRST_PRESETS = frozenset({"stick_figure"})
_STYLE_FIRST_LEAD = {"stick_figure": "a simple black stick figure line drawing of"}

DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, watermark, signature, blurry, "
    "jpeg artifacts, deformed, ugly, duplicate, cropped"
)

# mood -> (keywords VN+EN, style tokens)
_MOODS = {
    "mystery": (
        ("bí ẩn", "huyền bí", "kỳ lạ", "tín hiệu", "biến mất",
         "mystery", "unknown", "signal", "strange", "vanish"),
        "dark mysterious atmosphere, fog, moody shadows, eerie glow, suspense"),
    "space": (
        ("vũ trụ", "ngân hà", "hành tinh", "thiên hà", "ngôi sao", "không gian",
         "space", "galaxy", "planet", "cosmos", "star", "nebula"),
        "deep space, nebula, stars, vast cosmic scale, sci-fi, glowing"),
    "ancient": (
        ("cổ đại", "ngày xưa", "lịch sử", "đế chế", "kim tự tháp", "văn minh",
         "ancient", "history", "empire", "ruins", "civilization", "pyramid"),
        "ancient ruins, weathered stone, golden hour, historical, majestic"),
    "nature": (
        ("rừng", "đại dương", "biển", "núi", "sông", "thiên nhiên", "động vật",
         "forest", "ocean", "mountain", "river", "nature", "wildlife"),
        "breathtaking landscape, natural light, lush, national geographic style"),
    "science": (
        ("khoa học", "nghiên cứu", "thí nghiệm", "công nghệ", "phòng lab",
         "science", "research", "experiment", "technology", "laboratory"),
        "scientific laboratory, futuristic technology, clean blue lighting, precise"),
    "war": (
        ("chiến tranh", "trận chiến", "quân đội", "vũ khí", "xung đột",
         "war", "battle", "army", "weapon", "conflict", "soldier"),
        "dramatic battlefield, smoke, intense, gritty, desaturated, tension"),
    "city": (
        ("thành phố", "đô thị", "đường phố", "tòa nhà", "đèn neon",
         "city", "urban", "street", "building", "skyline", "neon"),
        "moody cityscape, neon lights, rain, cinematic night, reflections"),
}
_DEFAULT_MOOD = ("neutral", "balanced cinematic composition, soft natural light")

_STOPWORDS = set(
    "và là của có không một những các đã sẽ được người khi đó này thì mà ra vào "
    "cho nên rất cũng còn về với từ trong trên dưới the a an of to and is was "
    "were are be in on at it he she they we you".split())


# --------------------------------------------------------------------------- #
# Mood / anchor (rule-based)
# --------------------------------------------------------------------------- #
def detect_mood(text):
    low = (text or "").lower()
    best, best_hits = _DEFAULT_MOOD, 0
    for name, (keywords, style) in _MOODS.items():
        hits = sum(1 for kw in keywords if kw in low)
        if hits > best_hits:
            best, best_hits = (name, style), hits
    return best


def dominant_mood(chunks):
    if len(chunks) <= 1:
        return detect_mood(chunks[0] if chunks else "")
    low = " ".join(chunks).lower()
    best, best_hits = _DEFAULT_MOOD, 0
    for name, (keywords, style) in _MOODS.items():
        hits = sum(low.count(kw) for kw in keywords)
        if hits > best_hits:
            best, best_hits = (name, style), hits
    return best


def _keywords(text, limit=6):
    out = []
    for w in re.findall(r"[A-Za-zÀ-Ỹà-ỹ0-9]+", text or ""):
        wl = w.lower()
        if wl in _STOPWORDS or len(wl) < 3:
            continue
        if w[0].isupper() or len(wl) > 5:
            if w not in out:
                out.append(w)
        if len(out) >= limit:
            break
    return out


def derive_anchor(chunks, limit=4):
    """The story's recurring subject (counted across chunks) → prepended to every
    prompt so the cast/world stays on-model."""
    from collections import Counter
    cnt = Counter()
    for t in chunks:
        for w in set(_keywords(t, limit=12)):
            cnt[w] += 1
    recurring = [w for w, c in cnt.most_common(20) if c >= 2]
    chosen = recurring[:limit] or [w for w, _ in cnt.most_common(limit)]
    return ", ".join(chosen)


# --------------------------------------------------------------------------- #
# Translation (best-effort, login node has internet)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=512)
def translate_to_en(text):
    text = (text or "").strip()
    if not text:
        return text
    try:
        from deep_translator import GoogleTranslator
        out = GoogleTranslator(source="auto", target="en").translate(text[:1500])
        return out or text
    except Exception:  # noqa: BLE001 — offline / not installed → keep original
        return text


# --------------------------------------------------------------------------- #
# Rule-based prompt assembly
# --------------------------------------------------------------------------- #
def build_prompt(subject, anchor, mood_style, style_preset, negative):
    """Assemble {subject, anchor, mood, style, base} with per-preset handling
    (drawings drop the photoreal base + mood colours; style-first presets like
    stick figures lead with the style and bind it to the subject)."""
    subject = re.sub(r"\s+", " ", (subject or "")).strip().strip('"“”')
    style_first = style_preset in _STYLE_FIRST_PRESETS
    cap = 110 if style_first else 200   # style-first must let the style win
    if len(subject) > cap:
        subject = subject[:cap].rsplit(" ", 1)[0]
    if not subject:
        subject = "cinematic establishing shot"

    preset = _STYLE_PRESETS.get(style_preset, "")
    base = "" if style_preset in _DRAWING_PRESETS else _BASE_STYLE
    mood_part = "" if style_preset in _NO_MOOD_PRESETS else (mood_style or "")
    anchor_part = "" if style_preset in _NO_ANCHOR_PRESETS else (anchor or "").strip()

    if style_first and preset:
        lead = _STYLE_FIRST_LEAD.get(style_preset, "")
        if lead and subject:
            subject = f"{lead} {subject[0].lower()}{subject[1:]}"
        parts = [subject, preset, anchor_part, mood_part, base]
    else:
        parts = [subject, anchor_part, mood_part, preset, base]
    positive = ", ".join(p for p in parts if p)
    positive = re.sub(r"\s*,\s*,\s*", ", ", positive).strip(" ,")
    return positive, (negative or DEFAULT_NEGATIVE)


# --------------------------------------------------------------------------- #
# LLM rewrite via the API farm (OpenAI-compatible /v1/chat/completions)
# --------------------------------------------------------------------------- #
_LLM_INSTRUCTION = (
    "You are an art director writing prompts for a text-to-image model. "
    "Turn the narration line (which may be Vietnamese) into ONE vivid, CONCRETE "
    "visual scene in English.\n"
    "Rules:\n"
    "- Describe a specific, depictable moment: main subject, what it is doing, "
    "key objects, setting. Turn any abstract idea into a concrete visual metaphor.\n"
    "- 15-35 words, comma-separated visual phrases (not a full sentence).\n"
    "- Vary the framing (close-up, wide shot, over-the-shoulder, top-down…).\n"
    "- Do NOT mention any art style, medium or colour palette.\n"
    "- Do NOT put any words, text, letters or captions in the image.\n"
    "- Output ONLY the prompt: no quotes, no 'Prompt:', no explanation."
)


def llm_available():
    """True when an API-farm LLM server is ready to rewrite prompts."""
    if serve_service is None:
        return False
    try:
        return serve_service.resolve_endpoint() is not None
    except Exception:  # noqa: BLE001
        return False


_LLM_BATCH_INSTRUCTION = (
    "You are an art director writing prompts for a text-to-image model. You will "
    "receive a numbered list of narration lines (which may be Vietnamese). For "
    "EACH line, write ONE vivid, CONCRETE visual scene in English.\n"
    "Rules per scene:\n"
    "- A specific, depictable moment: main subject, action, key objects, setting; "
    "turn abstract ideas into a concrete visual metaphor.\n"
    "- 15-35 words, comma-separated visual phrases (not a full sentence).\n"
    "- Vary the framing across scenes (close-up, wide, over-the-shoulder, top-down…).\n"
    "- Do NOT mention any art style, medium or colour palette; no text/letters in the image.\n"
    "Output ONLY a JSON array of exactly N strings, in the SAME order as the input, "
    "nothing else (no keys, no markdown fences, no commentary)."
)


def _chat(messages, max_tokens, timeout):
    """One internal call to the API-farm LLM. Returns the content string or ''."""
    if serve_service is None:
        return ""
    try:
        ep = serve_service.resolve_endpoint()
    except Exception:  # noqa: BLE001
        return ""
    if not ep:
        return ""
    host, port, served_name = ep
    body = json.dumps({
        "model": served_name or "default",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "stream": False,
        "stop": ["<|im_end|>"],
    }).encode("utf-8")
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        if resp.status != 200:
            return ""
        return (json.loads(raw or b"{}")["choices"][0]["message"]["content"] or "").strip()
    except Exception:  # noqa: BLE001
        return ""


_NUMBERED_RE = re.compile(r"^\s*\[?(\d{1,3})\]?\s*[.):\-]\s*(.+?)\s*$")


def _parse_prompt_list(out, n):
    """Best-effort extract N rewritten prompts from the LLM reply. Tries a JSON
    array first, then a numbered list (Qwen often ignores 'JSON' and numbers the
    lines). Returns a list of length n ("" for any scene it couldn't map), or []
    if nothing usable was found. Partial is fine — the caller translates the gaps."""
    s = (out or "").strip()
    # 1) JSON array (tolerate a ```json fence and surrounding prose).
    a, b = s.find("["), s.rfind("]")
    if 0 <= a < b:
        try:
            arr = json.loads(s[a:b + 1])
            if isinstance(arr, list) and arr:
                vals = [(str(x).strip().strip('"“”')[:300] if x else "") for x in arr]
                if len(vals) == n:
                    return vals
                # Wrong count → keep going; the numbered parser may align better.
        except Exception:  # noqa: BLE001
            pass
    # 2) Numbered list: map "3. <prompt>" → index 2, so missing/extra lines don't
    #    misalign the rest.
    result = [""] * n
    hits = 0
    for line in s.splitlines():
        m = _NUMBERED_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        val = m.group(2).strip().strip('"“”').strip()
        if 0 <= idx < n and val and not result[idx]:
            result[idx] = val[:300]
            hits += 1
    return result if hits else []


def llm_visual_prompts(texts, timeout=180, simple=False, on_log=None):
    """Rewrite ALL narration lines in ONE LLM call (the API-farm server runs
    --parallel 1, so one batched request is far faster than N parallel ones that
    just queue server-side). Returns a list aligned with ``texts`` (each item is
    the rewritten prompt or "" → the caller translates that scene), or [] on total
    failure → the caller falls back to rule-based + translation. ``simple`` keeps
    each scene to the bare subject + one action (stick-figure / minimal styles)."""
    texts = [(t or "").strip() for t in texts]
    if not texts or serve_service is None:
        return []
    instruction = _LLM_BATCH_INSTRUCTION
    if simple:
        instruction += ("\nIMPORTANT: keep EACH scene VERY simple — name only the "
                        "main subject(s) and ONE clear action. No detailed scenery, "
                        "backgrounds, lighting or extra objects. Max ~12 words each.")
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    out = _chat(
        [{"role": "system", "content": instruction},
         {"role": "user", "content": f"{len(texts)} narration lines:\n{numbered}"}],
        max_tokens=min(2400, 110 * len(texts) + 200), timeout=timeout)
    if not out:
        if on_log:
            on_log("    (LLM không phản hồi hoặc rỗng)")
        return []
    parsed = _parse_prompt_list(out, len(texts))
    if not parsed and on_log:
        on_log(f"    (LLM trả về không đọc được, đầu ra: {out[:200]!r})")
    return parsed


def llm_visual_prompt(text, seed=-1, timeout=60):
    """Rewrite one narration line into a concrete English scene via the API farm.
    Returns "" on any failure so the caller falls back to the rule-based path."""
    text = (text or "").strip()
    if not text or serve_service is None:
        return ""
    try:
        ep = serve_service.resolve_endpoint()
    except Exception:  # noqa: BLE001
        return ""
    if not ep:
        return ""
    host, port, served_name = ep
    body = json.dumps({
        "model": served_name or "default",
        "messages": [
            {"role": "system", "content": _LLM_INSTRUCTION},
            {"role": "user", "content": f'Narration: "{text}"\nPrompt:'},
        ],
        "temperature": 0.8,
        "max_tokens": 120,
        "stream": False,
        "stop": ["<|im_end|>"],
        **({"seed": int(seed)} if seed is not None and seed >= 0 else {}),
    }).encode("utf-8")
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        if resp.status != 200:
            return ""
        data = json.loads(raw or b"{}")
        out = (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:  # noqa: BLE001
        return ""
    for line in out.splitlines():
        line = line.strip().strip('"“”').strip()
        if line.lower().startswith("prompt:"):
            line = line[7:].strip()
        if line:
            return line[:300]
    return ""


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
# How many scenes per LLM prompt-rewrite call. One call must comfortably fit its
# output budget (~90 tok/scene) and stay reliable; a giant 100+-scene request
# overruns it and the server returns empty. ~24 is a safe, fast group.
_LLM_PROMPT_BATCH = 24


def _stable_seed(text):
    return int(hashlib.sha256((text or "").encode()).hexdigest(), 16) % (2 ** 31)


def build_prompts(chunks, style_preset="cinematic", negative="", use_llm=True,
                  anchor="", on_log=None, workers=4):
    """Return a list of dicts: {prompt, negative, seed} aligned with ``chunks``.

    Derives the anchor + dominant mood once, then for each chunk writes a prompt:
    the LLM-rewritten concrete visual leads when available (parallelised since the
    calls are network-bound), otherwise the translated narration does.
    """
    def _log(m):
        if on_log:
            on_log(m)

    chunks = list(chunks or [])
    if not chunks:
        return []
    negative = negative or DEFAULT_NEGATIVE
    anchor = (anchor or "").strip() or (derive_anchor(chunks) if len(chunks) > 1 else "")
    _mood_name, mood_style = dominant_mood(chunks)
    seeds = [_stable_seed(t) for t in chunks]

    use_llm = bool(use_llm) and llm_available()
    visuals = [""] * len(chunks)
    if use_llm:
        # Rewrite in SUB-BATCHES (the server is --parallel 1, and one giant request
        # for 100+ scenes overruns the output budget → empty reply). Each group is
        # one call; a failed/partial group just leaves those scenes "" → translated.
        simple = (style_preset in _STYLE_FIRST_PRESETS
                  or style_preset in _DRAWING_PRESETS)
        ngrp = (len(chunks) + _LLM_PROMPT_BATCH - 1) // _LLM_PROMPT_BATCH
        _log(f"  ➤ viết prompt ảnh bằng LLM API farm ({len(chunks)} cảnh, "
             f"{ngrp} lượt × ≤{_LLM_PROMPT_BATCH})…")
        t0 = time.time()
        for s in range(0, len(chunks), _LLM_PROMPT_BATCH):
            sub = chunks[s:s + _LLM_PROMPT_BATCH]
            res = llm_visual_prompts(sub, simple=simple, on_log=on_log)
            for k, v in enumerate(res or []):
                visuals[s + k] = v
            n_ok = sum(1 for v in visuals[s:s + len(sub)] if v)
            _log(f"    cảnh {s + 1}–{s + len(sub)}: LLM viết {n_ok}/{len(sub)} "
                 "(còn lại tự dịch)")
        total_ok = sum(1 for v in visuals if v)
        if total_ok:
            _log(f"  ✓ LLM xong {total_ok}/{len(chunks)} cảnh "
                 f"({time.time() - t0:.1f}s); phần còn lại dùng prompt theo luật + dịch.")
        else:
            _log(f"  ! LLM không trả về hợp lệ ({time.time() - t0:.1f}s) — "
                 "dùng prompt theo luật + dịch cho tất cả.")
    else:
        _log("  ! Không có LLM (API farm chưa chạy) — dùng prompt theo luật + dịch.")

    out = []
    for i, t in enumerate(chunks):
        subject = visuals[i] or translate_to_en(t)
        pos, neg = build_prompt(subject, anchor, mood_style, style_preset, negative)
        out.append({"prompt": pos, "negative": neg, "seed": seeds[i]})
    return out
