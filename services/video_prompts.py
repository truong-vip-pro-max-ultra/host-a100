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

_STYLE_PRESETS = {
    "cinematic": "",
    "realistic": "photorealistic, ultra realistic, lifelike, natural skin texture",
    "documentary": "documentary photography, photojournalism, candid, natural realistic light",
    "3d": "3D render, pixar style, octane render, soft global illumination, subsurface scattering",
    "anime": "anime style, cel shaded, studio anime key visual, clean lineart",
    "oil_painting": "oil painting, visible textured brushstrokes, classical fine art, painterly",
    "watercolor": "watercolor painting, soft washes, paper texture, delicate",
}

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
    subject = re.sub(r"\s+", " ", (subject or "")).strip().strip('"“”')
    if len(subject) > 200:
        subject = subject[:200].rsplit(" ", 1)[0]
    if not subject:
        subject = "cinematic establishing shot"
    preset = _STYLE_PRESETS.get(style_preset, "")
    parts = [subject, (anchor or "").strip(), mood_style or "", preset, _BASE_STYLE]
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


def llm_visual_prompts(texts, timeout=180):
    """Rewrite ALL narration lines in ONE LLM call (the API-farm server runs
    --parallel 1, so one batched request is far faster than N parallel ones that
    just queue server-side). Returns a list aligned with ``texts`` (each item is
    the rewritten prompt or "" if unusable), or [] on total failure → the caller
    falls back to rule-based + translation."""
    texts = [(t or "").strip() for t in texts]
    if not texts or serve_service is None:
        return []
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    out = _chat(
        [{"role": "system", "content": _LLM_BATCH_INSTRUCTION},
         {"role": "user", "content": f"{len(texts)} narration lines:\n{numbered}"}],
        max_tokens=min(1600, 90 * len(texts) + 200), timeout=timeout)
    if not out:
        return []
    # Strip an accidental ```json fence, then parse the JSON array.
    s = out.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("["):] if "[" in s else s
    start, end = s.find("["), s.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        arr = json.loads(s[start:end + 1])
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(arr, list) or len(arr) != len(texts):
        return []
    return [(str(x).strip().strip('"“”')[:300] if x else "") for x in arr]


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
        # ONE batched request (the LLM server is --parallel 1, so N parallel calls
        # just queue → slow). Falls back to rule-based+translate on any failure.
        _log(f"  ➤ viết prompt ảnh bằng LLM API farm (1 lượt, {len(chunks)} cảnh)…")
        t0 = time.time()
        got = llm_visual_prompts(chunks)
        if got:
            visuals = got
            _log(f"  ✓ LLM xong {sum(1 for v in visuals if v)}/{len(chunks)} cảnh "
                 f"({time.time() - t0:.1f}s).")
        else:
            _log(f"  ! LLM không trả về hợp lệ ({time.time() - t0:.1f}s) — "
                 "dùng prompt theo luật + dịch.")
    else:
        _log("  ! Không có LLM (API farm chưa chạy) — dùng prompt theo luật + dịch.")

    out = []
    for i, t in enumerate(chunks):
        subject = visuals[i] or translate_to_en(t)
        pos, neg = build_prompt(subject, anchor, mood_style, style_preset, negative)
        out.append({"prompt": pos, "negative": neg, "seed": seeds[i]})
    return out
