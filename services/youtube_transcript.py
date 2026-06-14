"""
Fetch a YouTube video's subtitles via yt-dlp and flatten them to script text.

Used by the Gen-video tool: paste a YouTube URL → the spoken text drops straight
into the kịch bản editor so it can be split into scenes like any typed script.

Runs on the LOGIN node (which has internet). We shell out to the ``yt-dlp`` CLI
(or `python -m yt_dlp`) to download the subtitle track, ask it to
``--convert-subs srt``, then parse the .srt/.vtt locally into one flowing block.
Manual subtitles are preferred over auto-generated, Vietnamese over English.
Ported verbatim from the desktop gen-video project (app/core/youtube_transcript).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Languages to request, in priority order (yt-dlp picks whatever a video has).
_SUB_LANGS = "vi,vi-VN,en,en-US,en-GB,en-orig"

_INLINE_TAG = re.compile(r"<[^>]+>")                          # <00:00:00.000><c> tags
_SOUND_CUE = re.compile(r"[\[(][^\]\)]*[\])]|[♪♫]")          # [âm nhạc], (Applause), ♪


def _yt_dlp_cmd():
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    return [sys.executable, "-m", "yt_dlp"]


def yt_dlp_available():
    """True if yt-dlp can be invoked (CLI on PATH or importable module)."""
    if shutil.which("yt-dlp"):
        return True
    try:
        __import__("yt_dlp")
        return True
    except Exception:  # noqa: BLE001
        return False


def _sub_lang(path):
    parts = path.name.split(".")
    return parts[-2] if len(parts) >= 3 else ""


def _pick_sub_file(folder, langs):
    pref = [p.strip() for p in langs.split(",") if p.strip()]

    def score(path):
        lang = _sub_lang(path)
        rank = len(pref) + 1
        for i, p in enumerate(pref):
            base = p.split("-")[0]
            if lang == p or lang.startswith(base):
                rank = i
                break
        fmt = 0 if path.suffix.lower() == ".srt" else 1
        return (rank, fmt)

    files = list(folder.glob("*.srt")) + list(folder.glob("*.vtt"))
    if not files:
        return None
    files.sort(key=score)
    return files[0]


def _best_error(stderr):
    text = (stderr or "").strip()
    if not text:
        return ""
    errs = [ln.strip() for ln in text.splitlines()
            if ln.strip().upper().startswith("ERROR")]
    return (" ".join(errs) if errs else text)[-500:]


def _append_dedup(out, line):
    """Append while collapsing YouTube's rolling auto-captions (each cue repeats
    the previous line; merge only the new tail)."""
    if not line:
        return
    if not out:
        out.append(line)
        return
    prev = out[-1]
    if line == prev:
        return
    pw, lw = prev.split(), line.split()
    k = min(len(pw), len(lw))
    while k > 0 and pw[-k:] != lw[:k]:
        k -= 1
    if k > 0 and (k == len(pw) or k == len(lw)):
        tail = " ".join(lw[k:])
        out[-1] = (prev + " " + tail).strip() if tail else prev
        return
    out.append(line)


def _srt_to_text(raw):
    raw = raw.replace("﻿", "")
    out = []
    for block in re.split(r"\n\s*\n", raw):
        for line in block.splitlines():
            s = line.strip()
            if not s or s.isdigit() or "-->" in s:
                continue
            if s.upper().startswith("WEBVTT") or s.startswith(("Kind:", "Language:")):
                continue
            s = _INLINE_TAG.sub("", s)
            s = _SOUND_CUE.sub(" ", s)
            s = re.sub(r"\s+", " ", s).strip()
            _append_dedup(out, s)
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def fetch_transcript(url, langs=_SUB_LANGS, timeout=180):
    """Download a video's subtitles and return {text, title, lang, auto}.
    Raises RuntimeError (Vietnamese message) if yt-dlp is missing, the download
    fails, or the video has no usable subtitles."""
    url = (url or "").strip()
    if not url:
        raise RuntimeError("Chưa có link YouTube.")
    if not re.search(r"(youtube\.com|youtu\.be)/", url):
        raise RuntimeError("Link không phải YouTube.")

    base = _yt_dlp_cmd()
    with tempfile.TemporaryDirectory(prefix="ytsub_") as td:
        tmp = Path(td)
        cmd = base + [
            "--skip-download", "--no-playlist",
            "--write-subs", "--write-auto-subs",
            "--sub-langs", langs, "--convert-subs", "srt",
            "--write-info-json", "--retries", "5", "--retry-sleep", "3",
            "-o", str(tmp / "%(id)s.%(ext)s"), url,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=timeout)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Không tìm thấy yt-dlp trên login node. Cài: pip install -U yt-dlp"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("yt-dlp quá thời gian — thử lại hoặc video khác.") from exc

        title, info = "", {}
        info_files = sorted(tmp.glob("*.info.json"))
        if info_files:
            try:
                info = json.loads(info_files[0].read_text(
                    encoding="utf-8", errors="replace"))
                title = (info.get("title") or "").strip()
            except Exception:  # noqa: BLE001
                pass

        sub = _pick_sub_file(tmp, langs)
        if sub is None:
            err = _best_error(proc.stderr)
            if proc.returncode != 0 and err:
                raise RuntimeError(f"yt-dlp lỗi: {err}")
            raise RuntimeError(
                "Video này không có phụ đề tiếng Việt/Anh (kể cả tự động). "
                "Hãy thử video khác hoặc tự nhập kịch bản.")

        lang = _sub_lang(sub)
        auto = lang not in (info.get("subtitles") or {})
        text = _srt_to_text(sub.read_text(encoding="utf-8", errors="replace"))
        if not text:
            raise RuntimeError("Phụ đề tải về rỗng — không trích được lời thoại.")
        return {"text": text, "title": title, "lang": lang, "auto": auto}
