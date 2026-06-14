"""
Tools / Gen video — assemble per-scene (image + Ken Burns + narration) into MP4.

Runs entirely on the LOGIN node (no GPU) with ffmpeg, ported from the desktop
gen-video project (``app/ffmpeg/video_renderer.py`` + ``app/subtitle/
subtitle_engine.py``):

  1. For each scene render an intermediate clip: still → Ken Burns ``zoompan``
     motion + fade in/out (video-only). Independent clips render in PARALLEL
     (the filter chain is CPU-bound and only lightly threaded, so a many-core
     login node does ~N at once).
  2. Concat the clips (concat demuxer).
  3. One final pass burns the subtitles (.ass, PlayRes-pinned), rebuilds the
     audio GAPLESSLY from the raw per-scene wavs (each forced to its scene
     duration so it stays locked to the picture — no AAC priming gap / stutter),
     loops + ducks optional background music, and loudness-normalises.

A "scene" here is a plain dict: {text, image_path, audio_path, duration}.
"""
from __future__ import annotations

import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config

# Micro fade so each scene's audio reaches a zero crossing at every join (no pop).
_DECLICK_FADE = 0.015
# Gentle edge silence trim (well below speech) so chunks butt cleanly.
_NARRATION_SILENCE_DB = "-50dB"
_NARRATION_EDGE_KEEP = 0.05
# Login node is CPU-only → use every core for codec AND the (large) filter graph.
_FFMPEG_THREADS = ["-threads", "0", "-filter_complex_threads", "0"]

# Subtitle defaults (ASS). Fontsize(px) = font_size * height / _FONT_DIVISOR.
_FONT_DIVISOR = 380.0
DEFAULT_SUBTITLE = {
    "enabled": True,
    "font": "Arial",
    "font_size": 22,
    "primary_color": "&H00FFFFFF",   # white (ASS BGR)
    "outline_color": "&H00000000",   # black
    "outline": 3,
    "shadow": 1,
    "margin_v": 60,
    "bold": True,
    "max_chars_per_line": 42,
}

DEFAULT_RENDER = {
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "video_bitrate": "8M",
    "transition_duration": 0.6,
    "ken_burns": True,
    "ken_burns_intensity": 0.12,
    "music_volume": 0.12,
    "music_path": "",
    "crf": 20,
    "clip_workers": 0,   # 0 = auto from CPU count
    # x264 presets: the per-scene clips are intermediate (re-encoded in the final
    # pass), so a fast preset there is free quality-wise; the final pass governs
    # the output. Both default fast since this content (stills + slow zoom) stays
    # clean even at veryfast — the user wants the quickest render.
    "clip_preset": "veryfast",
    "final_preset": "veryfast",
}


def _run_ffmpeg(args, cwd=None, timeout=7200):
    cmd = [config.ffmpeg_path(), "-y", "-hide_banner", "-nostdin", *args]
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    return proc.returncode, (proc.stdout or "")


def _make_color_image(path, w, h, color="#101018"):
    """A flat fallback still when a scene image is missing."""
    rc, log = _run_ffmpeg(["-f", "lavfi", "-i", f"color=c={color}:s={w}x{h}",
                           "-frames:v", "1", str(path)])
    if rc != 0:
        raise RuntimeError(f"ffmpeg không tạo được ảnh nền: {(log or '')[-200:]}")
    return path


# --------------------------------------------------------------------------- #
# Per-scene clip (Ken Burns)
# --------------------------------------------------------------------------- #
def _ken_burns_filter(index, frames, w, h, intensity, fps):
    maxz = 1.0 + max(0.02, intensity)
    uw, uh = w * 2, h * 2
    if index % 2 == 0:
        z = f"1+(on/{frames})*{intensity:.4f}"
    else:
        z = f"{maxz:.4f}-(on/{frames})*{intensity:.4f}"
    x = "iw/2-(iw/zoom/2)"
    y = "ih/2-(ih/zoom/2)"
    return (
        f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={w}:{h},scale={uw}:{uh}:flags=lanczos,"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={w}x{h}:fps={fps},"
        f"unsharp=5:5:0.8:5:5:0.0,setsar=1"
    )


def _render_scene_clip(scene, index, out, r, work_dir, sub_name=None):
    """Render one scene's clip: still → Ken Burns + fade, and BURN that scene's
    subtitle here (``sub_name`` = an .ass file in work_dir timed 0..dur). Burning
    per-clip parallelises the (single-threaded) libass work across all clip
    workers, so the final pass can stream-COPY the video instead of re-encoding
    the whole movie just to burn subs — that was the slow 'Ghép hoàn thiện' step."""
    fps = int(r["fps"])
    dur = max(0.8, float(scene.get("duration") or 0.0))
    frames = max(1, int(round(dur * fps)))

    img = scene.get("image_path") or ""
    if not img or not Path(img).exists():
        img = str(Path(work_dir) / f"_fallback_{index}.png")
        _make_color_image(img, r["width"], r["height"])

    if r.get("ken_burns", True):
        vf = _ken_burns_filter(index, frames, r["width"], r["height"],
                               float(r["ken_burns_intensity"]), fps)
    else:
        vf = (f"scale={r['width']}:{r['height']}:force_original_aspect_ratio="
              f"increase:flags=lanczos,crop={r['width']}:{r['height']},"
              f"unsharp=5:5:0.8:5:5:0.0,setsar=1")

    fd = min(float(r["transition_duration"]), dur / 3.0)
    vf += (f",fade=t=in:st=0:d={fd:.3f},"
           f"fade=t=out:st={max(0, dur - fd):.3f}:d={fd:.3f}")
    # Burn this scene's subtitle (relative name + cwd=work_dir avoids the
    # subtitles= path-escaping pitfalls). Runs inside the existing per-clip encode.
    if sub_name:
        vf += f",subtitles={sub_name}"
    vf += ",format=yuv420p"

    args = ["-i", img, "-filter_complex", f"[0:v]{vf}[v]", "-map", "[v]",
            "-t", f"{dur:.3f}", "-r", str(fps),
            "-c:v", "libx264", "-preset", r.get("clip_preset", "veryfast"),
            "-crf", str(r["crf"]),
            "-pix_fmt", "yuv420p", "-an", str(out)]
    rc, log = _run_ffmpeg(args, cwd=work_dir)
    if rc != 0 or not Path(out).exists():
        raise RuntimeError(f"ffmpeg dựng cảnh {index + 1} lỗi: {(log or '')[-300:]}")
    return out


# --------------------------------------------------------------------------- #
# Audio (gapless, locked to each scene's duration)
# --------------------------------------------------------------------------- #
def _rel(path, base):
    try:
        return os.path.relpath(str(path), str(base))
    except ValueError:
        return str(path)


def _audio_inputs(scenes, base):
    inputs = []
    for s in scenes:
        dur = max(0.8, float(s.get("duration") or 0.0))
        a = s.get("audio_path") or ""
        if a and Path(a).exists():
            inputs += ["-i", _rel(a, base)]
        else:
            inputs += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i",
                       "anullsrc=channel_layout=stereo:sample_rate=48000"]
    return inputs


def _voice_concat_filters(scenes, start_idx):
    lead_trim = (f"silenceremove=start_periods=1:start_threshold="
                 f"{_NARRATION_SILENCE_DB}:start_silence={_NARRATION_EDGE_KEEP}:"
                 f"detection=peak,")
    lines, labels = [], []
    for k, s in enumerate(scenes):
        idx = start_idx + k
        dur = max(0.8, float(s.get("duration") or 0.0))
        fade_out_st = max(0.0, dur - _DECLICK_FADE)
        chain = (
            "aresample=async=1,"
            "aformat=sample_rates=48000:channel_layouts=stereo,"
            f"{lead_trim}"
            f"apad,atrim=0:{dur:.3f},asetpts=N/SR/TB,"
            f"afade=t=in:st=0:d={_DECLICK_FADE},"
            f"afade=t=out:st={fade_out_st:.3f}:d={_DECLICK_FADE}"
        )
        lines.append(f"[{idx}:a]{chain}[a{k}]")
        labels.append(f"[a{k}]")
    lines.append(f"{''.join(labels)}concat=n={len(scenes)}:v=0:a=1[voc]")
    return lines


# --------------------------------------------------------------------------- #
# Subtitles (.srt + burned .ass)
# --------------------------------------------------------------------------- #
def _fmt_srt(seconds):
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_ass(seconds):
    cs = int(round(max(0.0, seconds) * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6_000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _wrap(text, max_chars, sep="\\N"):
    lines, cur = [], ""
    for w in text.split():
        if len(cur) + len(w) + 1 > max_chars and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    if len(lines) > 2:
        lines = [lines[0], " ".join(lines[1:])]
    return sep.join(lines)


def _split_into_cues(text, max_chars):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    budget = max_chars * 2
    parts = re.split(r"(?<=[,;:.!?…])\s+", text)
    cues, cur = [], ""
    for part in parts:
        if len(cur) + len(part) + 1 > budget and cur:
            cues.append(cur.strip())
            cur = part
        else:
            cur = f"{cur} {part}".strip()
    if cur:
        cues.append(cur.strip())
    return cues


def _timed_cues(scenes, st):
    t = 0.0
    for scene in scenes:
        dur = max(0.5, float(scene.get("duration") or 0.0))
        cues = _split_into_cues(scene.get("text", ""), st["max_chars_per_line"]) \
            or [scene.get("text", "")]
        total_chars = sum(len(c) for c in cues) or 1
        cue_start = t
        for cue in cues:
            cue_dur = max(0.8, dur * (len(cue) / total_chars))
            cue_end = min(t + dur, cue_start + cue_dur)
            yield cue_start, cue_end, cue
            cue_start = cue_end
        t += dur


def build_srt(scenes, st, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for i, (start, end, cue) in enumerate(_timed_cues(scenes, st), start=1):
        blocks.append(f"{i}\n{_fmt_srt(start)} --> {_fmt_srt(end)}\n"
                      f"{_wrap(cue, st['max_chars_per_line'])}\n")
    out_path.write_text("\n".join(blocks), encoding="utf-8")
    return out_path


def _ass_header(st, width, height):
    fontsize = max(12, int(round(st["font_size"] * height / _FONT_DIVISOR)))
    bold = -1 if st["bold"] else 0
    style_line = (
        f"Style: Default,{st['font']},{fontsize},"
        f"{st['primary_color']},&H000000FF,{st['outline_color']},&H64000000,"
        f"{bold},0,0,0,100,100,0,0,1,{st['outline']},{st['shadow']},2,"
        f"60,60,{st['margin_v']},1")
    return (
        "[Script Info]\nScriptType: v4.00+\nScaledBorderAndShadow: yes\n"
        f"WrapStyle: 0\nPlayResX: {width}\nPlayResY: {height}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"{style_line}\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n")


def _ass_dialogue(start, end, cue, st):
    text = _wrap(cue, st["max_chars_per_line"]).replace("{", "(").replace("}", ")")
    return f"Dialogue: 0,{_fmt_ass(start)},{_fmt_ass(end)},Default,,0,0,0,,{text}\n"


def build_ass(scenes, st, width, height, out_path):
    """Whole-video burned subtitle track (absolute timing). Kept for callers that
    burn in one pass; the renderer now burns per-clip via build_scene_ass."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_ass_header(st, width, height)]
    for start, end, cue in _timed_cues(scenes, st):
        lines.append(_ass_dialogue(start, end, cue, st))
    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path


def build_scene_ass(scene, st, width, height, out_path):
    """ASS for ONE scene, cues timed 0..dur (burned into that scene's clip).
    Returns the path, or None if the scene has no text."""
    text = (scene.get("text") or "").strip()
    if not text:
        return None
    dur = max(0.5, float(scene.get("duration") or 0.0))
    cues = _split_into_cues(text, st["max_chars_per_line"]) or [text]
    total_chars = sum(len(c) for c in cues) or 1
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_ass_header(st, width, height)]
    t = 0.0
    for cue in cues:
        end = min(dur, t + max(0.8, dur * (len(cue) / total_chars)))
        lines.append(_ass_dialogue(t, end, cue, st))
        t = end
    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def render(scenes, work_dir, out_path, render=None, subtitle=None,
           on_log=None, on_stage=None, on_progress=None):
    """Render scenes → ``out_path`` (MP4). Returns (mp4_path, srt_path|None)."""
    r = {**DEFAULT_RENDER, **(render or {})}
    st = {**DEFAULT_SUBTITLE, **(subtitle or {})}
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    def _log(m):
        if on_log:
            on_log(m)

    def _stage(m):
        if on_stage:
            on_stage(m)
        _log(f"▶ {m}")

    scenes = [s for s in scenes if (s.get("text") or "").strip()]
    if not scenes:
        raise RuntimeError("Không có cảnh nào để dựng.")
    n = len(scenes)
    clip_paths = [work_dir / f"clip_{i:03d}.mp4" for i in range(n)]

    # --- 0. subtitles ------------------------------------------------------ #
    # Build the whole-video .srt sidecar (for download) AND one per-scene .ass
    # that each clip burns itself. Burning per-clip parallelises libass across the
    # clip workers so the final pass can stream-COPY the video (no full re-encode).
    srt_path = None
    sub_names = [None] * n
    if st.get("enabled", True):
        _stage("Đang tạo phụ đề")
        srt_path = build_srt(scenes, st, Path(out_path).with_suffix(".srt"))
        for i, scene in enumerate(scenes):
            name = f"sub_{i:03d}.ass"
            if build_scene_ass(scene, st, r["width"], r["height"], work_dir / name):
                sub_names[i] = name

    # --- 1. per-scene clips (parallel, subtitles burned in) ---------------- #
    workers = int(r.get("clip_workers") or 0)
    if workers <= 0:
        # Each clip's ffmpeg (zoompan + lanczos + libass) is ~single-threaded, so
        # on a many-core login node run as many clips at once as cores (capped).
        workers = min(16, max(2, os.cpu_count() or 4))
    workers = max(1, min(workers, n))
    _stage(f"Đang dựng {n} cảnh ({workers} luồng song song)")
    done = 0
    if workers == 1:
        for i, scene in enumerate(scenes):
            _render_scene_clip(scene, i, clip_paths[i], r, work_dir, sub_names[i])
            done += 1
            if on_progress:
                on_progress(0.05 + 0.80 * done / n)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_render_scene_clip, scenes[i], i, clip_paths[i],
                                r, work_dir, sub_names[i]): i for i in range(n)}
            try:
                for fut in as_completed(futs):
                    fut.result()
                    done += 1
                    _stage(f"Đã dựng {done}/{n} cảnh")
                    if on_progress:
                        on_progress(0.05 + 0.80 * done / n)
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise

    # --- 2. concat list ---------------------------------------------------- #
    _stage("Đang ghép các cảnh")
    concat_file = work_dir / "concat.txt"
    concat_file.write_text("".join(f"file '{c.name}'\n" for c in clip_paths),
                           encoding="utf-8")

    # --- 3. final assembly: COPY video, build audio, mux ------------------- #
    # Video is stream-copied from the (already subtitled) clips — no re-encode of
    # the whole movie, which was the slow "Ghép hoàn thiện" step. Only the audio is
    # (re)built gaplessly from the raw per-scene wavs + loudnorm, then muxed.
    _stage("Ghép hoàn thiện (ghép audio + nối video)")
    inputs = ["-f", "concat", "-safe", "0", "-fflags", "+genpts", "-i",
              concat_file.name]
    inputs += _audio_inputs(scenes, work_dir)   # inputs 1..n
    audio_start = 1

    music = r.get("music_path") or ""
    has_music = bool(music) and Path(music).exists()
    if has_music:
        inputs += ["-stream_loop", "-1", "-i", str(music)]
        music_idx = audio_start + n

    filters = list(_voice_concat_filters(scenes, audio_start))
    if has_music:
        filters.append(f"[{music_idx}:a]volume={r['music_volume']}[mus]")
        filters.append("[voc][mus]amix=inputs=2:duration=first:normalize=0[mix]")
        filters.append("[mix]loudnorm=I=-16:TP=-1.5:LRA=11[a]")
    else:
        filters.append("[voc]loudnorm=I=-16:TP=-1.5:LRA=11[a]")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    filter_script = work_dir / "_render_filter.txt"
    filter_script.write_text(";".join(filters), encoding="utf-8")

    args = [*inputs, *_FFMPEG_THREADS,
            "-filter_complex_script", filter_script.name,
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy",                      # ← no whole-video re-encode
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", str(out_path.resolve())]
    try:
        rc, log = _run_ffmpeg(args, cwd=work_dir)
    finally:
        filter_script.unlink(missing_ok=True)
    if rc != 0 or not out_path.exists() or out_path.stat().st_size < 1024:
        raise RuntimeError(f"ffmpeg dựng video thất bại: {(log or '')[-600:]}")
    if on_progress:
        on_progress(1.0)
    _stage("Hoàn tất")
    return str(out_path), (str(srt_path) if srt_path else None)
