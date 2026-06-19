"""
Tools / Gen video — the LOGIN-NODE orchestrator (script text → MP4).

Mirrors ``voice_pipeline._run_job`` but produces a full video. It reuses the
SAME GPU OmniVoice server (``voice_service``) for BOTH:
  * per-scene images  → POST /generate_image_batch  (diffusers SDXL on the GPU)
  * per-scene voice   → POST /synthesize_batch       (OmniVoice TTS, reused via
                        ``voice_pipeline._synthesize_chunks``)
and stitches the result with ffmpeg on the login node (``video_render``).

Stages: chia kịch bản → viết prompt (LLM API farm / rule-based) → tạo ảnh trên
GPU → tạo giọng trên GPU → dựng video (ffmpeg). One bad image falls back to a
flat frame so the video still renders; a bad voice chunk becomes timed silence.
"""
from __future__ import annotations

import http.client
import json
import os
import random
import re
import threading
import time
import unicodedata

import config
from services import storage_service as db
from services import video_prompts, video_render, voice_service, voice_pipeline
from utils import file_utils


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
def normalize_owner(username):
    """The username namespace a video lives in. Blank/whitespace → '' (public).
    Trimmed + capped so two spellings of the same name don't split a library."""
    return (username or "").strip()[:64]


def list_jobs(owner=None):
    """Videos in one namespace. owner blank/None → only PUBLIC videos (owner='');
    a non-blank owner → only that user's videos (exact match — wrong name sees none)."""
    owner = normalize_owner(owner)
    if owner:
        rows = db.execute(
            "SELECT * FROM video_jobs WHERE owner=? ORDER BY created_at DESC LIMIT 100",
            (owner,), fetch="all") or []
    else:
        rows = db.execute(
            "SELECT * FROM video_jobs WHERE COALESCE(owner,'')='' "
            "ORDER BY created_at DESC LIMIT 100", fetch="all") or []
    return [dict(r) for r in rows]


def get_job(job_id):
    row = db.execute("SELECT * FROM video_jobs WHERE id=?", (job_id,), fetch="one")
    return dict(row) if row else None


def _job_log(job_dir, text):
    try:
        with open(os.path.join(job_dir, "job.log"), "a",
                  encoding="utf-8", errors="replace") as fh:
            fh.write(text + "\n")
    except OSError:
        pass


def _write_scenes(job_dir, scenes):
    """Persist the per-scene storyboard (text + image prompt + image-ready flag) so
    the UI can show the generated images as they appear. Rewritten after the prompt
    step and after each image batch (cheap; n is small)."""
    # Write atomically (temp + os.replace) so a concurrent poll can never read a
    # half-written file and get an empty list mid-update.
    try:
        path = os.path.join(job_dir, "scenes.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"scenes": scenes}, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        pass


def get_scenes(job_id):
    """Storyboard for a job: list of {i, text, prompt, image} with image-existence
    reconciled against the files on disk at read time."""
    job = get_job(job_id)
    if not job or not job.get("logs_path"):
        return []
    job_dir = job["logs_path"]
    scenes = []
    try:
        with open(os.path.join(job_dir, "scenes.json"), encoding="utf-8") as fh:
            scenes = (json.load(fh) or {}).get("scenes") or []
    except (OSError, ValueError):
        return []
    for s in scenes:
        idx = int(s.get("i", 0)) - 1
        img = os.path.join(job_dir, f"img_{idx:04d}.png")
        ok = idx >= 0 and os.path.exists(img) and os.path.getsize(img) > 256
        s["image"] = ok
        # A version stamp (image mtime) so the UI can cache-bust the thumbnail and
        # auto-refresh it the moment a scene is re-uploaded or regenerated.
        s["ver"] = int(os.path.getmtime(img)) if ok else 0
    return scenes


def scene_image_path(job_id, idx):
    """Absolute path to a scene PNG if it exists inside the job dir, else None."""
    job = get_job(job_id)
    if not job or not job.get("logs_path"):
        return None
    p = os.path.join(job["logs_path"], f"img_{int(idx):04d}.png")
    if file_utils.is_within(config.VIDEO_OUTPUTS_DIR, p) and os.path.exists(p):
        return p
    return None


_HR_RE = re.compile(r"^[-=_*~]{3,}$")                  # --- / *** rule lines
_BRACKET_HEADER_RE = re.compile(r"^\[.*\]$")           # [PHẦN 1 – 0:10-0:30]
# A short label ending with ":" (Narrator:, Caster:, Khán giả hô vang:). \w under
# Python str regex already includes Vietnamese letters.
_SPEAKER_RE = re.compile(r"^[\w][\w .]{0,28}:$", re.UNICODE)
_INLINE_PARENS = re.compile(r"\([^)]*\)")
_INLINE_SPEAKER = re.compile(r'^[\w][\w .]{0,24}:\s+(?=["“\w])', re.UNICODE)


def clean_screenplay(text):
    """Strip non-spoken screenplay scaffolding so only the narration/dialogue is
    read aloud + split into scenes. Removes: section headers ([MỞ ĐẦU – 0:00],
    bold-wrapped *titles*/*on-screen text*), horizontal rules, stage directions in
    (parentheses), speaker labels (Narrator:/Caster:…), markdown *_` emphasis, and
    surrounding quotes. Safe on plain prose (nothing matches → unchanged)."""
    out = []
    for raw in (text or "").replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not line:
            out.append("")                                # keep paragraph breaks
            continue
        # A whole line wrapped in *bold* = title / header / on-screen text → not spoken.
        bold_line = len(line) > 2 and line[0] == "*" and line[-1] == "*"
        s = re.sub(r"[*_`]+", "", line).strip()           # drop markdown emphasis
        if not s or _HR_RE.match(s) or _BRACKET_HEADER_RE.match(s) or bold_line:
            continue
        # Full stage-direction line, or inline parentheticals, in (parentheses).
        if s.startswith("("):
            s = _INLINE_PARENS.sub("", s).strip()
            if not s:
                continue
        else:
            s = _INLINE_PARENS.sub("", s).strip()
        if not s or _SPEAKER_RE.match(s):                 # bare "Narrator:" label
            continue
        s = _INLINE_SPEAKER.sub("", s)                    # inline "Narrator: ..." prefix
        s = s.strip().strip('"“”').strip()
        if s:
            out.append(s)
    cleaned = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _ascii_slug(name, fallback):
    """An ASCII-only, filesystem-safe slug (strips Vietnamese diacritics). The
    output MP4/SRT names must be ASCII: a YouTube-title job name carries accents,
    and on a C/POSIX-locale login node Python encodes filenames with ascii →
    opening a file whose NAME has 'ờ'/'đ' raises 'latin-1' codec can't encode.
    The display name (job['name']) keeps its Vietnamese; only the file is sanitised."""
    s = unicodedata.normalize("NFKD", name or "")
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\-]+", "_", s).strip("_")
    return s or fallback


def _set_job(job_id, **fields):
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    db.execute(f"UPDATE video_jobs SET {cols} WHERE id=?",
               (*fields.values(), job_id), commit=True)


# --------------------------------------------------------------------------- #
# Start
# --------------------------------------------------------------------------- #
def start_job(name, script, profile_name="", language="vi", voice_seed=0,
              num_step=16, style="cinematic", use_llm=True, width=1920,
              height=1080, fps=30, ken_burns=True, image_steps=6,
              image_batch=None, voice_batch=None, music_path="", voice_speed=1.0,
              image_cfg=1.0, owner=""):
    """Create a video_jobs row + kick off the render thread. Validates that a GPU
    server is ready, ffmpeg exists, and the script is non-empty."""
    script = (script or "").strip()
    if not script:
        raise ValueError("Kịch bản trống — hãy nhập nội dung trước.")
    if not config.ffmpeg_available():
        raise ValueError("Không tìm thấy ffmpeg trên login node — không dựng được "
                         "video. Đặt ffmpeg vào thư mục ffmpeg/ cạnh app.")
    if not voice_service.resolve_endpoint():
        raise ValueError("Chưa có server GPU nào sẵn sàng. Hãy khởi động một server "
                         "OmniVoice ở tab Clone giọng nói trước (server đó lo cả "
                         "ảnh lẫn giọng cho video).")
    profile_name = (profile_name or "").strip()
    if profile_name and not voice_pipeline.get_profile(profile_name):
        raise ValueError(f"Không tìm thấy giọng “{profile_name}”.")
    try:
        voice_seed = int(voice_seed)
        num_step = max(1, int(num_step))
        width = max(256, int(width))
        height = max(256, int(height))
        fps = max(1, min(60, int(fps)))
        image_steps = max(1, int(image_steps))
        ib = max(1, int(image_batch)) if image_batch else config.IMAGE_MAX_BATCH
        vb = max(1, int(voice_batch)) if voice_batch else None
        voice_speed = float(voice_speed)
        # CFG for the image model. 1.0 = no classifier-free guidance (fast, the
        # negative prompt is ignored — fine, the resolution bucket + model handle
        # twins); >1 = CFG on (negative bites, ~2× slower). Clamp to a sane range.
        image_cfg = max(1.0, min(float(image_cfg), 8.0))
    except (TypeError, ValueError):
        raise ValueError("Tham số số (seed/steps/kích thước/batch/tốc độ) không hợp lệ.")
    if not (0.5 <= voice_speed <= 2.0):
        raise ValueError("Tốc độ đọc phải trong khoảng 0.5–2.0.")

    params = {
        "profile": profile_name, "language": language or "vi",
        "voice_seed": voice_seed, "num_step": num_step,
        "style": style or "cinematic", "use_llm": bool(use_llm),
        "width": width, "height": height, "fps": fps,
        "ken_burns": bool(ken_burns), "image_steps": image_steps,
        "image_batch": ib, "voice_batch": vb, "music_path": music_path or "",
        "voice_speed": voice_speed, "image_cfg": image_cfg,
    }
    name = (name or "video").strip()[:128]
    owner = normalize_owner(owner)
    job_id = db.execute(
        "INSERT INTO video_jobs (name, owner, status, progress, stage, params, created_at) "
        "VALUES (?, ?, 'queued', 0, ?, ?, ?)",
        (name, owner, "Đang chuẩn bị…", json.dumps(params, ensure_ascii=False), db.now()),
        commit=True)
    job_dir = file_utils.safe_join(config.VIDEO_OUTPUTS_DIR, str(job_id))
    os.makedirs(job_dir, exist_ok=True)
    _set_job(job_id, logs_path=job_dir)

    threading.Thread(target=_run_job, args=(job_id, script, params),
                     daemon=True).start()
    return job_id


# --------------------------------------------------------------------------- #
# GPU image calls (same server as voice)
# --------------------------------------------------------------------------- #
def _generate_images(host, port, items, timeout=3600):
    """POST one /generate_image_batch group. Returns the results list."""
    body = json.dumps({"items": items}).encode("utf-8")
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.request("POST", "/generate_image_batch", body=body,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"Server ảnh trả lỗi {resp.status}: "
                           f"{raw.decode('utf-8', 'replace')[:300]}")
    data = json.loads(raw or b"{}")
    if not data.get("ok"):
        raise RuntimeError(data.get("error") or "Lỗi không xác định từ server ảnh.")
    return data.get("results") or []


# --------------------------------------------------------------------------- #
# Per-scene edits + re-render (no GPU re-run — just swap an image, redo ffmpeg)
# --------------------------------------------------------------------------- #
# scenes.json is read-modify-written from a few places (regenerate threads, the
# upload route); serialise those updates so two edits can't clobber each other.
_SCENES_LOCK = threading.Lock()


def _job_params(job):
    try:
        return json.loads(job.get("params") or "{}")
    except (TypeError, ValueError):
        return {}


def _read_scenes_raw(job_dir):
    try:
        with open(os.path.join(job_dir, "scenes.json"), encoding="utf-8") as fh:
            return (json.load(fh) or {}).get("scenes") or []
    except (OSError, ValueError):
        return []


def _update_scene(job_dir, idx, **fields):
    """Atomically merge ``fields`` into the scene whose 1-based index is idx+1."""
    with _SCENES_LOCK:
        scenes = _read_scenes_raw(job_dir)
        for s in scenes:
            if int(s.get("i", 0)) - 1 == idx:
                s.update(fields)
                break
        _write_scenes(job_dir, scenes)


def _manifest_path(job_dir):
    return os.path.join(job_dir, "manifest.json")


def _write_manifest(job_dir, scenes, render_cfg, out_name):
    """Persist everything ffmpeg needs to rebuild the MP4 from the files already on
    disk (image/voice basenames + per-scene duration), so a re-render touches no
    GPU. Stored as basenames so the dir stays relocatable and a swapped image file
    (same name) is picked up automatically."""
    payload = {
        "scenes": [{"text": s.get("text", ""),
                    "image": os.path.basename(s.get("image_path") or "") or None,
                    "audio": os.path.basename(s.get("audio_path") or "") or None,
                    "duration": float(s.get("duration") or 0.0)} for s in scenes],
        "render": dict(render_cfg or {}),
        "out_name": out_name,
    }
    try:
        path = _manifest_path(job_dir)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        pass


def _probe_duration(path):
    """Audio length in seconds via ffprobe, or 0.0 if unknown."""
    import subprocess
    try:
        out = subprocess.run(
            [config.ffprobe_path(), "-v", "error", "-show_entries",
             "format=duration", "-of", "default=nw=1:nk=1", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=30)
        return max(0.0, float((out.stdout or "0").strip() or 0))
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def _build_render_scenes(job, manifest=None):
    """The list of render scenes ({text,image_path,audio_path,duration}) + the
    render config, rebuilt from the manifest when present, else reconstructed from
    scenes.json + the wavs on disk (so older jobs without a manifest still work)."""
    job_dir = job["logs_path"]
    params = _job_params(job)
    if manifest and manifest.get("scenes"):
        speed = float((manifest.get("render") or {}).get("voice_speed", 1.0)) or 1.0
        scenes = []
        for m in manifest["scenes"]:
            img = os.path.join(job_dir, m["image"]) if m.get("image") else ""
            wav = os.path.join(job_dir, m["audio"]) if m.get("audio") else ""
            scenes.append({
                "text": m.get("text", ""),
                "image_path": img if img and os.path.exists(img) else "",
                "audio_path": wav if wav and os.path.exists(wav) else "",
                "duration": float(m.get("duration") or 0.0)})
        render_cfg = manifest.get("render") or {}
        return scenes, render_cfg, speed
    # Fallback: rebuild from scenes.json text + measured wav durations.
    speed = float(params.get("voice_speed", 1.0)) or 1.0
    raw = _read_scenes_raw(job_dir)
    scenes = []
    for s in raw:
        idx = int(s.get("i", 0)) - 1
        img = os.path.join(job_dir, f"img_{idx:04d}.png")
        wav = os.path.join(job_dir, f"voice_{idx:04d}.wav")
        has_wav = os.path.exists(wav) and os.path.getsize(wav) > 256
        raw_dur = _probe_duration(wav) if has_wav else 0.0
        dur = (raw_dur / speed) if raw_dur > 0 else max(2.5, len(s.get("text", "")) / 16.0)
        scenes.append({"text": s.get("text", ""),
                       "image_path": img if os.path.exists(img) else "",
                       "audio_path": wav if has_wav else "",
                       "duration": dur})
    render_cfg = {"width": params.get("width", 1920), "height": params.get("height", 1080),
                  "fps": params.get("fps", 30), "ken_burns": params.get("ken_burns", True),
                  "music_path": params.get("music_path", ""), "voice_speed": speed}
    return scenes, render_cfg, speed


def replace_scene_image(job_id, idx, src_path):
    """Swap scene ``idx``'s image with the uploaded file. Re-encodes through ffmpeg
    to a PNG sized to the job's resolution (validates it's a real image, strips any
    bad metadata, matches the aspect the renderer expects). Returns the new mtime."""
    import subprocess
    job = get_job(job_id)
    if not job or not job.get("logs_path"):
        raise ValueError("Không tìm thấy tác vụ.")
    job_dir = job["logs_path"]
    n = len(_read_scenes_raw(job_dir))
    if not (0 <= idx < n):
        raise ValueError("Cảnh không hợp lệ.")
    params = _job_params(job)
    w = int(params.get("width", 1920)); h = int(params.get("height", 1080))
    out = os.path.join(job_dir, f"img_{idx:04d}.png")
    if not file_utils.is_within(config.VIDEO_OUTPUTS_DIR, out):
        raise ValueError("Đường dẫn không hợp lệ.")
    tmp = out + ".upload.png"
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=increase,"
          f"crop={w}:{h},setsar=1")
    cmd = [config.ffmpeg_path(), "-y", "-hide_banner", "-nostdin", "-i", str(src_path),
           "-vf", vf, "-frames:v", "1", tmp]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace", timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"Không xử lý được ảnh: {exc}")
    if proc.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) < 256:
        raise ValueError("File tải lên không phải ảnh hợp lệ.")
    os.replace(tmp, out)
    _update_scene(job_dir, idx, image=True, gen="ok", error="", edited=True)
    _job_log(job_dir, f"• Đã thay ảnh cảnh {idx + 1} bằng ảnh tải lên.")
    return int(os.path.getmtime(out))


def regenerate_scene_image(job_id, idx, prompt=None, negative=None, seed=None):
    """Re-generate scene ``idx``'s image on the GPU (optionally with an edited
    prompt/seed). Runs in a background thread; the storyboard poll shows gen=busy →
    ok/err. Returns nothing useful synchronously — raises only on bad input / no GPU."""
    job = get_job(job_id)
    if not job or not job.get("logs_path"):
        raise ValueError("Không tìm thấy tác vụ.")
    job_dir = job["logs_path"]
    scenes = _read_scenes_raw(job_dir)
    if not (0 <= idx < len(scenes)):
        raise ValueError("Cảnh không hợp lệ.")
    endpoint = voice_service.resolve_endpoint()
    if not endpoint:
        raise ValueError("Chưa có server GPU sẵn sàng — hãy khởi động ở tab Clone "
                         "giọng nói trước.")
    host, port = endpoint
    scene = scenes[idx]
    params = _job_params(job)
    w = int(params.get("width", 1920)); h = int(params.get("height", 1080))
    steps = int(params.get("image_steps", 6))
    cfg = float(params.get("image_cfg", 1.0) or 1.0)
    prompt = (prompt or "").strip() or scene.get("prompt", "")
    negative = (negative or "").strip() or scene.get("negative") or video_prompts.DEFAULT_NEGATIVE
    # No explicit seed → reroll with a fresh random one (re-using the stored seed +
    # prompt would just reproduce the SAME image, defeating the point of "Tạo lại").
    try:
        seed = int(seed) if seed is not None and str(seed).strip() != "" \
            else random.randint(0, 2 ** 31 - 1)
    except (TypeError, ValueError):
        seed = random.randint(0, 2 ** 31 - 1)
    if not prompt:
        raise ValueError("Prompt trống.")
    out = os.path.join(job_dir, f"img_{idx:04d}.png")
    _update_scene(job_dir, idx, prompt=prompt, negative=negative, seed=seed,
                  gen="busy", error="")

    def _work():
        item = {"prompt": prompt, "negative": negative, "out_path": out,
                "seed": seed, "width": w, "height": h, "steps": steps, "cfg": cfg}
        try:
            res = _generate_images(host, port, [item])
            r = (res or [{}])[0] or {}
            if r.get("ok") and os.path.exists(out) and os.path.getsize(out) > 256:
                _update_scene(job_dir, idx, image=True, gen="ok", error="", edited=True)
                _job_log(job_dir, f"• Đã tạo lại ảnh cảnh {idx + 1} trên GPU.")
            else:
                err = r.get("error") or "không tạo được ảnh"
                _update_scene(job_dir, idx, gen="err", error=str(err))
                _job_log(job_dir, f"! Tạo lại ảnh cảnh {idx + 1} lỗi: {err}")
        except Exception as exc:  # noqa: BLE001
            _update_scene(job_dir, idx, gen="err", error=str(exc))
            _job_log(job_dir, f"! Tạo lại ảnh cảnh {idx + 1} lỗi: {exc}")

    threading.Thread(target=_work, daemon=True).start()


def rerender_job(job_id):
    """Rebuild the MP4 from the images/voices already on disk (ffmpeg only, no GPU).
    Used after the user swaps one or more scene images. Runs in a background thread
    and drives the same status/progress fields the job list polls."""
    job = get_job(job_id)
    if not job or not job.get("logs_path"):
        raise ValueError("Không tìm thấy tác vụ.")
    if job.get("status") == "running":
        raise ValueError("Tác vụ đang chạy — đợi xong rồi hãy dựng lại.")
    if not config.ffmpeg_available():
        raise ValueError("Không tìm thấy ffmpeg trên login node.")
    job_dir = job["logs_path"]
    manifest = _read_manifest(job_dir)
    scenes, render_cfg, _speed = _build_render_scenes(job, manifest)
    if not any(s.get("text", "").strip() for s in scenes):
        raise ValueError("Không còn dữ liệu cảnh để dựng lại (tác vụ quá cũ?).")
    out_name = (manifest or {}).get("out_name") \
        or f"{_ascii_slug(job['name'], f'video_{job_id}')}.mp4"
    out_mp4 = os.path.join(job_dir, out_name)

    def _work():
        log = lambda m: _job_log(job_dir, m)  # noqa: E731
        try:
            _set_job(job_id, status="running", stage="Đang dựng lại video…",
                     progress=5, error="")
            log("↻ Dựng lại video từ ảnh/giọng đã có (không chạy lại GPU).")
            mp4, srt = video_render.render(
                scenes, job_dir, out_mp4, render=render_cfg, on_log=log,
                on_stage=lambda m: _set_job(job_id, stage=m),
                on_progress=lambda p: _set_job(
                    job_id, progress=5 + int(max(0.0, min(1.0, p)) * 95)))
            _set_job(job_id, status="done", stage="Hoàn tất (đã dựng lại)",
                     progress=100, output_path=mp4, srt_path=srt, finished_at=db.now())
            log(f"✓ MP4 (dựng lại): {mp4}")
        except Exception as exc:  # noqa: BLE001
            log(f"LỖI khi dựng lại: {exc}")
            _set_job(job_id, status="error", stage="Lỗi dựng lại", error=str(exc),
                     finished_at=db.now())

    threading.Thread(target=_work, daemon=True).start()


def _read_manifest(job_dir):
    try:
        with open(_manifest_path(job_dir), encoding="utf-8") as fh:
            return json.load(fh) or {}
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def _run_job(job_id, script, params):
    job = get_job(job_id)
    job_dir = job["logs_path"]
    log = lambda m: _job_log(job_dir, m)  # noqa: E731
    try:
        _set_job(job_id, status="running", stage="Đang chia kịch bản…", progress=2)
        cleaned = clean_screenplay(script)
        if cleaned and cleaned != script.strip():
            log("Đã lọc kịch bản (bỏ tiêu đề mục / chỉ dẫn cảnh / nhãn người nói / "
                "timestamp) — chỉ giữ lời thoại để đọc.")
        chunks = voice_pipeline.parse_script(cleaned or script)
        if not chunks:
            raise RuntimeError("Không tách được nội dung nào để dựng cảnh.")
        n = len(chunks)
        log(f"Chia kịch bản thành {n} cảnh.")

        endpoint = voice_service.resolve_endpoint()
        if not endpoint:
            raise RuntimeError("Server GPU không còn sẵn sàng.")
        host, port = endpoint

        # --- prompts ------------------------------------------------------- #
        _set_job(job_id, stage="Đang viết prompt ảnh…", progress=5)
        prompts = video_prompts.build_prompts(
            chunks, style_preset=params.get("style", "cinematic"),
            use_llm=params.get("use_llm", True), on_log=log)

        # Storyboard: record each scene's text + image prompt now (images not yet
        # ready) so the UI can show them filling in as the GPU renders. The negative
        # + seed are kept so a single scene can be regenerated on the GPU later.
        scene_meta = [{"i": i + 1, "text": chunks[i],
                       "prompt": prompts[i]["prompt"],
                       "negative": prompts[i]["negative"], "seed": prompts[i]["seed"],
                       "image": False}
                      for i in range(n)]
        _write_scenes(job_dir, scene_meta)

        # --- images (GPU, grouped) ---------------------------------------- #
        w, h = params["width"], params["height"]
        steps = params.get("image_steps", 4)
        img_paths = [os.path.join(job_dir, f"img_{i:04d}.png") for i in range(n)]
        ib = int(params.get("image_batch") or config.IMAGE_MAX_BATCH)
        log(f"Bắt đầu tạo {n} ảnh trên GPU (batch {ib}). Lần đầu GPU phải nạp "
            "model SDXL vào VRAM nên nhóm đầu chậm hơn, các nhóm sau nhanh.")
        done = 0
        for start in range(0, n, ib):
            grp = list(range(start, min(start + ib, n)))
            _set_job(job_id, stage=f"Đang tạo ảnh {grp[0] + 1}–{grp[-1] + 1}/{n} "
                     "trên GPU…", progress=5 + int(done / n * 35))
            items = [{"prompt": prompts[i]["prompt"],
                      "negative": prompts[i]["negative"],
                      "out_path": img_paths[i], "seed": prompts[i]["seed"],
                      "width": w, "height": h, "steps": steps,
                      "cfg": params.get("image_cfg", 1.0)} for i in grp]
            t0 = time.time()
            results = _generate_images(host, port, items)
            ok = sum(1 for r in results if (r or {}).get("ok"))
            log(f"  ✓ ảnh {grp[0] + 1}–{grp[-1] + 1}/{n}: {ok}/{len(grp)} cảnh "
                f"({time.time() - t0:.1f}s)")
            for k, i in enumerate(grp):
                r = results[k] if k < len(results) else {}
                if not (r or {}).get("ok"):
                    log(f"  ! ảnh cảnh {i + 1} lỗi: {(r or {}).get('error', '?')} "
                        "→ dùng nền phẳng")
                scene_meta[i]["image"] = os.path.exists(img_paths[i])
            _write_scenes(job_dir, scene_meta)   # surface the new images live
            done += len(grp)

        # --- voice (GPU, reuse voice_pipeline) ----------------------------- #
        wav_paths = [os.path.join(job_dir, f"voice_{i:04d}.wav") for i in range(n)]
        vparams = {"profile": params.get("profile", ""),
                   "language": params.get("language", "vi"),
                   "num_step": params.get("num_step", 16),
                   "seed": params.get("voice_seed", 0),
                   "batch": params.get("voice_batch") or config.OMNI_MAX_BATCH}
        vb = int(vparams["batch"])
        log(f"Bắt đầu tạo giọng {n} cảnh trên GPU (batch {vb}).")
        durations = [0.0] * n
        done = 0
        for start in range(0, n, vb):
            grp = list(range(start, min(start + vb, n)))
            _set_job(job_id, stage=f"Đang tạo giọng {grp[0] + 1}–{grp[-1] + 1}/{n} "
                     "trên GPU…", progress=40 + int(done / n * 30))
            items = [{"text": chunks[i], "out_path": wav_paths[i]} for i in grp]
            t0 = time.time()
            results = voice_pipeline._synthesize_chunks(host, port, items, vparams)
            ok = sum(1 for r in results if (r or {}).get("ok"))
            log(f"  ✓ giọng {grp[0] + 1}–{grp[-1] + 1}/{n}: {ok}/{len(grp)} cảnh "
                f"({time.time() - t0:.1f}s)")
            for k, i in enumerate(grp):
                r = results[k] if k < len(results) else {}
                if (r or {}).get("ok"):
                    durations[i] = float(r.get("duration") or 0.0)
                else:
                    log(f"  ! giọng cảnh {i + 1} lỗi: {(r or {}).get('error', '?')} "
                        "→ chèn khoảng lặng")
            done += len(grp)

        # --- assemble scenes ---------------------------------------------- #
        # Reading speed: the narration is time-stretched by `speed` in the renderer
        # (pitch-preserved atempo), so a voiced scene's ON-SCREEN time is the raw
        # speech length / speed. Faster reading → shorter scenes → shorter video;
        # the picture + subtitles follow the audio exactly. Silent scenes keep their
        # text-based estimate (no speech to speed up).
        speed = float(params.get("voice_speed", 1.0)) or 1.0
        scenes = []
        for i in range(n):
            img = img_paths[i] if os.path.exists(img_paths[i]) else ""
            wav = wav_paths[i] if (os.path.exists(wav_paths[i])
                                   and os.path.getsize(wav_paths[i]) > 256) else ""
            if durations[i] > 0:
                dur = durations[i] / speed
            else:
                dur = max(2.5, len(chunks[i]) / 16.0)
            scenes.append({"text": chunks[i], "image_path": img,
                           "audio_path": wav, "duration": dur})
        if abs(speed - 1.0) > 1e-3:
            log(f"Tốc độ đọc {speed:.2f}× — thời lượng mỗi cảnh & video co/giãn theo.")

        # --- render -------------------------------------------------------- #
        _set_job(job_id, stage="Đang dựng video (ffmpeg)…", progress=70)
        out_mp4 = os.path.join(job_dir, f"{_ascii_slug(job['name'], f'video_{job_id}')}.mp4")
        render_cfg = {"width": w, "height": h, "fps": params.get("fps", 30),
                      "ken_burns": params.get("ken_burns", True),
                      "music_path": params.get("music_path", ""),
                      "voice_speed": speed}
        # Persist a render manifest so the video can be RE-rendered later (after the
        # user swaps a scene image) without re-running any GPU work — only ffmpeg.
        _write_manifest(job_dir, scenes, render_cfg, os.path.basename(out_mp4))
        mp4, srt = video_render.render(
            scenes, job_dir, out_mp4, render=render_cfg,
            on_log=log,
            on_stage=lambda m: _set_job(job_id, stage=m),
            on_progress=lambda p: _set_job(job_id, progress=70 + int(max(0.0, min(1.0, p)) * 30)))

        _set_job(job_id, status="done", stage="Hoàn tất", progress=100,
                 output_path=mp4, srt_path=srt, finished_at=db.now())
        log(f"✓ MP4: {mp4}")
    except Exception as exc:  # noqa: BLE001
        log(f"LỖI: {exc}")
        _set_job(job_id, status="error", stage="Lỗi", error=str(exc),
                 finished_at=db.now())


# --------------------------------------------------------------------------- #
# Delete
# --------------------------------------------------------------------------- #
def delete_job(job_id):
    job = get_job(job_id)
    if not job:
        return False
    path = job.get("logs_path")
    if path and file_utils.is_within(config.VIDEO_OUTPUTS_DIR, path) \
            and os.path.isdir(path):
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    db.execute("DELETE FROM video_jobs WHERE id=?", (job_id,), commit=True)
    return True


def bulk_delete(ids=None, all_jobs=False, owner=None):
    """Delete many jobs. Returns the count removed. ``all_jobs`` is scoped to the
    given namespace (public when owner is blank) so it can't reach across users."""
    if all_jobs:
        ids = [j["id"] for j in list_jobs(owner)]
    count = 0
    for jid in ids or []:
        try:
            if delete_job(int(jid)):
                count += 1
        except (TypeError, ValueError):
            continue
    return count
