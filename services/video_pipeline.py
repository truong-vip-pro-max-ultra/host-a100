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
import threading

import config
from services import storage_service as db
from services import video_prompts, video_render, voice_service, voice_pipeline
from utils import file_utils


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
def list_jobs():
    rows = db.execute("SELECT * FROM video_jobs ORDER BY created_at DESC LIMIT 100",
                      fetch="all") or []
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
              height=1080, fps=30, ken_burns=True, image_steps=4,
              image_batch=None, voice_batch=None, music_path=""):
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
    except (TypeError, ValueError):
        raise ValueError("Tham số số (seed/steps/kích thước/batch) không hợp lệ.")

    params = {
        "profile": profile_name, "language": language or "vi",
        "voice_seed": voice_seed, "num_step": num_step,
        "style": style or "cinematic", "use_llm": bool(use_llm),
        "width": width, "height": height, "fps": fps,
        "ken_burns": bool(ken_burns), "image_steps": image_steps,
        "image_batch": ib, "voice_batch": vb, "music_path": music_path or "",
    }
    name = (name or "video").strip()[:128]
    job_id = db.execute(
        "INSERT INTO video_jobs (name, status, progress, stage, params, created_at) "
        "VALUES (?, 'queued', 0, ?, ?, ?)",
        (name, "Đang chuẩn bị…", json.dumps(params, ensure_ascii=False), db.now()),
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
# Run
# --------------------------------------------------------------------------- #
def _run_job(job_id, script, params):
    job = get_job(job_id)
    job_dir = job["logs_path"]
    log = lambda m: _job_log(job_dir, m)  # noqa: E731
    try:
        _set_job(job_id, status="running", stage="Đang chia kịch bản…", progress=2)
        chunks = voice_pipeline.parse_script(script)
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

        # --- images (GPU, grouped) ---------------------------------------- #
        w, h = params["width"], params["height"]
        steps = params.get("image_steps", 4)
        img_paths = [os.path.join(job_dir, f"img_{i:04d}.png") for i in range(n)]
        ib = int(params.get("image_batch") or config.IMAGE_MAX_BATCH)
        done = 0
        for start in range(0, n, ib):
            grp = list(range(start, min(start + ib, n)))
            _set_job(job_id, stage=f"Đang tạo ảnh {grp[0] + 1}–{grp[-1] + 1}/{n} "
                     "trên GPU…", progress=5 + int(done / n * 35))
            items = [{"prompt": prompts[i]["prompt"],
                      "negative": prompts[i]["negative"],
                      "out_path": img_paths[i], "seed": prompts[i]["seed"],
                      "width": w, "height": h, "steps": steps} for i in grp]
            results = _generate_images(host, port, items)
            for k, i in enumerate(grp):
                r = results[k] if k < len(results) else {}
                if not (r or {}).get("ok"):
                    log(f"  ! ảnh cảnh {i + 1} lỗi: {(r or {}).get('error', '?')} "
                        "→ dùng nền phẳng")
            done += len(grp)

        # --- voice (GPU, reuse voice_pipeline) ----------------------------- #
        wav_paths = [os.path.join(job_dir, f"voice_{i:04d}.wav") for i in range(n)]
        vparams = {"profile": params.get("profile", ""),
                   "language": params.get("language", "vi"),
                   "num_step": params.get("num_step", 16),
                   "seed": params.get("voice_seed", 0),
                   "batch": params.get("voice_batch") or config.OMNI_MAX_BATCH}
        vb = int(vparams["batch"])
        durations = [0.0] * n
        done = 0
        for start in range(0, n, vb):
            grp = list(range(start, min(start + vb, n)))
            _set_job(job_id, stage=f"Đang tạo giọng {grp[0] + 1}–{grp[-1] + 1}/{n} "
                     "trên GPU…", progress=40 + int(done / n * 30))
            items = [{"text": chunks[i], "out_path": wav_paths[i]} for i in grp]
            results = voice_pipeline._synthesize_chunks(host, port, items, vparams)
            for k, i in enumerate(grp):
                r = results[k] if k < len(results) else {}
                if (r or {}).get("ok"):
                    durations[i] = float(r.get("duration") or 0.0)
                else:
                    log(f"  ! giọng cảnh {i + 1} lỗi: {(r or {}).get('error', '?')} "
                        "→ chèn khoảng lặng")
            done += len(grp)

        # --- assemble scenes ---------------------------------------------- #
        scenes = []
        for i in range(n):
            img = img_paths[i] if os.path.exists(img_paths[i]) else ""
            wav = wav_paths[i] if (os.path.exists(wav_paths[i])
                                   and os.path.getsize(wav_paths[i]) > 256) else ""
            # No voice → give the scene a sane on-screen time from its text length.
            dur = durations[i] if durations[i] > 0 else max(2.5, len(chunks[i]) / 16.0)
            scenes.append({"text": chunks[i], "image_path": img,
                           "audio_path": wav, "duration": dur})

        # --- render -------------------------------------------------------- #
        _set_job(job_id, stage="Đang dựng video (ffmpeg)…", progress=70)
        out_mp4 = os.path.join(job_dir, f"{voice_pipeline._slug(job['name'] or 'video')}.mp4")
        render_cfg = {"width": w, "height": h, "fps": params.get("fps", 30),
                      "ken_burns": params.get("ken_burns", True),
                      "music_path": params.get("music_path", "")}
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


def bulk_delete(ids=None, all_jobs=False):
    """Delete many jobs. Returns the count removed."""
    if all_jobs:
        ids = [j["id"] for j in list_jobs()]
    count = 0
    for jid in ids or []:
        try:
            if delete_job(int(jid)):
                count += 1
        except (TypeError, ValueError):
            continue
    return count
