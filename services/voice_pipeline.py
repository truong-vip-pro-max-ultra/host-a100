"""
Tools / Clone giọng nói — the LOGIN-NODE half of the voice tool.

Everything here runs on the login node with no GPU: it splits a script into
chunks, asks the GPU OmniVoice server (``voice_service``) to render them, then
stitches the per-chunk WAVs into one smooth MP3 with ffmpeg — de-clicked, even
inter-sentence gaps, optional noise reduction, optional speed change, loudness
normalised — and writes a matching .SRT. The chunking / SRT / concat logic is
ported from the desktop ``clone-voice`` project (AI Story Studio narrator).

ffmpeg is told to use every core (``-threads 0`` + ``-filter_complex_threads 0``)
since the login node is CPU-only; the heavy neural work is the GPU server's job.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import http.client
from pathlib import Path

import config
from services import voice_service
from services import storage_service as db
from utils import file_utils

# --------------------------------------------------------------------------- #
# Script chunking (ported from clone-voice scene_parser — Vietnamese-aware).
# --------------------------------------------------------------------------- #
_SCENE_MARKER = re.compile(
    r"^\s*(?:scene|cảnh|canh|đoạn|doan|part)\s*\d+\s*[:\.\-–]\s*", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[\.!?…])\s+(?=[\"“(A-ZÀ-Ỹ0-9])")

MIN_WORDS = 8
TARGET_WORDS = 26
MAX_WORDS = 44
_BREAK_BEFORE = frozenset({
    "nhưng", "tuy", "song", "hoặc", "hay", "còn", "rồi", "vì", "bởi",
    "nên", "thì", "và", "với", "nếu", "khi", "vậy", "thế",
})


def _split_long(words):
    out = []
    start, n = 0, len(words)
    while start < n:
        if n - start <= MAX_WORDS:
            out.append(" ".join(words[start:]))
            break
        lo, hi = start + MIN_WORDS, min(start + MAX_WORDS, n)
        target = start + TARGET_WORDS
        best = None
        for i in range(lo, hi):
            prev = words[i - 1]
            cur = words[i].strip('"“”()').lower()
            if prev.endswith((",", ";", ":")):
                score = 3
            elif cur in _BREAK_BEFORE:
                score = 2
            else:
                continue
            cand = (score, -abs(i - target), i)
            if best is None or cand > best:
                best = cand
        cut = best[2] if best is not None else min(target, hi)
        out.append(" ".join(words[start:cut]))
        start = cut
    return out


def _split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]


def _strip_marker(line):
    return _SCENE_MARKER.sub("", line).strip().strip('"“”')


def _regroup(sentences):
    chunks, buffer, buf_words = [], [], 0

    def flush():
        nonlocal buffer, buf_words
        if buffer:
            chunks.append(" ".join(buffer).strip())
            buffer, buf_words = [], 0

    for sentence in sentences:
        words = sentence.split()
        if len(words) > MAX_WORDS:
            flush()
            chunks.extend(_split_long(words))
            continue
        if buf_words + len(words) > MAX_WORDS and buf_words >= MIN_WORDS:
            flush()
        buffer.append(sentence)
        buf_words += len(words)
        if buf_words >= TARGET_WORDS:
            flush()
    flush()
    return chunks


def parse_script(script):
    """Split raw script text into a list of chunk strings (Vietnamese-aware)."""
    script = (script or "").replace("\r\n", "\n").strip()
    if not script:
        return []
    has_markers = any(_SCENE_MARKER.match(ln) for ln in script.split("\n"))
    if has_markers:
        raw_blocks, current = [], []
        for line in script.split("\n"):
            if _SCENE_MARKER.match(line):
                if current:
                    raw_blocks.append("\n".join(current))
                current = [_strip_marker(line)]
            else:
                current.append(line)
        if current:
            raw_blocks.append("\n".join(current))
    else:
        raw_blocks = [b for b in re.split(r"\n\s*\n", script) if b.strip()]

    texts = []
    for block in raw_blocks:
        sentences = _split_sentences(block)
        if sentences:
            texts.extend(_regroup(sentences))
    if not texts and script:
        texts = _regroup(_split_sentences(script)) or [script]
    return [t for t in texts if t.strip()]


# --------------------------------------------------------------------------- #
# ffmpeg helpers
# --------------------------------------------------------------------------- #
# Match clone-voice's audio assembly constants.
_DECLICK_FADE = 0.015
_NARRATION_GAP = 0.55
_NARRATION_SILENCE_DB = "-50dB"
_NARRATION_EDGE_KEEP = 0.05
# ffmpeg threading: login node is CPU-only, so use every core for both the
# codec and the (large) filter graph.
_FFMPEG_THREADS = ["-threads", "0", "-filter_complex_threads", "0"]

# How many chunks per /synthesize_batch call = the server's true GPU batch size
# (config.OMNI_MAX_BATCH). One HTTP request is then exactly one GPU batch, so
# raising the config raises both the request size and the VRAM the L40S uses.
try:
    _SYNTH_BATCH = max(1, int(config.OMNI_MAX_BATCH))
except (TypeError, ValueError):
    _SYNTH_BATCH = 8


def _run_ffmpeg(args, cwd=None, timeout=3600):
    cmd = [config.ffmpeg_path(), "-y", "-hide_banner", "-nostdin", *args]
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    return proc.returncode, (proc.stdout or "")


def probe_duration(path):
    try:
        out = subprocess.run(
            [config.ffprobe_path(), "-v", "error", "-show_entries",
             "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace")
        return float(json.loads(out.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def _atempo_chain(speed):
    """ffmpeg atempo factors for an arbitrary speed, chained within 0.5–2.0 each
    (atempo preserves pitch — speed up without the chipmunk effect)."""
    speed = float(speed)
    if abs(speed - 1.0) < 1e-3 or speed <= 0:
        return []
    factors, remaining = [], speed
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return [f"atempo={f:.6f}" for f in factors]


def _concat_filters(n, speed=1.0):
    """Filter lines joining n per-chunk inputs SMOOTHLY into one stream ``[voc]``.

    Each chunk: resample → trim leading/trailing silence (edges only) → 15ms
    declick fades at both ends → optional per-chunk speed change → a uniform
    gap pad (except the last). Keeping the speed change PER CHUNK means the
    inter-chunk gap stays a natural fixed length regardless of speed."""
    lead_trim = (f"silenceremove=start_periods=1:start_threshold={_NARRATION_SILENCE_DB}:"
                 f"start_silence={_NARRATION_EDGE_KEEP}:detection=peak")
    tempo = _atempo_chain(speed)
    tempo_str = ("," + ",".join(tempo)) if tempo else ""
    lines, labels = [], []
    last = n - 1
    for k in range(n):
        chain = (
            "aresample=48000,"
            "aformat=sample_rates=48000:channel_layouts=stereo,"
            f"{lead_trim},areverse,{lead_trim},areverse,"
            f"afade=t=in:st=0:d={_DECLICK_FADE},"
            f"areverse,afade=t=in:st=0:d={_DECLICK_FADE},areverse"
            f"{tempo_str},"
            "asetpts=N/SR/TB"
        )
        if k != last:
            chain += f",apad=pad_dur={_NARRATION_GAP}"
        lines.append(f"[{k}:a]{chain}[a{k}]")
        labels.append(f"[a{k}]")
    lines.append(f"{''.join(labels)}concat=n={n}:v=0:a=1[voc]")
    return lines


def _export_mp3(wav_paths, out_path, speed=1.0, denoise=False, on_log=None):
    """Concatenate per-chunk wavs into one de-clicked, loudness-normalised MP3.

    ``denoise`` adds an FFT noise-reduction pass (afftdn) on the joined stream;
    ``speed`` time-stretches each chunk (pitch preserved)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = []
    for w in wav_paths:
        inputs += ["-i", str(w)]

    filters = _concat_filters(len(wav_paths), speed=speed)
    post = ""
    if denoise:
        # afftdn = FFT denoiser; nr=12dB reduction, gentle highpass to kill rumble.
        post = "highpass=f=60,afftdn=nr=12:nf=-25,"
    filters.append(f"[voc]{post}loudnorm=I=-16:TP=-1.5:LRA=11[a]")

    filter_script = out_path.parent / "_voice_filter.txt"
    filter_script.write_text(";".join(filters), encoding="utf-8")
    args = [*inputs, *_FFMPEG_THREADS,
            "-filter_complex_script", str(filter_script.resolve()),
            "-map", "[a]", "-c:a", "libmp3lame", "-b:a", "192k",
            "-ar", "48000", str(out_path.resolve())]
    try:
        rc, log = _run_ffmpeg(args)
    finally:
        filter_script.unlink(missing_ok=True)
    if rc != 0 or not out_path.exists() or out_path.stat().st_size < 256:
        tail = (log or "")[-800:]
        raise RuntimeError(f"ffmpeg ghép audio thất bại: {tail}")
    return out_path


def preprocess_reference(src, dst):
    """Clean a voice-clone reference clip: mono, 24kHz, gentle highpass + loudnorm.
    Falls back to a plain mono/24k conversion if the filter chain fails."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    base = ["-i", str(src), "-ac", "1", "-ar", "24000"]
    rc, _ = _run_ffmpeg(base + ["-af", "highpass=f=70,loudnorm=I=-18:TP=-2:LRA=11",
                                str(dst)], timeout=180)
    if rc == 0 and dst.exists() and dst.stat().st_size > 256:
        return dst
    rc, log = _run_ffmpeg(base + [str(dst)], timeout=180)
    if rc != 0 or not dst.exists():
        raise RuntimeError(f"Không xử lý được audio mẫu: {(log or '')[-200:]}")
    return dst


# --------------------------------------------------------------------------- #
# SRT (ported from clone-voice narrator.write_srt).
# --------------------------------------------------------------------------- #
def _srt_ts(seconds):
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(chunks, srt_path, total_duration, gap=_NARRATION_GAP):
    """Write an SRT whose cues line up with the final MP3 (durations scaled to the
    measured length so trimming/speed changes stay aligned). ``chunks`` is a list
    of (text, duration)."""
    srt_path = Path(srt_path)
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(chunks)
    raw = [max(0.2, float(d or 0.0)) for _t, d in chunks]
    raw_total = sum(raw)
    speech_total = total_duration - gap * (n - 1) if total_duration > 0 else 0.0
    scale = (speech_total / raw_total) if (raw_total > 0 and speech_total > 0) else 1.0
    lines, t = [], 0.0
    for i, (text, _d) in enumerate(chunks):
        dur = raw[i] * scale
        start = t
        speech_end = t + dur
        end = speech_end + gap if i < n - 1 else speech_end
        lines.append(str(i + 1))
        lines.append(f"{_srt_ts(start)} --> {_srt_ts(end)}")
        lines.append(" ".join(text.split()))
        lines.append("")
        t = speech_end + gap
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return srt_path


# --------------------------------------------------------------------------- #
# Cloned-voice profiles (DB-backed; reference stored on the shared FS).
# --------------------------------------------------------------------------- #
def _slug(name):
    s = re.sub(r"[^\w\-]+", "_", (name or "").strip(), flags=re.UNICODE).strip("_")
    return s or "voice"


def list_profiles():
    rows = db.execute("SELECT * FROM voice_profiles ORDER BY created_at DESC",
                      fetch="all") or []
    return [dict(r) for r in rows]


def get_profile(name):
    row = db.execute("SELECT * FROM voice_profiles WHERE name=?", (name,),
                     fetch="one")
    return dict(row) if row else None


def get_profile_by_id(pid):
    row = db.execute("SELECT * FROM voice_profiles WHERE id=?", (pid,), fetch="one")
    return dict(row) if row else None


def create_profile(name, reference_audio, ref_text="", language="vi"):
    """Preprocess the reference clip, store it on the shared FS, save the profile.
    Raises ValueError on bad input / duplicate name."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Tên giọng không được để trống.")
    if get_profile(name):
        raise ValueError(f"Đã có giọng tên “{name}”. Chọn tên khác.")
    src = Path(reference_audio)
    if not src.exists():
        raise ValueError("Không tìm thấy file audio mẫu.")
    pdir = file_utils.safe_join(config.VOICES_DIR, _slug(name))
    os.makedirs(pdir, exist_ok=True)
    stored = Path(pdir) / "reference.wav"
    try:
        preprocess_reference(src, stored)
    except RuntimeError as exc:
        raise ValueError(str(exc))
    pid = db.execute(
        "INSERT INTO voice_profiles (name, ref_audio, ref_text, language, created_at) "
        "VALUES (?,?,?,?,?)",
        (name, str(stored), ref_text or "", language or "vi", db.now()),
        commit=True)
    return pid


def delete_profile(pid):
    prof = get_profile_by_id(pid)
    if not prof:
        return False
    ref = prof.get("ref_audio") or ""
    pdir = os.path.dirname(ref)
    if pdir and file_utils.is_within(config.VOICES_DIR, pdir) and os.path.isdir(pdir):
        import shutil
        shutil.rmtree(pdir, ignore_errors=True)
    db.execute("DELETE FROM voice_profiles WHERE id=?", (pid,), commit=True)
    return True


# --------------------------------------------------------------------------- #
# Jobs: script → MP3 + SRT
# --------------------------------------------------------------------------- #
def normalize_owner(username):
    """The username namespace a job lives in. Blank/whitespace → '' (public).
    Trimmed + capped so two spellings of the same name don't split a library."""
    return (username or "").strip()[:64]


def list_jobs(owner=None):
    """Voice jobs in one namespace. owner blank/None → only PUBLIC jobs (owner='');
    a non-blank owner → only that user's jobs (exact match — wrong name sees none)."""
    owner = normalize_owner(owner)
    if owner:
        rows = db.execute(
            "SELECT * FROM voice_jobs WHERE owner=? ORDER BY created_at DESC LIMIT 100",
            (owner,), fetch="all") or []
    else:
        rows = db.execute(
            "SELECT * FROM voice_jobs WHERE COALESCE(owner,'')='' "
            "ORDER BY created_at DESC LIMIT 100", fetch="all") or []
    return [dict(r) for r in rows]


def list_owners():
    """[(username, job_count)] for every non-public namespace, A→Z (case-insensitive)."""
    rows = db.execute(
        "SELECT owner, COUNT(*) AS n FROM voice_jobs WHERE COALESCE(owner,'')<>'' "
        "GROUP BY owner ORDER BY owner COLLATE NOCASE", fetch="all") or []
    return [(r["owner"], r["n"]) for r in rows]


def get_job(job_id):
    row = db.execute("SELECT * FROM voice_jobs WHERE id=?", (job_id,), fetch="one")
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
    db.execute(f"UPDATE voice_jobs SET {cols} WHERE id=?",
               (*fields.values(), job_id), commit=True)


def start_job(name, text, profile_name="", language="vi", num_step=16, seed=-1,
              speed=1.0, denoise=False, batch=None, instruct="", owner=""):
    """Create a voice_jobs row and kick off the background render thread.
    Validates that a GPU server is ready and the script is non-empty."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Kịch bản trống — hãy nhập nội dung trước.")
    if not voice_service.resolve_endpoint():
        raise ValueError("Chưa có server giọng nói nào sẵn sàng. Hãy khởi động "
                         "một server OmniVoice ở mục bên dưới trước.")
    profile_name = (profile_name or "").strip()
    if profile_name and not get_profile(profile_name):
        raise ValueError(f"Không tìm thấy giọng “{profile_name}”.")
    try:
        num_step = max(1, int(num_step))
        seed = int(seed)
        speed = float(speed)
        batch = max(1, int(batch)) if batch else _SYNTH_BATCH
    except (TypeError, ValueError):
        raise ValueError("Tham số num_step / seed / speed / batch không hợp lệ.")
    if not (0.5 <= speed <= 2.0):
        raise ValueError("Tốc độ phải trong khoảng 0.5–2.0.")
    if not (1 <= batch <= 64):
        raise ValueError("Số đoạn cùng lúc (batch) phải trong khoảng 1–64.")

    # Voice-design instruct (gender/age/pitch/whisper) — only used for the default
    # voice; ignored when a clone profile is chosen (timbre comes from the clip).
    instruct = (instruct or "").strip()
    params = {"profile": profile_name, "language": language or "vi",
              "num_step": num_step, "seed": seed, "speed": speed,
              "denoise": bool(denoise), "batch": batch, "instruct": instruct}
    name = (name or "narration").strip()[:128]
    owner = normalize_owner(owner)
    job_id = db.execute(
        "INSERT INTO voice_jobs (name, owner, status, progress, stage, params, created_at) "
        "VALUES (?, ?, 'queued', 0, ?, ?, ?)",
        (name, owner, "Đang chuẩn bị…", json.dumps(params, ensure_ascii=False), db.now()),
        commit=True)
    job_dir = file_utils.safe_join(config.VOICE_OUTPUTS_DIR, str(job_id))
    os.makedirs(job_dir, exist_ok=True)
    _set_job(job_id, logs_path=job_dir)

    threading.Thread(target=_run_job, args=(job_id, text, params), daemon=True).start()
    return job_id


def _synthesize_chunks(host, port, items, params, timeout=3600):
    """POST a batch synth request to the GPU server. Returns the results list."""
    prof = get_profile(params.get("profile") or "") if params.get("profile") else None
    payload = {
        "items": items,
        "language": params.get("language", "vi"),
        "num_step": params.get("num_step", 16),
        "seed": params.get("seed", -1),
        "ref_audio": (prof or {}).get("ref_audio", "") or "",
        "ref_text": (prof or {}).get("ref_text", "") or "",
        "max_batch": params.get("batch") or _SYNTH_BATCH,
        # Server ignores this when a clone reference is present (ref_audio set).
        "instruct": params.get("instruct", "") or "",
    }
    body = json.dumps(payload).encode("utf-8")
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.request("POST", "/synthesize_batch", body=body,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"Server giọng nói trả lỗi {resp.status}: "
                           f"{raw.decode('utf-8', 'replace')[:300]}")
    data = json.loads(raw or b"{}")
    if not data.get("ok"):
        raise RuntimeError(data.get("error") or "Lỗi không xác định từ server.")
    return data.get("results") or []


def _run_job(job_id, text, params):
    job = get_job(job_id)
    job_dir = job["logs_path"]
    try:
        _set_job(job_id, status="running", stage="Đang chia kịch bản…", progress=2)
        chunks = parse_script(text)
        if not chunks:
            raise RuntimeError("Không tách được nội dung nào để đọc.")
        _job_log(job_dir, f"Chia kịch bản thành {len(chunks)} đoạn.")
        if params.get("instruct") and not params.get("profile"):
            _job_log(job_dir, f"Thiết kế giọng (voice design): {params['instruct']}")

        endpoint = voice_service.resolve_endpoint()
        if not endpoint:
            raise RuntimeError("Server giọng nói không còn sẵn sàng.")
        host, port = endpoint

        # Build the synth request: one item per chunk → one wav per chunk.
        items = []
        for i, t in enumerate(chunks):
            items.append({"text": t,
                          "out_path": os.path.join(job_dir, f"part_{i:04d}.wav")})

        # Render in mini-batches: each group is one /synthesize_batch call sized
        # to the job's batch so it's exactly one GPU batch on the server (fills
        # VRAM, no per-chunk HTTP latency); we advance the progress bar between
        # groups. The server caches the voice prompt process-wide, so splitting
        # into groups costs nothing extra.
        bs = int(params.get("batch") or _SYNTH_BATCH)
        n = len(items)
        results = [None] * n
        done = 0
        for start in range(0, n, bs):
            group = items[start:start + bs]
            _set_job(job_id, stage=f"Đang tổng hợp đoạn {start + 1}–"
                     f"{start + len(group)}/{n} trên GPU…",
                     progress=10 + int(done / n * 65))
            grp_results = _synthesize_chunks(host, port, group, params)
            for k, r in enumerate(grp_results):
                results[start + k] = r
            done += len(group)

        # Pair each successful chunk with its duration; skip failed ones (a single
        # bad chunk must not abort the whole narration).
        voiced = []          # (text, duration, wav_path)
        for i, t in enumerate(chunks):
            r = results[i] if i < len(results) else {}
            wav = (r or {}).get("out_path") or items[i]["out_path"]
            if (r or {}).get("ok") and os.path.exists(wav) and os.path.getsize(wav) > 256:
                voiced.append((t, float(r.get("duration") or 0.0), wav))
            else:
                _job_log(job_dir, f"  ! đoạn {i + 1} lỗi: {(r or {}).get('error', 'no audio')}")
        if not voiced:
            raise RuntimeError(
                "Không tổng hợp được đoạn nào (server trả audio rỗng). Thường do "
                "nội dung không đọc được — ví dụ lặp ký tự vô nghĩa như “Yyyy "
                "Yyyy…”; hãy thử văn bản có nghĩa. Nếu văn bản bình thường mà vẫn "
                "lỗi thì xem log server GPU.")

        _set_job(job_id, stage="Đang ghép & xử lý audio (ffmpeg)…", progress=80)
        out_mp3 = os.path.join(job_dir, f"{_slug(job['name'] or 'narration')}.mp3")
        _export_mp3([w for _t, _d, w in voiced], out_mp3,
                    speed=params.get("speed", 1.0),
                    denoise=bool(params.get("denoise")),
                    on_log=lambda m: _job_log(job_dir, m))

        # SRT scaled to the final measured length.
        srt_path = Path(out_mp3).with_suffix(".srt")
        try:
            total = probe_duration(out_mp3)
            write_srt([(t, d) for t, d, _w in voiced], srt_path, total)
            _job_log(job_dir, f"✓ SRT: {srt_path}")
        except Exception as exc:  # noqa: BLE001 — a bad SRT must not fail the MP3
            _job_log(job_dir, f"  ! không tạo được SRT ({exc})")
            srt_path = None

        _set_job(job_id, status="done", stage="Hoàn tất", progress=100,
                 output_path=out_mp3,
                 srt_path=str(srt_path) if srt_path else None,
                 finished_at=db.now())
        _job_log(job_dir, f"✓ MP3: {out_mp3}")
    except Exception as exc:  # noqa: BLE001
        _job_log(job_dir, f"LỖI: {exc}")
        _set_job(job_id, status="error", stage="Lỗi", error=str(exc),
                 finished_at=db.now())


def delete_job(job_id):
    job = get_job(job_id)
    if not job:
        return False
    path = job.get("logs_path")
    if path and file_utils.is_within(config.VOICE_OUTPUTS_DIR, path) \
            and os.path.isdir(path):
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    db.execute("DELETE FROM voice_jobs WHERE id=?", (job_id,), commit=True)
    return True
