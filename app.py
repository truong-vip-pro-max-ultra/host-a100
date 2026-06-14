#!/usr/bin/env python3
"""
host-a100 — a lightweight single-process Flask ML platform for an A100 HPC node.

Run with:

    python3 app.py

The app is intentionally single-process. All long-running work (upload finalize,
venv creation, pip installs, inference jobs) is dispatched to daemon threads, so
the Flask request threads stay responsive. Progress is exposed through JSON
polling endpoints that the Bootstrap UI queries on a timer.
"""
import hmac
import http.client
import json
import os
import sys
import tempfile
import threading
import uuid

from flask import (Flask, Response, abort, flash, jsonify, redirect,
                   render_template, request, send_file, send_from_directory,
                   session, stream_with_context, url_for)

import config
from services import (anthropic_bridge, apikey_service, env_service,
                      job_service, model_service, project_service, pty_service,
                      serve_service, shell_service, storage_service as db,
                      voice_pipeline, voice_service)
from utils import file_utils, gpu, progress, sysinfo

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.secret_key = config.SECRET_KEY

# Optional WebSocket support for the interactive (PTY) terminal. If flask-sock
# isn't installed the rest of the app is unaffected — the terminal page just
# shows install instructions instead of the live console.
try:
    from flask_sock import Sock
    sock = Sock(app)
except Exception:  # noqa: BLE001
    sock = None

# Temp area for in-flight uploads, on the same filesystem as /data so the
# background finalize can rename instead of copy.
UPLOAD_TMP = os.path.join(config.DATA_DIR, ".uploads")

_CHUNK = 8 * 1024 * 1024

# Chunked-upload sessions (in-memory; single process, like the progress
# registry). Lets the browser send a large model file as many <100MB pieces so
# it survives a Cloudflare tunnel's 100MB-per-request body cap. The pieces are
# appended in order into one temp file; the existing background finalize then
# moves it into the model dir. Maps upload_id -> session dict.
_chunk_sessions = {}
_chunk_sessions_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Authentication: a single shared password gates everything. Endpoints that must
# stay open: the login page and static assets.
# --------------------------------------------------------------------------- #
_PUBLIC_ENDPOINTS = {"login", "static"}


@app.context_processor
def _inject_auth():
    return {"auth_enabled": config.AUTH_ENABLED}


@app.before_request
def _require_login():
    # The public LLM API (/v1/*) is NOT gated by the session password: external
    # clients authenticate with a Bearer API key instead, enforced inside the
    # proxy view itself. Let it through here regardless of AUTH_ENABLED.
    if request.path == "/v1" or request.path.startswith("/v1/"):
        return
    if not config.AUTH_ENABLED:
        return  # auth disabled (internal/dev) — a startup warning is printed.
    if request.endpoint in _PUBLIC_ENDPOINTS:
        return
    if session.get("auth_ok"):
        return
    # Don't redirect API/JSON polls into an HTML login page; 401 is clearer.
    if request.path.startswith(("/status/", "/upload/", "/terminal/")) \
            or request.path.endswith(".json"):
        return jsonify({"error": "unauthenticated"}), 401
    return redirect(url_for("login", next=request.path))


@app.route("/login", methods=["GET", "POST"])
def login():
    if not config.AUTH_ENABLED or session.get("auth_ok"):
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        supplied = request.form.get("password", "")
        # Constant-time compare to avoid leaking the password via timing.
        if hmac.compare_digest(supplied, config.APP_PASSWORD):
            session["auth_ok"] = True
            session.permanent = True
            dest = request.args.get("next") or url_for("dashboard")
            # Only allow local redirects (no open-redirect to other hosts).
            if not dest.startswith("/"):
                dest = url_for("dashboard")
            return redirect(dest)
        flash("Mật khẩu không đúng.", "danger")
    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("Đã đăng xuất.", "success")
    return redirect(url_for("login"))


# --------------------------------------------------------------------------- #
# Template helpers
# --------------------------------------------------------------------------- #
@app.template_filter("hsize")
def _hsize(num):
    return file_utils.human_size(num)


@app.template_filter("dt")
def _dt(epoch):
    import datetime
    if not epoch:
        return "-"
    return datetime.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")


# Vietnamese labels for the internal status values.
_STATUS_VI = {
    "queued": "đang chờ",
    "running": "đang chạy",
    "done": "hoàn tất",
    "error": "lỗi",
    "unknown": "không rõ",
}


@app.template_filter("vstatus")
def _vstatus(status):
    return _STATUS_VI.get(status, status)


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
def _dashboard_context():
    """Gather all the live dashboard data. Shared by the full page and the
    fragment endpoint the page polls, so both render identical numbers."""
    used, total, free = db.disk_usage()
    gpus = gpu.gpu_summary()
    return dict(
        gpus=gpus,
        # Only meaningful when this host has a local GPU. On the login node it
        # would just run nvidia-smi + the lspci diagnostic every poll for a
        # "not found" message the SLURM per-model table already explains.
        nvidia_smi=gpu.raw_nvidia_smi() if gpus else "",
        models=model_service.list_models(),
        envs=env_service.list_envs(),
        jobs=job_service.list_jobs(),
        active=progress.active(),
        disk={"used": used, "total": total, "free": free,
              "pct": int(used * 100 / total) if total else 0},
        cpu=sysinfo.cpu_info(),
        ram=sysinfo.ram_info(),
        processes=sysinfo.top_processes(),
        slurm_active=job_service.slurm_active(),
        slurm_gres=config.SLURM_GRES,
        slurm_partition=config.SLURM_PARTITION,
        gpu_models=gpu.slurm_gpu_models(config.SLURM_PARTITION or None) or [],
    )


@app.route("/")
def dashboard():
    return render_template("dashboard.html", **_dashboard_context())


@app.route("/dashboard/fragment")
def dashboard_fragment():
    """Just the live part of the dashboard, polled by the page so it refreshes
    in place without a full reload (which would re-run nvidia-smi/sinfo and make
    the page stutter)."""
    return render_template("_dashboard_live.html", **_dashboard_context())


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #
@app.route("/upload", methods=["GET"])
def upload_page():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload_submit():
    """
    Receive the model file (request thread streams it to a temp file) and hand
    off the finalize to a background thread. Returns JSON with an upload_id the
    client polls for server-side progress.
    """
    model_name = request.form.get("model_name", "")
    file = request.files.get("file")

    try:
        model_name = file_utils.validate_name(model_name)
    except file_utils.UnsafeName as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    if file is None or not file.filename:
        return jsonify({"ok": False, "error": "Chưa chọn file."}), 400

    try:
        filename = file_utils.safe_filename(file.filename)
    except file_utils.UnsafeName as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    ext = os.path.splitext(filename)[1].lower()
    if ext and ext not in config.ALLOWED_MODEL_EXTENSIONS:
        return jsonify({
            "ok": False,
            "error": f"Phần mở rộng '{ext}' không được phép.",
        }), 400

    if model_service.name_taken(model_name):
        return jsonify({"ok": False,
                        "error": f"Mô hình '{model_name}' đã tồn tại."}), 400

    upload_id = uuid.uuid4().hex
    progress.init("upload", upload_id, step="receiving")

    os.makedirs(UPLOAD_TMP, exist_ok=True)
    temp_path = os.path.join(UPLOAD_TMP, f"{upload_id}_{filename}")

    # Stream the request body to a temp file on the same fs as /data.
    received = 0
    try:
        with open(temp_path, "wb") as out:
            while True:
                chunk = file.stream.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                received += len(chunk)
    except Exception as exc:  # noqa: BLE001
        progress.update("upload", upload_id, status="error",
                        step="receive failed", append_log=f"ERROR: {exc}\n")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"ok": False, "error": f"Tải lên thất bại: {exc}"}), 500

    progress.update("upload", upload_id, progress=0, step="received",
                    append_log=f"Đã nhận {file_utils.human_size(received)}.\n")
    model_service.start_upload(upload_id, model_name, temp_path, filename)
    return jsonify({"ok": True, "upload_id": upload_id})


# --------------------------------------------------------------------------- #
# Chunked upload. The single-shot /upload above sends the whole file in one POST,
# which a Cloudflare tunnel rejects past 100MB. These three endpoints let the
# browser slice the file into sub-100MB pieces: init validates the name/ext and
# opens a temp file, chunk appends pieces IN ORDER, finalize hands the assembled
# temp file to the same background finalize (model_service.start_upload).
# --------------------------------------------------------------------------- #
@app.route("/upload/init", methods=["POST"])
def upload_init():
    data = request.get_json(silent=True) or {}
    model_name = data.get("model_name", "")
    filename = data.get("filename", "")
    try:
        total_chunks = int(data.get("total_chunks", 0))
        chunk_size = int(data.get("chunk_size", 0))
        size = int(data.get("size", 0))
    except (TypeError, ValueError):
        total_chunks = chunk_size = size = 0

    try:
        model_name = file_utils.validate_name(model_name)
    except file_utils.UnsafeName as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    if not filename:
        return jsonify({"ok": False, "error": "Chưa chọn file."}), 400
    try:
        filename = file_utils.safe_filename(filename)
    except file_utils.UnsafeName as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    ext = os.path.splitext(filename)[1].lower()
    if ext and ext not in config.ALLOWED_MODEL_EXTENSIONS:
        return jsonify({"ok": False,
                        "error": f"Phần mở rộng '{ext}' không được phép."}), 400
    if model_service.name_taken(model_name):
        return jsonify({"ok": False,
                        "error": f"Mô hình '{model_name}' đã tồn tại."}), 400
    if total_chunks < 1 or chunk_size < 1 or size < 1:
        return jsonify({"ok": False, "error": "Tham số mảnh không hợp lệ."}), 400

    upload_id = uuid.uuid4().hex
    os.makedirs(UPLOAD_TMP, exist_ok=True)
    temp_path = os.path.join(UPLOAD_TMP, f"{upload_id}_{filename}")
    open(temp_path, "wb").close()   # create the target; chunks seek+write by offset
    with _chunk_sessions_lock:
        _chunk_sessions[upload_id] = {
            "model_name": model_name, "filename": filename,
            "temp_path": temp_path, "total_chunks": total_chunks,
            "chunk_size": chunk_size, "size": size,
            # Set of fully-written chunk indices. Idempotent: re-sending a chunk
            # overwrites the same byte range, so a retried/duplicated piece can
            # never corrupt the file or double-count.
            "received": set(),
        }
    progress.init("upload", upload_id, step="receiving")
    return jsonify({"ok": True, "upload_id": upload_id})


@app.route("/upload/chunk/<upload_id>", methods=["POST"])
def upload_chunk(upload_id):
    with _chunk_sessions_lock:
        sess = _chunk_sessions.get(upload_id)
    if not sess:
        return jsonify({"ok": False, "error": "Phiên tải lên không tồn tại."}), 404

    index = request.args.get("index", type=int)
    total = sess["total_chunks"]
    if index is None or index < 0 or index >= total:
        return jsonify({"ok": False, "error": "Chỉ số mảnh không hợp lệ."}), 400

    # Each chunk owns a fixed byte range [offset, offset+expected_len). We seek
    # to its offset and write there, so chunks may arrive in ANY order and a
    # retried chunk simply overwrites the same bytes — no corruption, no need
    # for strict sequencing. The raw octet-stream body is read in small reads so
    # a chunk never sits whole in memory.
    chunk_size = sess["chunk_size"]
    offset = index * chunk_size
    expected_len = min(chunk_size, sess["size"] - offset)
    written = 0
    try:
        with open(sess["temp_path"], "r+b") as out:
            out.seek(offset)
            while True:
                buf = request.stream.read(_CHUNK)
                if not buf:
                    break
                out.write(buf)
                written += len(buf)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"Ghi mảnh thất bại: {exc}"}), 500

    # Truncated body (tunnel dropped mid-transfer): do NOT mark received, so the
    # client's retry re-sends the full chunk over the same range.
    if written != expected_len:
        return jsonify({"ok": False,
                        "error": f"Mảnh {index} thiếu byte "
                                 f"({written}/{expected_len}), sẽ gửi lại."}), 422

    with _chunk_sessions_lock:
        sess["received"].add(index)
        received = len(sess["received"])
    # Cap at 99 here; the server-side finalize drives 0->100 afterwards.
    pct = min(99, int(received * 100 / total)) if total else 0
    progress.update("upload", upload_id, progress=pct,
                    step=f"đã nhận mảnh {received}/{total}")
    return jsonify({"ok": True, "received": received, "total": total})


@app.route("/upload/finalize/<upload_id>", methods=["POST"])
def upload_finalize(upload_id):
    with _chunk_sessions_lock:
        sess = _chunk_sessions.pop(upload_id, None)
    if not sess:
        return jsonify({"ok": False, "error": "Phiên tải lên không tồn tại."}), 404
    got, total = len(sess["received"]), sess["total_chunks"]
    if got != total:
        with _chunk_sessions_lock:    # keep the session so the client can resume
            _chunk_sessions[upload_id] = sess
        # Tell the client exactly which indices are still missing so it resends
        # only those, instead of restarting the whole upload.
        missing = sorted(set(range(total)) - sess["received"])
        return jsonify({
            "ok": False,
            "error": f"Thiếu mảnh ({got}/{total}).",
            "missing": missing[:50],
        }), 400

    received = os.path.getsize(sess["temp_path"]) \
        if os.path.exists(sess["temp_path"]) else 0
    if received != sess["size"]:
        with _chunk_sessions_lock:
            _chunk_sessions[upload_id] = sess
        return jsonify({
            "ok": False,
            "error": f"Kích thước sai ({received}/{sess['size']} byte).",
        }), 400
    progress.update("upload", upload_id, progress=0, step="received",
                    append_log=f"Đã nhận {file_utils.human_size(received)}.\n")
    model_service.start_upload(upload_id, sess["model_name"],
                               sess["temp_path"], sess["filename"])
    return jsonify({"ok": True, "upload_id": upload_id})


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@app.route("/models")
def models_page():
    return render_template("models.html", models=model_service.list_models())


@app.route("/models/<int:model_id>/delete", methods=["POST"])
def model_delete(model_id):
    if model_service.delete_model(model_id):
        flash("Đã xoá mô hình.", "success")
    else:
        flash("Không tìm thấy mô hình.", "danger")
    return redirect(url_for("models_page"))


@app.route("/models/<int:model_id>/download")
def model_download(model_id):
    info = model_service.model_file(model_id)
    if not info:
        abort(404)
    directory, filename = info
    if filename is None:
        flash("Mô hình này có nhiều file; hãy tải trực tiếp từ hệ thống file "
              "trên máy chủ.", "warning")
        return redirect(url_for("models_page"))
    # Confinement: directory is always under MODELS_DIR (set at creation).
    if not file_utils.is_within(config.MODELS_DIR, directory):
        abort(403)
    return send_from_directory(directory, filename, as_attachment=True)


# --------------------------------------------------------------------------- #
# Environments
# --------------------------------------------------------------------------- #
@app.route("/envs")
def envs_page():
    envs = env_service.list_envs()
    selected_id = request.args.get("env", type=int)
    # The installed-package list is fetched asynchronously (see packages.json)
    # so opening an env never blocks page render on a slow listing.
    return render_template("envs.html", envs=envs, selected_id=selected_id,
                           whisper_models=env_service.WHISPER_MODELS)


@app.route("/envs/<int:env_id>/packages.json")
def env_packages(env_id):
    return jsonify({"packages": env_service.pip_freeze(env_id)})


@app.route("/envs/create", methods=["POST"])
def env_create():
    name = request.form.get("env_name", "")
    try:
        name = file_utils.validate_name(name)
    except file_utils.UnsafeName as exc:
        flash(str(exc), "danger")
        return redirect(url_for("envs_page"))
    if env_service.name_taken(name):
        flash(f"Môi trường '{name}' đã tồn tại.", "danger")
        return redirect(url_for("envs_page"))
    task_id = uuid.uuid4().hex
    env_service.create_env(task_id, name)
    flash(f"Đang tạo môi trường '{name}'… (tác vụ {task_id[:8]})", "info")
    return redirect(url_for("envs_page", task=task_id))


@app.route("/envs/<int:env_id>/install", methods=["POST"])
def env_install(env_id):
    raw = request.form.get("packages", "")
    try:
        packages = env_service.validate_packages(raw)
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("envs_page", env=env_id))
    task_id = uuid.uuid4().hex
    try:
        env_service.install_packages(task_id, env_id, packages)
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("envs_page", env=env_id))
    flash(f"Đang cài {', '.join(packages)}… (tác vụ {task_id[:8]})", "info")
    return redirect(url_for("envs_page", env=env_id, task=task_id))


@app.route("/envs/<int:env_id>/prefetch-whisper", methods=["POST"])
def env_prefetch_whisper(env_id):
    """Download a faster-whisper model on the login node (with internet) so
    compute-node jobs can load it offline. Streams via the env progress UI."""
    model = request.form.get("whisper_model", "").strip()
    task_id = uuid.uuid4().hex
    try:
        env_service.prefetch_whisper_model(task_id, env_id, model)
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("envs_page", env=env_id))
    flash(f"Đang tải model Whisper '{model}'… (tác vụ {task_id[:8]})", "info")
    return redirect(url_for("envs_page", env=env_id, task=task_id))


@app.route("/envs/<int:env_id>/install-requirements", methods=["POST"])
def env_install_requirements(env_id):
    """Upload a requirements.txt and run `pip install -r` into the env."""
    file = request.files.get("requirements")
    if file is None or not file.filename:
        flash("Chưa chọn file requirements.", "danger")
        return redirect(url_for("envs_page", env=env_id))
    # Stash the uploaded requirements file inside the env directory.
    env = env_service.get_env(env_id)
    if not env:
        flash("Không tìm thấy môi trường.", "danger")
        return redirect(url_for("envs_page"))
    req_dir = os.path.join(env["path"], "_uploaded_requirements")
    os.makedirs(req_dir, exist_ok=True)
    req_path = os.path.join(req_dir, "requirements.txt")
    file.save(req_path)

    task_id = uuid.uuid4().hex
    try:
        env_service.install_requirements_file(task_id, env_id, req_path)
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("envs_page", env=env_id))
    flash(f"Đang cài từ requirements.txt… (tác vụ {task_id[:8]})", "info")
    return redirect(url_for("envs_page", env=env_id, task=task_id))


@app.route("/envs/<int:env_id>/delete", methods=["POST"])
def env_delete(env_id):
    if env_service.delete_env(env_id):
        flash("Đã xoá môi trường.", "success")
    else:
        flash("Không tìm thấy môi trường.", "danger")
    return redirect(url_for("envs_page"))


# --------------------------------------------------------------------------- #
# Projects (user code)
# --------------------------------------------------------------------------- #
@app.route("/projects")
def projects_page():
    projects = project_service.list_projects()
    selected_id = request.args.get("project", type=int)
    selected = None
    files = []
    open_file = request.args.get("file")
    open_content = None
    if selected_id:
        selected = project_service.get_project(selected_id)
        if selected:
            files = project_service.list_files(selected["path"])
            if open_file:
                try:
                    open_content = project_service.read_text_file(
                        selected_id, open_file)
                except ValueError:
                    open_content = None
    return render_template("projects.html", projects=projects,
                           selected=selected, files=files,
                           open_file=open_file, open_content=open_content)


@app.route("/projects/create", methods=["POST"])
def project_create():
    name = request.form.get("project_name", "")
    try:
        name = file_utils.validate_name(name)
    except file_utils.UnsafeName as exc:
        flash(str(exc), "danger")
        return redirect(url_for("projects_page"))
    if project_service.name_taken(name):
        flash(f"Dự án '{name}' đã tồn tại.", "danger")
        return redirect(url_for("projects_page"))
    pid = project_service.create_project(name)
    flash(f"Đã tạo dự án '{name}'.", "success")
    return redirect(url_for("projects_page", project=pid))


@app.route("/projects/<int:project_id>/save-file", methods=["POST"])
def project_save_file(project_id):
    relpath = request.form.get("relpath", "")
    content = request.form.get("content", "")
    set_main = request.form.get("set_main") == "on"
    try:
        rel = project_service.write_text_file(project_id, relpath, content)
        if set_main:
            project_service.set_main_file(project_id, rel)
        flash(f"Đã lưu {rel}.", "success")
    except (ValueError, file_utils.UnsafeName) as exc:
        flash(str(exc), "danger")
    return redirect(url_for("projects_page", project=project_id, file=relpath))


@app.route("/projects/<int:project_id>/upload", methods=["POST"])
def project_upload(project_id):
    files = request.files.getlist("files")
    if not files or all(not f.filename for f in files):
        flash("Chưa chọn file nào.", "danger")
        return redirect(url_for("projects_page", project=project_id))
    saved, errors = 0, []
    for f in files:
        if not f.filename:
            continue
        ext = os.path.splitext(f.filename)[1].lower()
        try:
            if ext in config.ALLOWED_ARCHIVE_EXTENSIONS:
                tmp = os.path.join(config.PROJECTS_DIR,
                                   f".zip_{uuid.uuid4().hex}")
                f.save(tmp)
                extracted = project_service.extract_zip(project_id, tmp)
                saved += len(extracted)
            else:
                project_service.save_uploaded_file(
                    project_id, f.filename, f.stream)
                saved += 1
        except (ValueError, file_utils.UnsafeName) as exc:
            errors.append(f"{f.filename}: {exc}")
    if saved:
        flash(f"Đã thêm {saved} file.", "success")
    for e in errors:
        flash(e, "warning")
    return redirect(url_for("projects_page", project=project_id))


@app.route("/projects/<int:project_id>/set-main", methods=["POST"])
def project_set_main(project_id):
    relpath = request.form.get("main_file", "")
    try:
        rel = project_service.set_main_file(project_id, relpath)
        flash(f"Đã đặt file chính là {rel}.", "success")
    except (ValueError, file_utils.UnsafeName) as exc:
        flash(str(exc), "danger")
    return redirect(url_for("projects_page", project=project_id))


@app.route("/projects/<int:project_id>/file/delete", methods=["POST"])
def project_file_delete(project_id):
    relpath = request.form.get("relpath", "")
    try:
        if project_service.delete_file(project_id, relpath):
            flash(f"Đã xoá {relpath}.", "success")
        else:
            flash("Không tìm thấy file.", "warning")
    except file_utils.UnsafeName as exc:
        flash(str(exc), "danger")
    return redirect(url_for("projects_page", project=project_id))


@app.route("/projects/<int:project_id>/delete", methods=["POST"])
def project_delete(project_id):
    if project_service.delete_project(project_id):
        flash("Đã xoá dự án.", "success")
    else:
        flash("Không tìm thấy dự án.", "danger")
    return redirect(url_for("projects_page"))


@app.route("/projects/<int:project_id>/files.json")
def project_files_json(project_id):
    project = project_service.get_project(project_id)
    if not project:
        return jsonify({"files": [], "main_file": None})
    return jsonify({
        "files": project_service.python_files(project["path"]),
        "main_file": project.get("main_file"),
    })


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
@app.route("/jobs")
def jobs_page():
    return render_template(
        "jobs.html",
        jobs=job_service.list_jobs(),
        models=model_service.list_models(),
        envs=env_service.list_envs(),
        projects=project_service.list_projects(),
        slurm_active=job_service.slurm_active(),
        slurm_gres=config.SLURM_GRES,
        gpu_models=gpu.slurm_gpu_models(config.SLURM_PARTITION or None) or [],
    )


@app.route("/jobs/submit", methods=["POST"])
def job_submit():
    run_mode = request.form.get("run_mode", "runner")
    model_id = request.form.get("model_id", type=int)
    env_id = request.form.get("env_id", type=int)
    project_id = request.form.get("project_id", type=int)
    main_file = request.form.get("main_file", "")
    job_name = request.form.get("job_name", "")
    params_json = request.form.get("params", "")
    # Checkbox: present only when ticked. Default to GPU on (A100 platform).
    use_gpu = request.form.get("use_gpu") is not None
    # Optional GPU-model pin (e.g. "a100"); "" = any GPU of the configured kind.
    gpu_model = request.form.get("gpu_model", "").strip()
    # Checkbox: run on the login node (has internet, no SLURM) instead of a
    # compute node. Lets jobs that fetch at runtime (requests/HF download) work.
    run_local = request.form.get("run_local") is not None

    if not env_id:
        flash("Hãy chọn một môi trường.", "danger")
        return redirect(url_for("jobs_page"))
    if run_mode == "runner" and not model_id:
        flash("Runner mặc định cần một mô hình.", "danger")
        return redirect(url_for("jobs_page"))
    if run_mode == "project" and not project_id:
        flash("Hãy chọn một dự án để chạy.", "danger")
        return redirect(url_for("jobs_page"))

    try:
        job_id = job_service.submit_job(
            model_id, env_id, job_name, params_json,
            run_mode=run_mode, project_id=project_id, main_file=main_file,
            use_gpu=use_gpu, gpu_model=gpu_model, run_local=run_local,
        )
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("jobs_page"))
    flash(f"Đã gửi tác vụ #{job_id}.", "success")
    return redirect(url_for("jobs_page"))


@app.route("/jobs/<int:job_id>/outputs.json")
def job_outputs(job_id):
    files = job_service.list_outputs(job_id)
    for f in files:
        f["size_h"] = file_utils.human_size(f["size"])
    return jsonify({"files": files})


@app.route("/jobs/<int:job_id>/output")
def job_output_download(job_id):
    relpath = request.args.get("path", "")
    resolved = job_service.resolve_output(job_id, relpath)
    if not resolved:
        abort(404)
    out_dir, rel = resolved
    return send_from_directory(out_dir, rel, as_attachment=True)


@app.route("/jobs/<int:job_id>/outputs.zip")
def job_outputs_zip(job_id):
    bundle = job_service.zip_outputs(job_id)
    if not bundle:
        abort(404)
    buf, name = bundle
    return send_file(buf, mimetype="application/zip",
                     as_attachment=True, download_name=name)


@app.route("/jobs/<int:job_id>/delete", methods=["POST"])
def job_delete(job_id):
    if job_service.delete_job(job_id):
        flash("Đã xoá tác vụ.", "success")
    else:
        flash("Không tìm thấy tác vụ.", "danger")
    return redirect(url_for("jobs_page"))


@app.route("/jobs/<int:job_id>/result")
def job_result(job_id):
    job = job_service.get_job(job_id)
    if not job or not job.get("result_path"):
        abort(404)
    rp = job["result_path"]
    if not file_utils.is_within(config.RESULTS_DIR, rp) or not os.path.exists(rp):
        abort(404)
    return send_from_directory(config.RESULTS_DIR, os.path.basename(rp),
                               as_attachment=True)


# --------------------------------------------------------------------------- #
# Tools — a hub of GPU-backed utilities. First tool: Clone giọng nói (OmniVoice
# TTS). A long-running OmniVoice server runs on a GPU node (same lifecycle as the
# API farm); the login node chunks the script, calls the server per chunk, and
# stitches the audio with ffmpeg into one MP3 + SRT.
# --------------------------------------------------------------------------- #
# Audio formats accepted for a voice-clone reference clip (ffmpeg converts them).
_AUDIO_REF_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm",
                   ".aac", ".wma"}


@app.route("/tools")
def tools_page():
    return render_template("tools.html")


def _voice_context():
    return dict(
        servers=voice_service.list_servers(),
        envs=env_service.list_envs(),
        profiles=voice_pipeline.list_profiles(),
        jobs=voice_pipeline.list_jobs(),
        gpu_models=gpu.slurm_gpu_models(config.SLURM_PARTITION or None) or [],
        slurm_active=job_service.slurm_active(),
        omni_model=config.OMNI_MODEL_ID,
        ffmpeg_ok=config.ffmpeg_available(),
    )


@app.route("/tools/clone-voice")
def tool_clone_voice():
    return render_template("tool_clone_voice.html", **_voice_context())


@app.route("/tools/voice/servers/start", methods=["POST"])
def voice_server_start():
    f = request.form
    try:
        voice_service.start_server(
            name=f.get("name", ""),
            env_id=f.get("env_id", type=int),
            model_id=f.get("model_id", "").strip(),
            gpu_model=f.get("gpu_model", "").strip(),
            time_limit=f.get("time_limit", "").strip(),
            auto_resubmit=f.get("auto_resubmit") is not None,
        )
        flash("Đã gửi server giọng nói tới SLURM, đang chờ cấp GPU…", "success")
    except ValueError as exc:
        flash(str(exc), "danger")
    return redirect(url_for("tool_clone_voice"))


@app.route("/tools/voice/servers/<int:server_id>/stop", methods=["POST"])
def voice_server_stop(server_id):
    if voice_service.stop_server(server_id):
        flash("Đã dừng server giọng nói.", "success")
    else:
        flash("Không tìm thấy server.", "danger")
    return redirect(url_for("tool_clone_voice"))


@app.route("/tools/voice/servers/<int:server_id>/delete", methods=["POST"])
def voice_server_delete(server_id):
    if voice_service.delete_server(server_id):
        flash("Đã xoá server giọng nói.", "success")
    else:
        flash("Không tìm thấy server.", "danger")
    return redirect(url_for("tool_clone_voice"))


@app.route("/tools/voice/servers.json")
def voice_servers_json():
    out = []
    for s in voice_service.list_servers():
        out.append({
            "id": s["id"], "name": s["name"], "status": s["status"],
            "node": s.get("node"), "port": s.get("port"),
            "env_name": s.get("env_name"),
        })
    return jsonify({"servers": out})


@app.route("/tools/voice/servers/<int:server_id>/log")
def voice_server_log(server_id):
    return jsonify({"log": voice_service.read_log(server_id)})


@app.route("/tools/voice/profiles/create", methods=["POST"])
def voice_profile_create():
    name = request.form.get("name", "").strip()
    ref_text = request.form.get("ref_text", "").strip()
    language = request.form.get("language", "vi").strip() or "vi"
    f = request.files.get("reference")
    if not f or not f.filename:
        flash("Hãy chọn một file ghi âm mẫu (5–15 giây).", "danger")
        return redirect(url_for("tool_clone_voice"))
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in _AUDIO_REF_EXTS:
        flash(f"Định dạng audio không hỗ trợ ({ext}). Dùng wav/mp3/m4a/flac/ogg…",
              "danger")
        return redirect(url_for("tool_clone_voice"))
    config.ensure_dirs()
    tmp = os.path.join(config.VOICES_DIR, f".upload_{uuid.uuid4().hex}{ext}")
    try:
        f.save(tmp)
        voice_pipeline.create_profile(name, tmp, ref_text=ref_text, language=language)
        flash(f"Đã tạo giọng “{name}”.", "success")
    except ValueError as exc:
        flash(str(exc), "danger")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return redirect(url_for("tool_clone_voice"))


@app.route("/tools/voice/profiles/<int:profile_id>/delete", methods=["POST"])
def voice_profile_delete(profile_id):
    if voice_pipeline.delete_profile(profile_id):
        flash("Đã xoá giọng.", "success")
    else:
        flash("Không tìm thấy giọng.", "danger")
    return redirect(url_for("tool_clone_voice"))


@app.route("/tools/voice/jobs/submit", methods=["POST"])
def voice_job_submit():
    f = request.form
    try:
        voice_pipeline.start_job(
            name=f.get("name", ""),
            text=f.get("text", ""),
            profile_name=f.get("profile", "").strip(),
            language=f.get("language", "vi").strip() or "vi",
            num_step=f.get("num_step", "16"),
            seed=f.get("seed", "-1"),
            speed=f.get("speed", "1.0"),
            denoise=f.get("denoise") is not None,
        )
        flash("Đã bắt đầu tạo giọng đọc — theo dõi tiến trình bên dưới.", "success")
    except ValueError as exc:
        flash(str(exc), "danger")
    return redirect(url_for("tool_clone_voice"))


@app.route("/tools/voice/jobs.json")
def voice_jobs_json():
    out = []
    for j in voice_pipeline.list_jobs():
        out.append({
            "id": j["id"], "name": j["name"], "status": j["status"],
            "progress": j["progress"], "stage": j.get("stage"),
            "error": j.get("error"),
            "has_mp3": bool(j.get("output_path")),
            "has_srt": bool(j.get("srt_path")),
        })
    return jsonify({"jobs": out})


def _voice_job_file(job_id, kind):
    job = voice_pipeline.get_job(job_id)
    if not job:
        abort(404)
    path = job.get("output_path") if kind == "mp3" else job.get("srt_path")
    if not path or not file_utils.is_within(config.VOICE_OUTPUTS_DIR, path) \
            or not os.path.exists(path):
        abort(404)
    return send_from_directory(os.path.dirname(path), os.path.basename(path),
                               as_attachment=True)


@app.route("/tools/voice/jobs/<int:job_id>/mp3")
def voice_job_mp3(job_id):
    return _voice_job_file(job_id, "mp3")


@app.route("/tools/voice/jobs/<int:job_id>/srt")
def voice_job_srt(job_id):
    return _voice_job_file(job_id, "srt")


@app.route("/tools/voice/jobs/<int:job_id>/log")
def voice_job_log(job_id):
    job = voice_pipeline.get_job(job_id)
    if not job or not job.get("logs_path"):
        return jsonify({"log": ""})
    log_file = os.path.join(job["logs_path"], "job.log")
    if not os.path.exists(log_file):
        return jsonify({"log": ""})
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as fh:
            return jsonify({"log": fh.read()[-20000:]})
    except OSError:
        return jsonify({"log": ""})


@app.route("/tools/voice/jobs/<int:job_id>/delete", methods=["POST"])
def voice_job_delete(job_id):
    if voice_pipeline.delete_job(job_id):
        flash("Đã xoá tác vụ giọng đọc.", "success")
    else:
        flash("Không tìm thấy tác vụ.", "danger")
    return redirect(url_for("tool_clone_voice"))


# --------------------------------------------------------------------------- #
# API farm — manage long-running LLM servers and the keys that gate the public
# /v1/* proxy. These management pages stay behind the session login (owner UI);
# only the /v1/* proxy below is opened to Bearer-key clients.
# --------------------------------------------------------------------------- #
def _api_context():
    # Build the PUBLIC base URL the way an external client must use it. We sit
    # behind the Cloudflare tunnel, which terminates HTTPS and calls the app over
    # plain http — so request.host_url/request.scheme say "http" even though the
    # public URL is https. Honour the tunnel's X-Forwarded-Proto/Host headers so
    # the copy-paste base_url shows https (a http base_url triggers a 301 that
    # downgrades the client's POST to GET upstream -> 405).
    proto = request.headers.get("X-Forwarded-Proto", request.scheme)
    host = request.headers.get("X-Forwarded-Host", request.host)
    base = f"{proto}://{host}/v1"
    return dict(
        servers=serve_service.list_servers(),
        models=model_service.list_models(),
        envs=env_service.list_envs(),
        keys=[{**k, "masked": apikey_service.mask(k["key"])}
              for k in apikey_service.list_keys()],
        gpu_models=gpu.slurm_gpu_models(config.SLURM_PARTITION or None) or [],
        slurm_active=job_service.slurm_active(),
        base_url=base,
        new_key=session.pop("_new_api_key", None),
    )


@app.route("/api")
def api_page():
    return render_template("api.html", **_api_context())


@app.route("/api/servers/start", methods=["POST"])
def api_server_start():
    f = request.form
    try:
        serve_service.start_server(
            name=f.get("name", ""),
            model_id=f.get("model_id", type=int),
            env_id=f.get("env_id", type=int),
            served_name=f.get("served_name", ""),
            engine=f.get("engine", "llamacpp").strip(),
            gpu_model=f.get("gpu_model", "").strip(),
            n_gpu_layers=f.get("n_gpu_layers", "99"),
            n_ctx=f.get("n_ctx", "8192"),
            chat_format=f.get("chat_format", "").strip(),
            extra_args=f.get("extra_args", ""),
            time_limit=f.get("time_limit", "").strip(),
            auto_resubmit=f.get("auto_resubmit") is not None,
        )
        flash("Đã gửi server tới SLURM, đang chờ cấp GPU…", "success")
    except ValueError as exc:
        flash(str(exc), "danger")
    return redirect(url_for("api_page"))


@app.route("/api/servers/<int:server_id>/stop", methods=["POST"])
def api_server_stop(server_id):
    if serve_service.stop_server(server_id):
        flash("Đã dừng server.", "success")
    else:
        flash("Không tìm thấy server.", "danger")
    return redirect(url_for("api_page"))


@app.route("/api/servers/<int:server_id>/delete", methods=["POST"])
def api_server_delete(server_id):
    if serve_service.delete_server(server_id):
        flash("Đã xoá server.", "success")
    else:
        flash("Không tìm thấy server.", "danger")
    return redirect(url_for("api_page"))


@app.route("/api/servers.json")
def api_servers_json():
    out = []
    for s in serve_service.list_servers():
        out.append({
            "id": s["id"], "name": s["name"], "served_name": s["served_name"],
            "status": s["status"], "node": s.get("node"), "port": s.get("port"),
            "model_name": s.get("model_name"), "env_name": s.get("env_name"),
        })
    return jsonify({"servers": out})


@app.route("/api/servers/<int:server_id>/log")
def api_server_log(server_id):
    return jsonify({"log": serve_service.read_log(server_id)})


@app.route("/api/keys/create", methods=["POST"])
def api_key_create():
    token = apikey_service.create_key(request.form.get("label", ""))
    # Stash the plaintext so the page can show it ONCE after the redirect.
    session["_new_api_key"] = token
    flash("Đã tạo API key mới — sao chép ngay, nó chỉ hiện một lần.", "success")
    return redirect(url_for("api_page"))


@app.route("/api/keys/<int:key_id>/delete", methods=["POST"])
def api_key_delete(key_id):
    apikey_service.delete_key(key_id)
    flash("Đã xoá API key.", "success")
    return redirect(url_for("api_page"))


# --------------------------------------------------------------------------- #
# Public OpenAI-compatible proxy. Forwards /v1/* to whichever GPU server is
# ready, after checking the Bearer API key. Supports SSE streaming so coding
# clients (Claude Code, Cline, the OpenAI SDK) get live token output.
# --------------------------------------------------------------------------- #
_HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate",
               "proxy-authorization", "te", "trailers", "transfer-encoding",
               "upgrade", "content-length", "host", "content-encoding"}


def _cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Authorization, Content-Type, x-api-key",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    }


def _check_api_key():
    auth = request.headers.get("Authorization", "")
    token = auth[7:].strip() if auth.startswith("Bearer ") else auth.strip()
    if not token:
        token = request.headers.get("x-api-key", "").strip()
    return apikey_service.verify(token)


def _api_error(message, status, etype="invalid_request_error"):
    rv = jsonify({"error": {"message": message, "type": etype}})
    for k, v in _cors_headers().items():
        rv.headers[k] = v
    return rv, status


# --------------------------------------------------------------------------- #
# Chat — a ChatGPT-style conversation UI over the ready GPU server. Owner page
# (session-gated), so the browser talks to this endpoint with the login cookie
# instead of a Bearer key; we forward to the upstream OpenAI server and relay the
# token stream. History lives in the browser (localStorage), never the DB.
# --------------------------------------------------------------------------- #
@app.route("/chat")
def chat_page():
    ready = serve_service.ready_servers()
    return render_template(
        "chat.html",
        servers=[{"served_name": s["served_name"], "name": s["name"]}
                 for s in ready],
    )


@app.route("/chat/send", methods=["POST"])
def chat_send():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "messages trống."}), 400
    endpoint = serve_service.resolve_endpoint(data.get("model"))
    if not endpoint:
        return jsonify({"error": "Chưa có server nào sẵn sàng. Hãy khởi động "
                        "một server ở tab API farm."}), 503
    host, port, served = endpoint
    payload = {"model": served, "messages": messages, "stream": True}
    body = json.dumps(payload).encode("utf-8")
    try:
        conn = http.client.HTTPConnection(host, port, timeout=900)
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Không kết nối được server GPU "
                        f"({host}:{port}): {exc}"}), 502
    if resp.status != 200:
        detail = resp.read().decode("utf-8", "replace")
        conn.close()
        return jsonify({"error": f"Server GPU trả lỗi {resp.status}: {detail}"}), 502

    def generate():
        try:
            for line in resp:
                yield line
        finally:
            conn.close()
    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# --- Anthropic Messages bridge (so Claude Code can point straight here) ----- #
# Claude Code speaks the Anthropic /v1/messages API and authenticates with the
# x-api-key header. We translate its request to OpenAI chat-completions, forward
# to the ready GPU server, and translate the (streaming or not) reply back to
# Anthropic's shape. These static routes win over the /v1/<path> catch-all.
@app.route("/v1/messages", methods=["POST", "OPTIONS"])
def anthropic_messages():
    if request.method == "OPTIONS":
        return ("", 204, _cors_headers())
    if not _check_api_key():
        return _api_error("Sai hoặc thiếu API key.", 401)
    try:
        a = json.loads(request.get_data() or b"{}")
    except ValueError:
        return _api_error("Body không phải JSON hợp lệ.", 400)

    endpoint = serve_service.resolve_endpoint(a.get("model"))
    if not endpoint:
        return _api_error("Chưa có server nào sẵn sàng. Hãy khởi động một "
                          "server ở tab 'API farm'.", 503, "server_error")
    host, port, served = endpoint

    want_stream = bool(a.get("stream"))
    # The native llama-server engine parses Qwen's tool calls into real OpenAI
    # `tool_calls` (forwarded as proper deltas in stream mode), so a tool-bearing
    # request streams token-by-token like any other — no need to buffer the whole
    # reply to recover text `<tool_call>` blocks. Tokens flow live, so the tunnel
    # never goes silent (no Error 524). Only a non-streaming CLIENT gets a
    # blocking read + JSON reply. tool_types is still used by the non-stream path
    # as a fallback for a server that emits Qwen tool calls as text.
    tool_types = anthropic_bridge.tool_param_types(a.get("tools"))

    oai = anthropic_bridge.to_openai_request(a, served)
    if want_stream:
        oai["stream"] = True
        oai.setdefault("stream_options", {"include_usage": True})
    body = json.dumps(oai).encode("utf-8")
    input_est = anthropic_bridge.estimate_tokens(a)

    try:
        conn = http.client.HTTPConnection(host, port, timeout=900)
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
    except Exception as exc:  # noqa: BLE001
        return _api_error(f"Không kết nối được tới server GPU ({host}:{port}): "
                          f"{exc}", 502, "server_error")

    if resp.status != 200:
        detail = resp.read().decode("utf-8", "replace")
        conn.close()
        return _api_error(f"Server GPU trả lỗi {resp.status}: {detail}",
                          resp.status if resp.status >= 400 else 502, "server_error")

    if want_stream:
        # Live token stream; native tool_calls arrive as input_json_delta deltas.
        gen = anthropic_bridge.stream(resp, conn, served, input_est)
        rv = Response(stream_with_context(gen), status=200,
                      mimetype="text/event-stream")
        for k, v in _cors_headers().items():
            rv.headers[k] = v
        return rv

    # Non-streaming client → blocking read + one JSON reply (upstream non-stream).
    data = resp.read()
    conn.close()
    try:
        oai_resp = json.loads(data)
    except ValueError:
        return _api_error("Server GPU trả về JSON không hợp lệ.", 502, "server_error")
    message = anthropic_bridge.openai_response_to_anthropic(oai_resp, served,
                                                            tool_types)
    rv = jsonify(message)
    for k, v in _cors_headers().items():
        rv.headers[k] = v
    return rv


@app.route("/v1/messages/count_tokens", methods=["POST", "OPTIONS"])
def anthropic_count_tokens():
    if request.method == "OPTIONS":
        return ("", 204, _cors_headers())
    if not _check_api_key():
        return _api_error("Sai hoặc thiếu API key.", 401)
    try:
        a = json.loads(request.get_data() or b"{}")
    except ValueError:
        return _api_error("Body không phải JSON hợp lệ.", 400)
    rv = jsonify({"input_tokens": anthropic_bridge.estimate_tokens(a)})
    for k, v in _cors_headers().items():
        rv.headers[k] = v
    return rv


@app.route("/v1/<path:subpath>", methods=["GET", "POST", "OPTIONS"])
def api_proxy(subpath):
    if request.method == "OPTIONS":
        return ("", 204, _cors_headers())
    if not _check_api_key():
        return _api_error("Sai hoặc thiếu API key.", 401)

    # GET /v1/models: aggregate every ready server's served name, so a client can
    # discover all available models even though each upstream only serves one.
    if subpath == "models" and request.method == "GET":
        data = [{"id": s["served_name"], "object": "model", "owned_by": "host-a100"}
                for s in serve_service.ready_servers()]
        rv = jsonify({"object": "list", "data": data})
        for k, v in _cors_headers().items():
            rv.headers[k] = v
        return rv

    body = request.get_data()
    model_name = None
    if body:
        try:
            model_name = (json.loads(body) or {}).get("model")
        except (ValueError, TypeError):
            model_name = None

    endpoint = serve_service.resolve_endpoint(model_name)
    if not endpoint:
        return _api_error("Chưa có server nào sẵn sàng. Hãy khởi động một "
                          "server ở tab 'API farm'.", 503, "server_error")
    host, port, _served = endpoint
    return _proxy_to(host, port, "/v1/" + subpath, body)


def _proxy_to(host, port, path, body):
    fwd = {}
    for k, v in request.headers.items():
        lk = k.lower()
        # Drop the client's Authorization (upstream llama.cpp has no auth) and
        # all hop-by-hop headers; forward the rest (Content-Type, Accept, …).
        if lk in _HOP_BY_HOP or lk == "authorization":
            continue
        fwd[k] = v
    try:
        conn = http.client.HTTPConnection(host, port, timeout=900)
        conn.request(request.method, path, body=body, headers=fwd)
        resp = conn.getresponse()
    except Exception as exc:  # noqa: BLE001
        return _api_error(f"Không kết nối được tới server GPU ({host}:{port}): "
                          f"{exc}", 502, "server_error")

    out_headers = {k: v for k, v in resp.getheaders()
                   if k.lower() not in _HOP_BY_HOP}
    out_headers.update(_cors_headers())
    ctype = resp.getheader("Content-Type", "") or ""

    # Streaming completions arrive as Server-Sent Events; relay line-by-line so
    # tokens reach the client live. Non-streaming responses are read whole.
    if "text/event-stream" in ctype:
        def generate():
            try:
                for line in resp:
                    yield line
            finally:
                conn.close()
        rv = Response(stream_with_context(generate()), status=resp.status)
        for k, v in out_headers.items():
            rv.headers[k] = v
        return rv

    data = resp.read()
    conn.close()
    rv = Response(data, status=resp.status)
    for k, v in out_headers.items():
        rv.headers[k] = v
    return rv


# --------------------------------------------------------------------------- #
# Terminal (SSH-like command console). Runs real commands on the host for the
# authenticated owner — see services/shell_service.py for the trust model.
# --------------------------------------------------------------------------- #
@app.route("/terminal")
def terminal_page():
    # A fresh page load (or reload) starts back at the initial directory — a
    # reload shouldn't inherit a cwd left over from earlier `cd`s. The cwd then
    # persists across commands within the session until the next reload.
    cwd = shell_service.initial_cwd()
    session["term_cwd"] = cwd
    return render_template(
        "terminal.html",
        cwd=cwd,
        slurm_active=job_service.slurm_active(),
        slurm_gres=config.SLURM_GRES,
        gpu_models=gpu.slurm_gpu_models(config.SLURM_PARTITION or None) or [],
        pty_enabled=bool(sock) and pty_service.available(),
    )


if sock is not None:
    @sock.route("/terminal/pty")
    def terminal_pty(ws):
        """Interactive PTY shell over a WebSocket (vim/nano/htop/REPL work).

        Auth is enforced by the same before_request gate as /terminal/* (an
        unauthenticated handshake gets a 401 and never upgrades); we re-check
        here as defence in depth. Query params choose where the shell runs.
        """
        if config.AUTH_ENABLED and not session.get("auth_ok"):
            return
        on_compute = request.args.get("on_compute") == "1"
        use_gpu = request.args.get("use_gpu") == "1"
        gpu_model = (request.args.get("gpu_model") or "").strip()
        pty_service.open_session(ws, on_compute=on_compute, use_gpu=use_gpu,
                                 gpu_model=gpu_model)


@app.route("/terminal/run", methods=["POST"])
def terminal_run():
    data = request.get_json(silent=True) or {}
    command = data.get("command", "")
    on_compute = bool(data.get("on_compute"))
    use_gpu = bool(data.get("use_gpu"))
    gpu_model = (data.get("gpu_model") or "").strip()
    cwd = session.get("term_cwd") or shell_service.initial_cwd()
    result = shell_service.run_command(command, cwd, use_gpu=use_gpu,
                                       on_compute=on_compute, gpu_model=gpu_model)
    # Persist the (possibly changed) working directory so `cd` sticks.
    session["term_cwd"] = result["cwd"]
    return jsonify(result)


@app.route("/terminal/complete", methods=["POST"])
def terminal_complete():
    data = request.get_json(silent=True) or {}
    cwd = session.get("term_cwd") or shell_service.initial_cwd()
    return jsonify(shell_service.complete(data.get("text", ""), cwd))


# --------------------------------------------------------------------------- #
# Progress polling endpoints
# --------------------------------------------------------------------------- #
@app.route("/status/upload/<upload_id>")
def status_upload(upload_id):
    return jsonify(progress.get("upload", upload_id))


@app.route("/status/env/<task_id>")
def status_env(task_id):
    return jsonify(progress.get("env", task_id))


@app.route("/status/job/<int:job_id>")
def status_job(job_id):
    snap = progress.get("job", job_id)
    snap["log"] = job_service.read_log(job_id) or snap.get("log", "")
    # The in-memory registry only has live jobs from the current process; for a
    # job that finished before the last app restart it is empty, so fall back to
    # the persisted row (status/progress) — otherwise the dashboard popup would
    # show "không rõ / 0%" for a job the table lists as done.
    if not snap.get("status") or snap.get("status") == "unknown":
        job = job_service.get_job(job_id)
        if job:
            snap["status"] = job.get("status") or snap.get("status")
            if not snap.get("progress"):
                snap["progress"] = job.get("progress") or 0
            # "unknown" is the registry's placeholder step — drop it so the popup
            # doesn't show a literal "unknown" for a finished historical job.
            if not snap.get("step") or snap.get("step") == "unknown":
                snap["step"] = ""
    return jsonify(snap)


# Alias to match the spec's /upload/progress/<id> path.
@app.route("/upload/progress/<upload_id>")
def upload_progress(upload_id):
    return jsonify(progress.get("upload", upload_id))


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"ok": False, "error": "File vượt quá dung lượng tối đa "
                    "cho phép."}), 413


def main():
    # Some HPC nodes run a latin-1 locale; make console output UTF-8 safe so a
    # stray non-ASCII print can't crash startup.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    config.ensure_dirs()
    db.init_db()
    os.makedirs(UPLOAD_TMP, exist_ok=True)
    # API farm: make sure a Bearer key exists, and re-attach monitors to any
    # LLM server still alive on SLURM from before this (re)start.
    new_key = apikey_service.ensure_default_key()
    if new_key:
        print(f"[host-a100] API farm key (xem ở tab API farm): {new_key}")
    serve_service.resume_monitors()
    # Tools / Clone giọng nói: re-attach monitors to any OmniVoice server still
    # alive on SLURM from before this (re)start (same reconciliation as above).
    voice_service.resume_monitors()
    # ASCII-only console logs: some HPC nodes use a latin-1 locale and would
    # crash on a non-ASCII print at startup.
    if config.AUTH_ENABLED:
        print("[host-a100] Password login: ENABLED.")
    else:
        print("[host-a100] WARNING: no password set (HOSTA100_PASSWORD / "
              "config.DEFAULT_PASSWORD). Login is DISABLED -- do NOT expose "
              "publicly (cloudflare/ngrok) until you set one, because 'run "
              "project code' executes arbitrary Python under your account.")
    # threaded=True so the upload POST and the polling GETs run concurrently in
    # this single process. debug/reloader OFF to keep one stable process with
    # its background threads intact.
    app.run(host=config.HOST, port=config.PORT, threaded=True,
            debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
