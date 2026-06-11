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
import os
import sys
import tempfile
import uuid

from flask import (Flask, abort, flash, jsonify, redirect, render_template,
                   request, send_file, send_from_directory, session, url_for)

import config
from services import (env_service, job_service, model_service,
                      project_service, pty_service, shell_service,
                      storage_service as db)
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
        slurm_active=job_service.slurm_active(),
        slurm_gres=config.SLURM_GRES,
        slurm_partition=config.SLURM_PARTITION,
        gpu_models=gpu.slurm_gpu_models(config.SLURM_PARTITION or None),
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
        gpu_models=gpu.slurm_gpu_models(config.SLURM_PARTITION or None),
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
            use_gpu=use_gpu, gpu_model=gpu_model,
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
        gpu_models=gpu.slurm_gpu_models(config.SLURM_PARTITION or None),
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
