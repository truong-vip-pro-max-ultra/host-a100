"""
Tools / Clone giọng nói — a long-running OmniVoice TTS server on a GPU node.

This is the voice-tool sibling of ``serve_service`` (the API farm). One row in
``voice_servers`` is one OmniVoice HTTP server (``scripts/omnivoice_server.py``)
launched inside a user venv and dispatched to a GPU node via ``sbatch`` — exactly
the same detached, auto-resubmitting lifecycle as the API-farm servers. The
server picks a free port on the compute node and writes ``<node>:<port>`` into
endpoint.json on the shared filesystem; the login-node app POSTs synthesis
requests to ``http://<node>:<port>/synthesize`` (see ``voice_pipeline``).

Status flow:  queued → loading (job RUNNING, weights loading) → ready (/health
reports the model is loaded) → stopped / error.

It reuses ``job_service``'s SLURM helpers so the GPU flags / partition / account
match the rest of the platform.
"""
import http.client
import json
import os
import shlex
import shutil
import subprocess
import threading
import time
import urllib.request

import config
from services import env_service, job_service
from services import storage_service as db
from utils import file_utils

_POLL_SECS = 4
_ENDPOINT_FILE = "endpoint.json"
_SLURM_ID_FILE = "slurm_job_id"
_ACTIVE = ("queued", "loading", "ready")

_monitors = set()
_monitors_lock = threading.Lock()

_QUICK_FAIL_SECS = 60
_QUICK_FAIL_MAX = 3
_RESUBMIT_DELAY = 5


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
def list_servers():
    rows = db.execute(
        """
        SELECT voice_servers.*, envs.name AS env_name
        FROM voice_servers
        LEFT JOIN envs ON voice_servers.env_id = envs.id
        ORDER BY voice_servers.created_at DESC
        """,
        fetch="all",
    ) or []
    return [dict(r) for r in rows]


def get_server(server_id):
    row = db.execute(
        """
        SELECT voice_servers.*, envs.name AS env_name, envs.path AS env_path
        FROM voice_servers
        LEFT JOIN envs ON voice_servers.env_id = envs.id
        WHERE voice_servers.id = ?
        """,
        (server_id,), fetch="one",
    )
    return dict(row) if row else None


def ready_servers():
    rows = db.execute(
        "SELECT * FROM voice_servers WHERE status='ready' ORDER BY created_at",
        fetch="all") or []
    return [dict(r) for r in rows]


def resolve_endpoint():
    """Return (host, port) of a ready OmniVoice server, or None."""
    ready = ready_servers()
    if not ready:
        return None
    s = ready[-1]
    return s["node"], s["port"]


def read_log(server_id):
    s = get_server(server_id)
    if not s or not s.get("logs_path"):
        return ""
    log_file = os.path.join(s["logs_path"], "server.log")
    if not os.path.exists(log_file):
        return ""
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()[-20000:]
    except OSError:
        return ""


# --------------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------------- #
def start_server(name, env_id, model_id="", gpu_model="", time_limit="",
                 auto_resubmit=True):
    """Validate, create the DB row + server dir, start the run-loop thread.
    Returns the new server_id. Raises ValueError on bad input."""
    env = env_service.get_env(env_id)
    if not env:
        raise ValueError("Môi trường đã chọn không tồn tại.")
    py = env_service._python_path(env["path"])
    if not os.path.exists(py):
        raise ValueError("Thiếu trình thông dịch của môi trường trên đĩa.")
    if not os.path.isfile(config.OMNI_SERVER_SCRIPT):
        raise ValueError(
            f"Không tìm thấy script server OmniVoice tại {config.OMNI_SERVER_SCRIPT}.")

    name = (name or "OmniVoice").strip()[:128]
    model_id = (model_id or config.OMNI_MODEL_ID).strip()
    gpu_model = (gpu_model or "").strip()
    time_limit = (time_limit or config.SLURM_TIME or "").strip()

    if not (job_service.slurm_active() and job_service._SBATCH):
        raise ValueError("SLURM/sbatch không khả dụng trên node này — "
                         "không thể cấp GPU cho server giọng nói.")

    server_id = db.execute(
        "INSERT INTO voice_servers (name, env_id, model_id, status, gpu_model, "
        "time_limit, auto_resubmit, created_at) "
        "VALUES (?,?,?, 'queued', ?,?,?,?)",
        (name, env_id, model_id, gpu_model, time_limit,
         1 if auto_resubmit else 0, db.now()),
        commit=True,
    )

    server_dir = file_utils.safe_join(config.VOICE_SERVERS_DIR, str(server_id))
    os.makedirs(server_dir, exist_ok=True)
    db.execute("UPDATE voice_servers SET logs_path=? WHERE id=?",
               (server_dir, server_id), commit=True)

    threading.Thread(target=_run_loop, args=(server_id,), daemon=True).start()
    return server_id


def _build_cmd(server):
    """Rebuild (py, cmd) from a server row — used on first submit AND resubmit."""
    env = env_service.get_env(server["env_id"])
    if not env:
        raise ValueError("Môi trường của server không còn tồn tại.")
    py = env_service._python_path(env["path"])
    if not os.path.exists(py):
        raise ValueError("Thiếu trình thông dịch của môi trường trên đĩa.")
    if not os.path.isfile(config.OMNI_SERVER_SCRIPT):
        raise ValueError(
            f"Không tìm thấy script server OmniVoice tại {config.OMNI_SERVER_SCRIPT}.")
    model_id = (server.get("model_id") or config.OMNI_MODEL_ID).strip()
    cmd = [py, "-u", config.OMNI_SERVER_SCRIPT,
           "--model", model_id,
           # Same GPU process also serves the "Gen video" tool's images (diffusers
           # SDXL, lazy-loaded). "" disables it. The model must be in the shared
           # HF cache (compute nodes have no internet).
           "--image-model", (config.IMAGE_MODEL_ID or "").strip(),
           "--host", "0.0.0.0"]
    return py, cmd


def _append_log(server_dir, text):
    try:
        with open(os.path.join(server_dir, "server.log"), "a",
                  encoding="utf-8", errors="replace") as fh:
            fh.write(text)
    except OSError:
        pass


def _cuda_lib_dirs(py):
    """nvidia-*-cu12 wheel lib dirs inside the env so the compute node can dlopen
    libcudart/libcublas/… without a `module load cuda` (the driver libcuda.so.1
    is already on GPU nodes). Torch usually bundles these as nvidia-* wheels."""
    code = ("import sysconfig, glob, os;"
            "p = sysconfig.get_paths()['purelib'];"
            "print(chr(10).join(glob.glob(os.path.join(p, 'nvidia', '*', 'lib'))))")
    try:
        out = subprocess.run([py, "-c", code], stdout=subprocess.PIPE,
                             text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        return []
    return [d.strip() for d in out.splitlines() if d.strip()]


def _write_batch_script(server_dir, py, cmd):
    """run.sh executed by sbatch on the GPU node: point HF at the shared offline
    cache, make the env's CUDA libs findable, pick a free port, advertise
    <node>:<port> in endpoint.json, then exec the OmniVoice server."""
    script = os.path.join(server_dir, "run.sh")
    endpoint = os.path.join(server_dir, _ENDPOINT_FILE)
    quoted = " ".join(shlex.quote(c) for c in cmd)
    hf_home = shlex.quote(config.OMNI_HF_HOME)
    ld_line = ""
    cuda_dirs = _cuda_lib_dirs(py)
    if cuda_dirs:
        joined = shlex.quote(":".join(cuda_dirs))
        ld_line = (f"export LD_LIBRARY_PATH={joined}"
                   f'"${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"\n')
    body = f"""#!/bin/bash
set -e
export HF_HOME={hf_home}
export HUGGINGFACE_HUB_CACHE={hf_home}/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# True GPU batching knobs (server reads these): fill the L40S by generating up to
# OMNI_MAX_BATCH utterances per call. Tune via config.OMNI_MAX_BATCH.
export OMNI_BATCH={shlex.quote(str(config.OMNI_BATCH_ENABLED))}
export OMNI_MAX_BATCH={shlex.quote(str(config.OMNI_MAX_BATCH))}
# Diffusers image model for the "Gen video" tool (lazy-loaded on first image
# request, shares this GPU). "" via --image-model disables it.
export IMAGE_MODEL={shlex.quote(str(config.IMAGE_MODEL_ID))}
# Let torch/OpenMP actually use all the CPUs SLURM gave us (the CPU-side cost of
# diffusion sampling + tokenize), instead of defaulting to 1 thread.
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-8}}
export MKL_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-8}}
{ld_line}PORT=$({shlex.quote(py)} -c 'import socket;s=socket.socket();s.bind(("0.0.0.0",0));p=s.getsockname()[1];s.close();print(p)')
HOST=${{SLURMD_NODENAME:-$(hostname -s)}}
cat > {shlex.quote(endpoint)} <<EOF
{{"host":"$HOST","port":$PORT}}
EOF
echo "[host-a100] OmniVoice server on $HOST:$PORT"
exec {quoted} --port "$PORT"
"""
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(body)
    try:
        os.chmod(script, 0o755)
    except OSError:
        pass
    return script


# OmniVoice's diffusion sampling + tokenize + multi-GB weight load have real
# CPU-side cost, and many clusters default a job to 1 CPU / a tiny RAM slice,
# which throttles an otherwise-idle L40S. Request a sensible amount unless the
# operator pinned explicit values in config. L40S nodes have plenty of CPUs.
_VOICE_DEFAULT_CPUS = "8"
_VOICE_DEFAULT_MEM = "32G"


def _sbatch_flags(server_id, server_dir, slurm_out, gpu_model, time_limit):
    flags = ["--job-name", f"hosta100-voice-{server_id}",
             "--chdir", server_dir,
             "--export=ALL",
             "--output", slurm_out]
    flags += job_service._gpu_flags(gpu_model)
    if config.SLURM_PARTITION:
        flags += ["--partition", config.SLURM_PARTITION]
    if config.SLURM_ACCOUNT:
        flags += ["--account", config.SLURM_ACCOUNT]
    if time_limit:
        flags += ["--time", time_limit]
    flags += ["--cpus-per-task", config.SLURM_CPUS or _VOICE_DEFAULT_CPUS]
    flags += ["--mem", config.SLURM_MEM or _VOICE_DEFAULT_MEM]
    return flags


def _run_loop(server_id, existing_slurm_id=None):
    """Own one server's whole lifecycle (mirror of serve_service._run_loop)."""
    with _monitors_lock:
        if server_id in _monitors:
            return
        _monitors.add(server_id)
    try:
        quick_fail = 0
        first = True
        while True:
            server = get_server(server_id)
            if not server:
                return
            if server["status"] == "stopped":
                return
            server_dir = server["logs_path"]

            if first and existing_slurm_id:
                slurm_id = existing_slurm_id
            else:
                try:
                    py, cmd = _build_cmd(server)
                except ValueError as exc:
                    _set_status(server_id, "error")
                    _append_log(server_dir, f"\n{exc}\n")
                    _alert(server_id, f"cấu hình lỗi, không khởi động được: {exc}")
                    return
                slurm_id = _submit_once(server_id, server_dir, py, cmd,
                                        server["gpu_model"], server["time_limit"])
                if not slurm_id:
                    _set_status(server_id, "error")
                    _alert(server_id, "sbatch thất bại — không gửi được job lên SLURM")
                    return
            first = False

            outcome, ready_at = _monitor(server_id, slurm_id, server_dir)
            alive = (time.time() - ready_at) if ready_at else 0

            if outcome == "user_stopped":
                return
            if outcome == "ended_ready" and server["auto_resubmit"]:
                quick_fail = quick_fail + 1 if alive < _QUICK_FAIL_SECS else 0
                if quick_fail >= _QUICK_FAIL_MAX:
                    _set_status(server_id, "error")
                    _append_log(server_dir, "\n[auto-resubmit] server chết liên tục "
                                "(<60s) — dừng tự khởi động lại. Kiểm tra log "
                                "(thường hết VRAM hoặc thiếu weights trong HF cache).\n")
                    _alert(server_id, f"chết liên tục <60s sau khi ready "
                           f"{_QUICK_FAIL_MAX} lần — cầu dao auto-resubmit đã ngắt")
                    return
                _append_log(server_dir, f"\n[auto-resubmit] phiên kết thúc, gửi lại "
                            f"sau {_RESUBMIT_DELAY}s…\n")
                db.execute("UPDATE voice_servers SET status='queued', node=NULL, "
                           "port=NULL, slurm_job_id=NULL WHERE id=?",
                           (server_id,), commit=True)
                time.sleep(_RESUBMIT_DELAY)
                continue

            final = "stopped" if outcome == "ended_ready" else "error"
            db.execute("UPDATE voice_servers SET status=?, stopped_at=? WHERE id=?",
                       (final, db.now(), server_id), commit=True)
            _append_log(server_dir, f"\nSLURM job {slurm_id} kết thúc ({final}).\n")
            if final == "error":
                _alert(server_id, "job kết thúc mà server chưa từng phục vụ "
                       "(never ready) — kiểm tra slurm.out/server.log")
            return
    finally:
        with _monitors_lock:
            _monitors.discard(server_id)


def _submit_once(server_id, server_dir, py, cmd, gpu_model, time_limit):
    slurm_out = os.path.join(server_dir, "slurm.out")
    open(slurm_out, "w").close()
    try:
        os.remove(os.path.join(server_dir, _ENDPOINT_FILE))
    except OSError:
        pass
    _set_status(server_id, "queued")

    script = _write_batch_script(server_dir, py, cmd)
    submit = ([job_service._SBATCH, "--parsable"]
              + _sbatch_flags(server_id, server_dir, slurm_out, gpu_model, time_limit)
              + [script])
    _append_log(server_dir, "$ " + " ".join(submit) + "\n\n")
    try:
        res = subprocess.run(submit, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", timeout=60)
    except Exception as exc:  # noqa: BLE001
        _append_log(server_dir, f"sbatch lỗi: {exc}\n")
        return None
    out = (res.stdout or "").strip()
    slurm_id = out.split(";")[0].strip() if out else ""
    if res.returncode != 0 or not slurm_id.isdigit():
        _append_log(server_dir, out + f"\nsbatch thất bại (mã {res.returncode}).\n")
        return None
    try:
        with open(os.path.join(server_dir, _SLURM_ID_FILE), "w") as fh:
            fh.write(slurm_id)
    except OSError:
        pass
    db.execute("UPDATE voice_servers SET slurm_job_id=? WHERE id=?",
               (slurm_id, server_id), commit=True)
    _append_log(server_dir, f"Đã gửi SLURM job {slurm_id}; chờ cấp GPU…\n")
    return slurm_id


def _monitor(server_id, slurm_id, server_dir):
    """Poll squeue, tail slurm.out, discover endpoint, health-check until ready;
    keep watching until the job ends. Returns (outcome, ready_at)."""
    pos = 0
    slurm_out = os.path.join(server_dir, "slurm.out")
    endpoint_path = os.path.join(server_dir, _ENDPOINT_FILE)
    host = port = None
    became_ready = False
    ready_at = None
    unknown = 0
    while True:
        pos = _tail(slurm_out, server_dir, pos)
        cur = get_server(server_id)
        if not cur or cur.get("status") not in _ACTIVE:
            return "user_stopped", ready_at
        state = job_service._squeue_state(slurm_id)
        if state is None:
            break
        if state == "UNKNOWN":
            unknown += 1
            if unknown > 30:
                break
            time.sleep(_POLL_SECS)
            continue
        unknown = 0
        if state in ("PENDING", "CONFIGURING"):
            _set_status(server_id, "queued")
        elif state in ("RUNNING", "COMPLETING"):
            if host is None:
                host, port = _read_endpoint(endpoint_path)
                if host:
                    db.execute("UPDATE voice_servers SET node=?, port=? WHERE id=?",
                               (host, port, server_id), commit=True)
            if host and not became_ready:
                if _health(host, port):
                    became_ready = True
                    ready_at = time.time()
                    _set_status(server_id, "ready")
                    _append_log(server_dir,
                                f"\nSERVER READY: http://{host}:{port}\n")
                else:
                    _set_status(server_id, "loading")
            elif not host:
                _set_status(server_id, "loading")
        time.sleep(_POLL_SECS)

    _tail(slurm_out, server_dir, pos)
    return ("ended_ready" if became_ready else "ended_never_ready"), ready_at


def _tail(slurm_out, server_dir, pos):
    if not os.path.exists(slurm_out):
        return pos
    try:
        with open(slurm_out, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(pos)
            chunk = fh.read()
            pos = fh.tell()
    except OSError:
        return pos
    if chunk:
        _append_log(server_dir, chunk)
    return pos


def _read_endpoint(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("host"), int(data.get("port"))
    except (OSError, ValueError, TypeError):
        return None, None


def _health(host, port):
    """True only when /health reports the model is loaded (ready==true).

    OmniVoice loads multi-GB weights AFTER the HTTP server binds, so an HTTP 200
    alone means "process up, still loading" — we must wait for ready so the first
    synthesis request doesn't race the load."""
    if not host or not port:
        return False
    try:
        conn = http.client.HTTPConnection(host, port, timeout=4)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        if resp.status != 200:
            return False
        data = json.loads(body or b"{}")
        return bool(data.get("ready"))
    except Exception:  # noqa: BLE001
        return False


def _set_status(server_id, status):
    db.execute("UPDATE voice_servers SET status=? WHERE id=?",
               (status, server_id), commit=True)


def _alert(server_id, reason):
    url = (getattr(config, "ALERT_WEBHOOK", "") or "").strip()
    if not url:
        return
    s = get_server(server_id) or {}
    name = s.get("name") or f"#{server_id}"
    msg = f"⚠️ [host-a100] Voice server “{name}” (#{server_id}) lỗi: {reason}"

    def _post():
        try:
            data = json.dumps({"content": msg, "text": msg}).encode("utf-8")
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).close()
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_post, daemon=True).start()


# --------------------------------------------------------------------------- #
# Stop / delete
# --------------------------------------------------------------------------- #
def _scancel(server):
    sid = (server.get("slurm_job_id") or "").strip()
    if sid.isdigit() and shutil.which("scancel"):
        try:
            subprocess.run(["scancel", sid], timeout=20,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001
            pass


def stop_server(server_id):
    server = get_server(server_id)
    if not server:
        return False
    _scancel(server)
    db.execute("UPDATE voice_servers SET status='stopped', stopped_at=? WHERE id=?",
               (db.now(), server_id), commit=True)
    return True


def delete_server(server_id):
    server = get_server(server_id)
    if not server:
        return False
    if server.get("status") in _ACTIVE:
        _scancel(server)
    path = server.get("logs_path")
    if path and file_utils.is_within(config.VOICE_SERVERS_DIR, path) \
            and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    db.execute("DELETE FROM voice_servers WHERE id=?", (server_id,), commit=True)
    return True


# --------------------------------------------------------------------------- #
# Restart reconciliation
# --------------------------------------------------------------------------- #
def resume_monitors():
    """On app startup, re-attach a monitor to any voice server still active."""
    rows = db.execute(
        "SELECT * FROM voice_servers WHERE status IN ('queued','loading','ready') "
        "AND slurm_job_id IS NOT NULL", fetch="all") or []
    for r in rows:
        s = dict(r)
        slurm_id = (s.get("slurm_job_id") or "").strip()
        server_dir = s.get("logs_path")
        if not (slurm_id.isdigit() and server_dir and os.path.isdir(server_dir)):
            continue
        threading.Thread(
            target=_run_loop, args=(s["id"], slurm_id), daemon=True,
        ).start()
