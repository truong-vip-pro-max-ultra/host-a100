"""
API farm — long-running OpenAI-compatible LLM servers on GPU compute nodes.

Each "server" is a llama.cpp OpenAI server (`python -m llama_cpp.server`) launched
inside a user venv and dispatched to a GPU node via `sbatch` (detached, so it
survives an app restart — exactly like the run jobs). The server picks a free
port ON the compute node and writes its `<node>:<port>` into endpoint.json on the
shared filesystem; the Flask app on the login node reads that and reverse-proxies
the public /v1/* traffic to it (see the proxy in app.py).

Lifecycle / status:
    queued   — sbatch submitted, waiting for SLURM to grant a GPU
    loading  — job is RUNNING; llama.cpp is loading the model into VRAM
    ready    — health check on /v1/models passes; the endpoint serves requests
    stopped  — user stopped it (scancel) or its SLURM time limit expired
    error    — sbatch failed, or the job died before becoming ready

This reuses job_service's SLURM helpers (gpu flags, partition/account, sbatch
discovery) so the dispatch options match the rest of the platform.
"""
import http.client
import json
import os
import shlex
import shutil
import subprocess
import threading
import time

import config
from services import env_service, job_service, model_service
from services import storage_service as db
from utils import file_utils

_POLL_SECS = 4
_ENDPOINT_FILE = "endpoint.json"
_SLURM_ID_FILE = "slurm_job_id"

# Status values that mean "this server may still be alive on SLURM" — used both
# for the proxy (only `ready` accepts traffic) and for restart reconciliation.
_ACTIVE = ("queued", "loading", "ready")

# Guards the per-server monitor threads so we never start two for one server
# (e.g. start_server + a resume on restart racing).
_monitors = set()
_monitors_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
def list_servers():
    rows = db.execute(
        """
        SELECT servers.*, models.name AS model_name, envs.name AS env_name
        FROM servers
        LEFT JOIN models ON servers.model_id = models.id
        LEFT JOIN envs   ON servers.env_id   = envs.id
        ORDER BY servers.created_at DESC
        """,
        fetch="all",
    ) or []
    return [dict(r) for r in rows]


def get_server(server_id):
    row = db.execute(
        """
        SELECT servers.*, models.name AS model_name, models.path AS model_path,
               envs.name AS env_name, envs.path AS env_path
        FROM servers
        LEFT JOIN models ON servers.model_id = models.id
        LEFT JOIN envs   ON servers.env_id   = envs.id
        WHERE servers.id = ?
        """,
        (server_id,), fetch="one",
    )
    return dict(row) if row else None


def ready_servers():
    rows = db.execute(
        "SELECT * FROM servers WHERE status='ready' ORDER BY created_at",
        fetch="all") or []
    return [dict(r) for r in rows]


def resolve_endpoint(model_name=None):
    """Pick a ready server to forward a /v1 request to.

    If `model_name` matches a ready server's served_name, route there; otherwise
    fall back to the most recently created ready server (so a client using a
    default model id still works when only one server is up). Returns
    (host, port, served_name) or None when nothing is ready.
    """
    ready = ready_servers()
    if not ready:
        return None
    if model_name:
        for s in ready:
            if s["served_name"] == model_name:
                return s["node"], s["port"], s["served_name"]
    s = ready[-1]
    return s["node"], s["port"], s["served_name"]


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
def _resolve_gguf(model):
    """Return the .gguf file path for a model row, or raise ValueError.

    The platform stores a model as either a single file or a directory; for
    llama.cpp we need a .gguf weights file. Accept a direct .gguf path, or the
    first *.gguf found inside a model directory.
    """
    path = model.get("path") or ""
    if os.path.isfile(path) and path.lower().endswith(".gguf"):
        return path
    if os.path.isdir(path):
        for root, _dirs, names in os.walk(path):
            for n in sorted(names):
                if n.lower().endswith(".gguf"):
                    return os.path.join(root, n)
    raise ValueError("Không tìm thấy file .gguf cho mô hình này. "
                     "llama.cpp cần model định dạng GGUF.")


_ENGINES = ("llamacpp", "llama-server")


def start_server(name, model_id, env_id, served_name="", gpu_model="",
                 n_gpu_layers=99, n_ctx=8192, chat_format="", extra_args="",
                 time_limit="", auto_resubmit=True, engine="llamacpp"):
    """Validate, create the DB row + server dir, and start the run-loop thread.
    Returns the new server_id. Raises ValueError on bad input.
    """
    engine = (engine or "llamacpp").strip()
    if engine not in _ENGINES:
        raise ValueError("Engine không hợp lệ.")
    # Fail fast (before creating the row/thread) if the native engine is picked
    # but its binary isn't on the shared FS yet — clearer than a server that
    # errors only once SLURM grants it a GPU.
    if engine == "llama-server" and not os.path.isfile(config.LLAMA_SERVER_BIN):
        raise ValueError(
            f"Engine 'llama-server (native)' cần binary tại "
            f"{config.LLAMA_SERVER_BIN} — tải bản CUDA của ggml-org/llama.cpp "
            "về đường dẫn đó (hoặc đặt biến môi trường "
            "HOSTA100_LLAMA_SERVER_BIN). Hiện chưa có.")

    env = env_service.get_env(env_id)
    if not env:
        raise ValueError("Môi trường đã chọn không tồn tại.")
    py = env_service._python_path(env["path"])
    if not os.path.exists(py):
        raise ValueError("Thiếu trình thông dịch của môi trường trên đĩa.")

    model = model_service.get_model(model_id)
    if not model:
        raise ValueError("Mô hình đã chọn không tồn tại.")
    _resolve_gguf(model)            # fail fast if the model has no .gguf weights

    name = (name or model["name"]).strip()[:128]
    served_name = (served_name or model["name"]).strip()[:128]
    try:
        n_gpu_layers = max(0, int(n_gpu_layers))
        n_ctx = max(512, int(n_ctx))
    except (TypeError, ValueError):
        raise ValueError("n_gpu_layers / n_ctx phải là số nguyên.")
    chat_format = (chat_format or "").strip()
    time_limit = (time_limit or config.SLURM_TIME or "").strip()

    # Owner-only power tool (same trust model as the Terminal): validate that
    # extra_args parse with shell rules now, but store the raw string — the
    # run-loop re-parses it (and rebuilds the whole command) on every (re)submit.
    try:
        shlex.split(extra_args or "")
    except ValueError as exc:
        raise ValueError(f"Tham số bổ sung không hợp lệ: {exc}")

    if not (job_service.slurm_active() and job_service._SBATCH):
        raise ValueError("SLURM/sbatch không khả dụng trên node này — "
                         "không thể cấp GPU cho server.")

    server_id = db.execute(
        "INSERT INTO servers (name, served_name, model_id, env_id, engine, "
        "status, gpu_model, n_gpu_layers, n_ctx, chat_format, extra_args, "
        "time_limit, auto_resubmit, created_at) "
        "VALUES (?,?,?,?, ?, 'queued', ?,?,?,?,?,?,?,?)",
        (name, served_name, model_id, env_id, engine, gpu_model, n_gpu_layers,
         n_ctx, chat_format, extra_args, time_limit, 1 if auto_resubmit else 0,
         db.now()),
        commit=True,
    )

    server_dir = file_utils.safe_join(config.SERVERS_DIR, str(server_id))
    os.makedirs(server_dir, exist_ok=True)
    db.execute("UPDATE servers SET logs_path=? WHERE id=?",
               (server_dir, server_id), commit=True)

    threading.Thread(target=_run_loop, args=(server_id,), daemon=True).start()
    return server_id


def _build_cmd(server):
    """Rebuild the server command from a server DB row. Used on the first submit
    AND on every auto-resubmit / restart-resume, so the source of truth is the
    row, not a value captured once in memory.

    Returns (py, cmd, extra_lib_dirs). `py` is the env's interpreter (the batch
    script uses it to pick a free port and to locate the env's nvidia CUDA
    wheels). `extra_lib_dirs` is prepended to LD_LIBRARY_PATH — the native
    llama-server binary dlopen's its sibling .so files (libllama/libggml/cudart)
    from its own directory, so that directory must be on the path.
    """
    env = env_service.get_env(server["env_id"])
    if not env:
        raise ValueError("Môi trường của server không còn tồn tại.")
    py = env_service._python_path(env["path"])
    if not os.path.exists(py):
        raise ValueError("Thiếu trình thông dịch của môi trường trên đĩa.")
    model = model_service.get_model(server["model_id"])
    if not model:
        raise ValueError("Mô hình của server không còn tồn tại.")
    gguf = _resolve_gguf(model)
    n_gpu_layers = str(server["n_gpu_layers"] or 99)
    n_ctx = str(server["n_ctx"] or 8192)

    if (server.get("engine") or "llamacpp") == "llama-server":
        binpath = config.LLAMA_SERVER_BIN
        if not os.path.isfile(binpath):
            raise ValueError(
                f"Không tìm thấy binary llama-server tại {binpath}. Tải bản "
                "CUDA của ggml-org/llama.cpp về đó (hoặc đặt "
                "HOSTA100_LLAMA_SERVER_BIN).")
        # Native ggml-org server: --jinja makes it use the GGUF's embedded chat
        # template + its built-in tool-call parser (Qwen3-Coder → real
        # tool_calls). Flag names differ from the Python server (-ngl/-c).
        cmd = [binpath,
               "--model", gguf,
               "--alias", server["served_name"],
               "--host", "0.0.0.0",
               "-ngl", n_gpu_layers,
               "-c", n_ctx,
               "--jinja"]
        if server.get("extra_args"):
            cmd += shlex.split(server["extra_args"])
        return py, cmd, [os.path.dirname(os.path.abspath(binpath))]

    cmd = [py, "-m", "llama_cpp.server",
           "--model", gguf,
           "--model_alias", server["served_name"],
           "--host", "0.0.0.0",
           "--n_gpu_layers", n_gpu_layers,
           "--n_ctx", n_ctx]
    if server.get("chat_format"):
        cmd += ["--chat_format", server["chat_format"]]
    if server.get("extra_args"):
        cmd += shlex.split(server["extra_args"])
    return py, cmd, []


def _append_log(server_dir, text):
    try:
        with open(os.path.join(server_dir, "server.log"), "a",
                  encoding="utf-8", errors="replace") as fh:
            fh.write(text)
    except OSError:
        pass


def _cuda_lib_dirs(py):
    """nvidia-*-cu12 wheel lib dirs inside the env (libcudart / libcublas / …).

    The prebuilt CUDA build of llama.cpp dlopen's libcudart.so.12 etc. at load
    time; on an HPC compute node those aren't on the default library path. If the
    user pip-installed the matching `nvidia-*-cu12` runtime wheels into the env,
    this finds them so the batch script can put them on LD_LIBRARY_PATH — no
    `module load cuda` needed (libcuda.so.1, the driver, is already on GPU nodes).
    Returns [] when no such wheels are present (e.g. a CPU-only build)."""
    code = ("import sysconfig, glob, os;"
            "p = sysconfig.get_paths()['purelib'];"
            "print(chr(10).join(glob.glob(os.path.join(p, 'nvidia', '*', 'lib'))))")
    try:
        out = subprocess.run([py, "-c", code], stdout=subprocess.PIPE,
                             text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        return []
    return [d.strip() for d in out.splitlines() if d.strip()]


def _write_batch_script(server_dir, py, cmd, extra_lib_dirs=()):
    """The run.sh that sbatch executes on the compute node: make the env's bundled
    CUDA libs (and any engine-specific lib dirs) findable, pick a free port,
    advertise <node>:<port> in endpoint.json, then exec the server."""
    script = os.path.join(server_dir, "run.sh")
    endpoint = os.path.join(server_dir, _ENDPOINT_FILE)
    quoted = " ".join(shlex.quote(c) for c in cmd)
    ld_line = ""
    cuda_dirs = list(extra_lib_dirs) + _cuda_lib_dirs(py)
    if cuda_dirs:
        joined = shlex.quote(":".join(cuda_dirs))
        ld_line = (f"export LD_LIBRARY_PATH={joined}"
                   f'"${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"\n')
    body = f"""#!/bin/bash
set -e
{ld_line}PORT=$({shlex.quote(py)} -c 'import socket;s=socket.socket();s.bind(("0.0.0.0",0));p=s.getsockname()[1];s.close();print(p)')
HOST=${{SLURMD_NODENAME:-$(hostname -s)}}
cat > {shlex.quote(endpoint)} <<EOF
{{"host":"$HOST","port":$PORT}}
EOF
echo "[host-a100] llama.cpp server on $HOST:$PORT"
exec {quoted} --port "$PORT"
"""
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(body)
    try:
        os.chmod(script, 0o755)
    except OSError:
        pass
    return script


def _sbatch_flags(server_id, server_dir, slurm_out, gpu_model, time_limit):
    flags = ["--job-name", f"hosta100-srv-{server_id}",
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
    if config.SLURM_CPUS:
        flags += ["--cpus-per-task", config.SLURM_CPUS]
    if config.SLURM_MEM:
        flags += ["--mem", config.SLURM_MEM]
    return flags


# Auto-resubmit: a server held for its SLURM walltime gets killed when the time
# runs out, so for ~24/7 availability we re-`sbatch` it automatically. To avoid a
# tight crash-loop hammering the queue, we ONLY resubmit a run that had become
# ready (a healthy server that later died = walltime/preempt), and we stop after
# this many consecutive ready-but-died-within-60s runs (a server that loads then
# crashes fast = a real problem the logs will show).
_QUICK_FAIL_SECS = 60
_QUICK_FAIL_MAX = 3
_RESUBMIT_DELAY = 5


def _run_loop(server_id, existing_slurm_id=None):
    """Own one server's whole lifecycle: submit → monitor → (auto-resubmit) →
    repeat, until the user stops/deletes it or it fails for real.

    existing_slurm_id is set on restart-resume: the first iteration re-attaches
    to a job that is ALREADY running on SLURM instead of submitting a new one.
    """
    with _monitors_lock:
        if server_id in _monitors:
            return                      # a loop already owns this server
        _monitors.add(server_id)
    try:
        quick_fail = 0
        first = True
        while True:
            server = get_server(server_id)
            if not server:
                return                  # deleted
            if server["status"] == "stopped":
                return                  # user stopped it (possibly during a gap)
            server_dir = server["logs_path"]

            if first and existing_slurm_id:
                slurm_id = existing_slurm_id
            else:
                try:
                    py, cmd, lib_dirs = _build_cmd(server)
                except ValueError as exc:
                    _set_status(server_id, "error")
                    _append_log(server_dir, f"\n{exc}\n")
                    return
                slurm_id = _submit_once(server_id, server_dir, py, cmd, lib_dirs,
                                        server["gpu_model"], server["time_limit"])
                if not slurm_id:
                    _set_status(server_id, "error")
                    return
            first = False

            t0 = time.time()
            outcome = _monitor(server_id, slurm_id, server_dir)
            ran = time.time() - t0

            if outcome == "user_stopped":
                return
            if outcome == "ended_ready" and server["auto_resubmit"]:
                quick_fail = quick_fail + 1 if ran < _QUICK_FAIL_SECS else 0
                if quick_fail >= _QUICK_FAIL_MAX:
                    _set_status(server_id, "error")
                    _append_log(server_dir, "\n[auto-resubmit] server chết liên "
                                "tục (<60s) — dừng tự khởi động lại. Kiểm tra log "
                                "(thường là hết VRAM: hạ n_gpu_layers/n_ctx).\n")
                    return
                _append_log(server_dir, f"\n[auto-resubmit] phiên kết thúc, gửi "
                            f"lại sau {_RESUBMIT_DELAY}s…\n")
                db.execute("UPDATE servers SET status='queued', node=NULL, "
                           "port=NULL, slurm_job_id=NULL WHERE id=?",
                           (server_id,), commit=True)
                time.sleep(_RESUBMIT_DELAY)
                continue

            # Terminal: either it never came up (error) or it ended and we are
            # not resubmitting (stopped).
            final = "stopped" if outcome == "ended_ready" else "error"
            db.execute("UPDATE servers SET status=?, stopped_at=? WHERE id=?",
                       (final, db.now(), server_id), commit=True)
            _append_log(server_dir, f"\nSLURM job {slurm_id} kết thúc ({final}).\n")
            return
    finally:
        with _monitors_lock:
            _monitors.discard(server_id)


def _submit_once(server_id, server_dir, py, cmd, lib_dirs, gpu_model, time_limit):
    """Write the batch script and sbatch it. Returns the SLURM job id, or None on
    failure (the caller marks the server errored)."""
    slurm_out = os.path.join(server_dir, "slurm.out")
    open(slurm_out, "w").close()                # fresh file for this run's tail
    try:
        os.remove(os.path.join(server_dir, _ENDPOINT_FILE))   # drop stale endpoint
    except OSError:
        pass
    _set_status(server_id, "queued")

    script = _write_batch_script(server_dir, py, cmd, lib_dirs)
    submit = ([job_service._SBATCH, "--parsable"]
              + _sbatch_flags(server_id, server_dir, slurm_out, gpu_model,
                              time_limit)
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
    db.execute("UPDATE servers SET slurm_job_id=? WHERE id=?",
               (slurm_id, server_id), commit=True)
    _append_log(server_dir, f"Đã gửi SLURM job {slurm_id}; chờ cấp GPU…\n")
    return slurm_id


def _monitor(server_id, slurm_id, server_dir):
    """Poll squeue, tail slurm.out, discover the endpoint, and health-check until
    the server is ready; keep watching until the SLURM job ends. Returns:
        'user_stopped'       — the row went non-active (user stop/delete)
        'ended_ready'        — job left the queue AFTER becoming ready (resubmit-able)
        'ended_never_ready'  — job left the queue without ever serving (real error)
    """
    pos = 0
    slurm_out = os.path.join(server_dir, "slurm.out")
    endpoint_path = os.path.join(server_dir, _ENDPOINT_FILE)
    host = port = None
    became_ready = False
    unknown = 0
    while True:
        pos = _tail(slurm_out, server_dir, pos)
        cur = get_server(server_id)
        if not cur or cur.get("status") not in _ACTIVE:
            return "user_stopped"
        state = job_service._squeue_state(slurm_id)
        if state is None:
            break                       # job left the queue → it ended
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
                    db.execute("UPDATE servers SET node=?, port=? WHERE id=?",
                               (host, port, server_id), commit=True)
            if host and not became_ready:
                if _health(host, port):
                    became_ready = True
                    _set_status(server_id, "ready")
                    _append_log(server_dir,
                                f"\nSERVER READY: http://{host}:{port}/v1\n")
                else:
                    _set_status(server_id, "loading")
            elif not host:
                _set_status(server_id, "loading")
        time.sleep(_POLL_SECS)

    _tail(slurm_out, server_dir, pos)
    return "ended_ready" if became_ready else "ended_never_ready"


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
    """True when the llama.cpp OpenAI server answers /v1/models with 200."""
    if not host or not port:
        return False
    try:
        conn = http.client.HTTPConnection(host, port, timeout=4)
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def _set_status(server_id, status):
    db.execute("UPDATE servers SET status=? WHERE id=?",
               (status, server_id), commit=True)


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
    db.execute("UPDATE servers SET status='stopped', stopped_at=? WHERE id=?",
               (db.now(), server_id), commit=True)
    return True


def delete_server(server_id):
    server = get_server(server_id)
    if not server:
        return False
    if server.get("status") in _ACTIVE:
        _scancel(server)
    path = server.get("logs_path")
    if path and file_utils.is_within(config.SERVERS_DIR, path) \
            and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    db.execute("DELETE FROM servers WHERE id=?", (server_id,), commit=True)
    return True


# --------------------------------------------------------------------------- #
# Restart reconciliation
# --------------------------------------------------------------------------- #
def resume_monitors():
    """On app startup, re-attach a monitor to any server still marked active.

    sbatch jobs survive an app restart (the SLURM job keeps running), but the
    in-memory monitor thread does not — so without this an alive server would be
    stuck showing its last status forever. We re-poll each one; squeue tells us
    if it is still alive, and the health check flips it back to ready.
    """
    rows = db.execute(
        "SELECT * FROM servers WHERE status IN ('queued','loading','ready') "
        "AND slurm_job_id IS NOT NULL", fetch="all") or []
    for r in rows:
        s = dict(r)
        slurm_id = (s.get("slurm_job_id") or "").strip()
        server_dir = s.get("logs_path")
        if not (slurm_id.isdigit() and server_dir and os.path.isdir(server_dir)):
            continue
        # Re-attach to the still-running SLURM job; the loop then keeps the
        # auto-resubmit behaviour going for future walltime expiries.
        threading.Thread(
            target=_run_loop, args=(s["id"], slurm_id), daemon=True,
        ).start()
