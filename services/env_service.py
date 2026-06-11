"""
Virtual environment management: create venvs and install pip packages.

Both venv creation and pip installs run in background threads and stream their
output into utils.progress so the UI can poll. All subprocess calls use fixed
argument lists (never shell=True). Package names are validated against a strict
pattern so a user cannot inject extra pip flags or shell metacharacters.
"""
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import venv

import config
from services import storage_service as db
from utils import file_utils, progress

# A pip requirement spec we are willing to install: name plus optional extras
# and a version constraint. Deliberately conservative — no URLs, no flags.
_PKG_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*"          # package name
    r"(\[[A-Za-z0-9._,-]+\])?"              # optional [extras]
    r"([<>=!~]=?[A-Za-z0-9._*+-]+"          # optional version op + version
    r"(,[<>=!~]=?[A-Za-z0-9._*+-]+)*)?$"    # additional comma-separated ops
)


def list_envs():
    rows = db.execute("SELECT * FROM envs ORDER BY created_at DESC",
                      fetch="all") or []
    envs = []
    for r in rows:
        d = dict(r)
        d["exists"] = os.path.isdir(d["path"])
        d["python"] = _python_path(d["path"])
        envs.append(d)
    return envs


def get_env(env_id):
    row = db.execute("SELECT * FROM envs WHERE id=?", (env_id,), fetch="one")
    return dict(row) if row else None


def name_taken(name):
    return db.execute("SELECT 1 FROM envs WHERE name=?", (name,),
                      fetch="one") is not None


def _pip_env():
    """Environment for pip subprocesses tuned for speed on a shared FS.

    - PIP_CACHE_DIR: a shared cache so a wheel is downloaded once, then reused
      by every later install/env (the biggest win on a slow network filesystem).
    - PIP_DISABLE_PIP_VERSION_CHECK: skips a network round trip to PyPI that pip
      otherwise makes on every run just to see if a newer pip exists.
    - PIP_NO_INPUT: never block waiting on a prompt (we also pass --no-input).
    - PYTHONUNBUFFERED: pip's stdout is block-buffered when piped (not a TTY),
      so without this its "Collecting/Downloading" lines stay stuck in pip's
      buffer and only appear when it exits — making the log look frozen during a
      slow download. Unbuffered output streams those lines in real time.
    """
    env = dict(os.environ)
    env["PIP_CACHE_DIR"] = config.PIP_CACHE_DIR
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_INPUT"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


# Per-env install locks. Two installs into the SAME venv at once can corrupt its
# site-packages (pip does not lock it), so we serialize them. Installs into
# DIFFERENT envs still run in parallel. The process is single-threaded-Flask with
# daemon worker threads, so in-memory locks are sufficient and correct.
_install_locks = {}
_install_locks_guard = threading.Lock()


def _env_lock(env_id):
    with _install_locks_guard:
        lock = _install_locks.get(env_id)
        if lock is None:
            lock = threading.Lock()
            _install_locks[env_id] = lock
        return lock


def _pip_progress(line, cur, step):
    """Map one pip output line to (progress, step) for a meaningful bar.

    pip is silent while it unpacks wheels to disk — the slowest phase on a
    network FS — so a naive per-line bar stalls then jumps to 100. We instead
    jump to ~85% the moment pip starts writing files, and creep during the
    download/resolve phase, so the bar tracks what is actually happening.
    """
    low = line.lower().lstrip()
    if "installing collected packages" in low:
        return 85, "đang ghi thư viện vào môi trường…"
    if low.startswith("collecting"):
        return min(78, cur + 3), "đang tải & phân giải phụ thuộc…"
    if low.startswith(("downloading", "using cached")):
        return min(78, cur + 2), step
    if low.startswith("successfully installed"):
        return 97, "gần xong…"
    if cur < 78:
        return cur + 1, step
    return cur, step


def _run_pip(task_id, cmd, start_step, success_msg):
    """Stream a pip subprocess into the progress registry. Returns success bool."""
    progress.update("env", task_id, status="running", progress=5,
                    step=start_step, append_log="$ " + " ".join(cmd) + "\n")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace", env=_pip_env(),
    )

    # While pip is silently unpacking wheels to disk, no lines arrive, so we
    # creep the bar in the background and halt it as soon as the next line does.
    ticker = {"stop": None, "thread": None}

    def stop_ticker():
        if ticker["stop"] is not None:
            ticker["stop"].set()
            if ticker["thread"]:
                ticker["thread"].join(timeout=1)
            ticker["stop"], ticker["thread"] = None, None

    for line in proc.stdout:
        stop_ticker()
        snap = progress.get("env", task_id)
        new_prog, new_step = _pip_progress(line, snap["progress"], snap["step"])
        kwargs = {"append_log": line}
        if new_prog != snap["progress"]:
            kwargs["progress"] = new_prog
        if new_step != snap["step"]:
            kwargs["step"] = new_step
        progress.update("env", task_id, **kwargs)
        if "installing collected packages" in line.lower():
            ticker["stop"] = threading.Event()
            ticker["thread"] = threading.Thread(
                target=_creep, args=(task_id, ticker["stop"], new_prog, 96),
                daemon=True)
            ticker["thread"].start()
    stop_ticker()
    code = proc.wait()
    if code == 0:
        progress.update("env", task_id, progress=100, status="done",
                        step="hoàn tất", append_log=f"\n{success_msg}\n")
        return True
    progress.update("env", task_id, status="error", step="pip thất bại",
                    append_log=f"\npip kết thúc với mã {code}.\n")
    return False


def _install_worker(task_id, env_id, cmd, start_step, success_msg):
    """Acquire the env's install lock (waiting if busy), then run pip."""
    lock = _env_lock(env_id)
    if not lock.acquire(blocking=False):
        progress.update("env", task_id, status="running", progress=2,
                        step="đang chờ một tác vụ cài đặt khác trên môi trường này…",
                        append_log="Môi trường này đang có tác vụ cài đặt khác. "
                                   "Đang chờ tới lượt…\n")
        lock.acquire()
    try:
        _run_pip(task_id, cmd, start_step, success_msg)
    finally:
        lock.release()


def _python_path(env_dir):
    """Path to the interpreter inside a venv (Linux layout, Windows fallback)."""
    posix = os.path.join(env_dir, "bin", "python")
    win = os.path.join(env_dir, "Scripts", "python.exe")
    if os.path.exists(posix):
        return posix
    if os.path.exists(win):
        return win
    return posix  # expected location on the Linux HPC target


def _creep(task_id, stop, start, cap, interval=1.2):
    """Slowly advance the progress bar while a blocking, silent step runs.

    ensurepip writes hundreds of small files to disk and emits no output, so
    without this the bar would freeze. We tick it up ~1%/interval toward a cap
    (never reaching 100) so the user sees the task is alive; the real step bumps
    it to its final value when it finishes.
    """
    val = start
    while val < cap and not stop.wait(interval):
        val += 1
        progress.update("env", task_id, progress=val)


class _ProgressEnvBuilder(venv.EnvBuilder):
    """venv builder that reports each phase into the progress registry.

    venv.create() runs several blocking phases with no output of its own. By
    hooking each phase we can show a meaningful, moving status — most usefully
    a creeping bar during _setup_pip, the slow silent step where ensurepip
    writes many small files onto the network filesystem.
    """

    def __init__(self, task_id, **kw):
        super().__init__(**kw)
        self._tid = task_id

    def ensure_directories(self, env_dir):
        ctx = super().ensure_directories(env_dir)
        progress.update("env", self._tid, progress=20,
                        step="tạo cấu trúc thư mục",
                        append_log="Tạo thư mục môi trường…\n")
        return ctx

    def setup_python(self, context):
        progress.update("env", self._tid, progress=40,
                        step="sao chép trình thông dịch Python",
                        append_log="Sao chép Python…\n")
        super().setup_python(context)

    def _setup_pip(self, context):
        progress.update("env", self._tid, progress=55,
                        step="cài pip + setuptools (đang ghi nhiều file, hãy đợi)…",
                        append_log="Cài pip + setuptools…\n")
        # ensurepip is one blocking, silent call: creep the bar while it runs.
        stop = threading.Event()
        ticker = threading.Thread(
            target=_creep, args=(self._tid, stop, 55, 88), daemon=True)
        ticker.start()
        try:
            super()._setup_pip(context)
        finally:
            stop.set()
            ticker.join(timeout=2)
        progress.update("env", self._tid, progress=90,
                        step="pip đã cài xong", append_log="Đã cài pip.\n")

    def post_setup(self, context):
        progress.update("env", self._tid, progress=92, step="hoàn thiện")
        super().post_setup(context)


def create_env(task_id, env_name):
    """Create a venv in the background, registering it in the DB on success."""
    env_name = file_utils.validate_name(env_name)
    progress.init("env", task_id, step="đang khởi tạo")

    def worker():
        try:
            env_dir = file_utils.safe_join(config.ENVS_DIR, env_name)
            progress.update("env", task_id, status="running", progress=10,
                            step="đang dựng môi trường ảo",
                            append_log=f"Tạo venv tại {env_dir}\n")
            builder = _ProgressEnvBuilder(task_id, with_pip=True, clear=False)
            builder.create(env_dir)
            progress.update("env", task_id, progress=95,
                            step="đăng ký môi trường",
                            append_log="Đã tạo venv.\n")
            db.execute(
                "INSERT INTO envs (name, path, created_at) VALUES (?, ?, ?)",
                (env_name, env_dir, db.now()), commit=True,
            )
            progress.update("env", task_id, progress=100, status="done",
                            step="hoàn tất", append_log="Môi trường sẵn sàng.\n")
        except Exception as exc:  # noqa: BLE001
            progress.update("env", task_id, status="error", step="failed",
                            append_log=f"ERROR: {exc}\n")

    threading.Thread(target=worker, daemon=True).start()


def validate_packages(raw):
    """
    Parse and validate a whitespace/comma separated package list.

    Returns a list of clean specs or raises ValueError naming the bad token.
    """
    tokens = [t for t in re.split(r"[\s,]+", raw.strip()) if t]
    if not tokens:
        raise ValueError("Chưa nhập gói nào.")
    for t in tokens:
        if not _PKG_RE.match(t):
            raise ValueError(f"Tên gói không hợp lệ: {t!r}")
    return tokens


def install_packages(task_id, env_id, packages):
    """Run `pip install <packages>` inside the env, streaming output."""
    env = get_env(env_id)
    if not env:
        raise ValueError("Không tìm thấy môi trường.")
    py = _python_path(env["path"])
    if not os.path.exists(py):
        raise ValueError("Thiếu trình thông dịch của môi trường trên đĩa.")
    progress.init("env", task_id, step="đang chuẩn bị cài đặt")
    cmd = [py, "-m", "pip", "install", "--no-input", "--prefer-binary"] + packages

    def worker():
        try:
            _install_worker(task_id, env_id, cmd, "pip install",
                            "Đã cài đặt thư viện thành công.")
        except Exception as exc:  # noqa: BLE001
            progress.update("env", task_id, status="error", step="failed",
                            append_log=f"ERROR: {exc}\n")

    threading.Thread(target=worker, daemon=True).start()


def install_requirements_file(task_id, env_id, req_path):
    """Run `pip install -r <req_path>` inside the env, streaming output."""
    env = get_env(env_id)
    if not env:
        raise ValueError("Không tìm thấy môi trường.")
    if not os.path.isfile(req_path):
        raise ValueError("Không tìm thấy file requirements.")
    py = _python_path(env["path"])
    if not os.path.exists(py):
        raise ValueError("Thiếu trình thông dịch của môi trường trên đĩa.")
    progress.init("env", task_id, step="đang chuẩn bị cài từ requirements.txt")
    cmd = [py, "-m", "pip", "install", "--no-input", "--prefer-binary",
           "-r", req_path]

    def worker():
        try:
            _install_worker(task_id, env_id, cmd, "pip install -r",
                            "Đã cài đặt từ requirements.txt thành công.")
        except Exception as exc:  # noqa: BLE001
            progress.update("env", task_id, status="error", step="failed",
                            append_log=f"ERROR: {exc}\n")

    threading.Thread(target=worker, daemon=True).start()


# faster-whisper model sizes offered for one-click prefetch. Validated against
# this allowlist so the name can't inject anything into the download subprocess.
WHISPER_MODELS = (
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v2", "large-v3", "large-v3-turbo",
)


def _run_whisper_download(task_id, cmd, model):
    """Stream a faster-whisper model download into the env progress channel."""
    progress.update("env", task_id, status="running", progress=5,
                    step=f"đang tải model {model} (trên node đăng nhập có mạng)…",
                    append_log=f"$ tải faster-whisper model: {model}\n")
    # huggingface_hub emits download progress sparsely when stdout isn't a TTY,
    # so creep the bar in the background and bump it whenever an aggregate
    # "Fetching N files: X%" line shows up.
    env = _pip_env()
    env["HF_HUB_OFFLINE"] = "0"          # we ARE online here (login node)
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    stop = threading.Event()
    ticker = threading.Thread(target=_creep, args=(task_id, stop, 5, 92),
                              daemon=True)
    ticker.start()
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace", env=env,
        )
        for line in proc.stdout:
            progress.update("env", task_id, append_log=line)
            m = re.search(r"Fetching\s+\d+\s+files:\s*(\d+)%", line)
            if m:
                progress.update("env", task_id,
                                progress=max(5, min(95, 10 + int(0.85 * int(m.group(1))))))
        code = proc.wait()
    finally:
        stop.set()
        ticker.join(timeout=1)
    if code == 0:
        progress.update("env", task_id, progress=100, status="done",
                        step=f"đã tải xong model {model}",
                        append_log=f"\nĐã tải model '{model}' vào HF cache. "
                                   "Job trên compute node giờ đọc offline được.\n")
    else:
        progress.update("env", task_id, status="error", step="tải model thất bại",
                        append_log=f"\nTiến trình kết thúc với mã {code}.\n")


def prefetch_whisper_model(task_id, env_id, model):
    """Download a faster-whisper model into the HF cache, on THIS (login) node.

    Runs as a LOCAL subprocess (never srun): only the login node has internet,
    and the model lands in the shared-FS HF cache so compute-node jobs read it
    offline. Reuses the "env" progress channel so the existing progress bar/poll
    UI shows it. The env must already have faster-whisper installed.
    """
    if model not in WHISPER_MODELS:
        raise ValueError(f"Model Whisper không hợp lệ: {model!r}")
    env = get_env(env_id)
    if not env:
        raise ValueError("Không tìm thấy môi trường.")
    py = _python_path(env["path"])
    if not os.path.exists(py):
        raise ValueError("Thiếu trình thông dịch của môi trường trên đĩa.")
    progress.init("env", task_id, step=f"chuẩn bị tải model {model}")
    # model passed as argv (not interpolated) so it can never break out of -c.
    script = (
        "import sys\n"
        "try:\n"
        "    from faster_whisper.utils import download_model\n"
        "except Exception:\n"
        "    from faster_whisper import download_model\n"
        "print('CACHED ' + download_model(sys.argv[1]))\n"
    )
    cmd = [py, "-c", script, model]

    def worker():
        try:
            _run_whisper_download(task_id, cmd, model)
        except Exception as exc:  # noqa: BLE001
            progress.update("env", task_id, status="error", step="failed",
                            append_log=f"ERROR: {exc}\n")

    threading.Thread(target=worker, daemon=True).start()


# List installed distributions via stdlib importlib.metadata instead of
# `pip freeze`. It avoids importing the whole pip package (hundreds of files to
# stat on a network FS), so it returns noticeably faster when viewing an env.
_FREEZE_SNIPPET = (
    "import importlib.metadata as m\n"
    "pkgs = sorted({(d.metadata['Name'], d.version) for d in m.distributions()"
    " if d.metadata['Name']}, key=lambda x: x[0].lower())\n"
    "print('\\n'.join('%s==%s' % p for p in pkgs))\n"
)


def pip_freeze(env_id):
    """Return installed packages (list of strings) for an env, or an error."""
    env = get_env(env_id)
    if not env:
        return ["(không tìm thấy môi trường)"]
    py = _python_path(env["path"])
    if not os.path.exists(py):
        return ["(thiếu trình thông dịch)"]
    try:
        out = subprocess.run([py, "-c", _FREEZE_SNIPPET],
                             capture_output=True, text=True, timeout=30)
        lines = [l for l in out.stdout.splitlines() if l.strip()]
        return lines or ["(chưa cài gói nào)"]
    except Exception as exc:  # noqa: BLE001
        return [f"(không liệt kê được thư viện: {exc})"]


def delete_env(env_id):
    env = get_env(env_id)
    if not env:
        return False
    path = env["path"]
    if path and file_utils.is_within(config.ENVS_DIR, path) \
            and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    db.execute("DELETE FROM envs WHERE id=?", (env_id,), commit=True)
    return True
