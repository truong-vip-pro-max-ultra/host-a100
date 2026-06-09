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


def _python_path(env_dir):
    """Path to the interpreter inside a venv (Linux layout, Windows fallback)."""
    posix = os.path.join(env_dir, "bin", "python")
    win = os.path.join(env_dir, "Scripts", "python.exe")
    if os.path.exists(posix):
        return posix
    if os.path.exists(win):
        return win
    return posix  # expected location on the Linux HPC target


def create_env(task_id, env_name):
    """Create a venv in the background, registering it in the DB on success."""
    env_name = file_utils.validate_name(env_name)
    progress.init("env", task_id, step="creating venv")

    def worker():
        try:
            env_dir = file_utils.safe_join(config.ENVS_DIR, env_name)
            progress.update("env", task_id, status="running", progress=10,
                            step="building virtual environment",
                            append_log=f"Creating venv at {env_dir}\n")
            builder = venv.EnvBuilder(with_pip=True, clear=False)
            builder.create(env_dir)
            progress.update("env", task_id, progress=80,
                            step="registering", append_log="venv created.\n")
            db.execute(
                "INSERT INTO envs (name, path, created_at) VALUES (?, ?, ?)",
                (env_name, env_dir, db.now()), commit=True,
            )
            progress.update("env", task_id, progress=100, status="done",
                            step="complete", append_log="Environment ready.\n")
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
    progress.init("env", task_id, step="installing packages")

    def worker():
        try:
            cmd = [py, "-m", "pip", "install", "--no-input"] + packages
            progress.update("env", task_id, status="running", progress=5,
                            step="pip install",
                            append_log="$ " + " ".join(cmd) + "\n")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                progress.update("env", task_id, append_log=line)
                # Nudge the bar forward as pip works (capped below 95).
                cur = progress.get("env", task_id)["progress"]
                if cur < 95:
                    progress.update("env", task_id, progress=cur + 1)
            code = proc.wait()
            if code == 0:
                progress.update("env", task_id, progress=100, status="done",
                                step="installed",
                                append_log="\npip install succeeded.\n")
            else:
                progress.update("env", task_id, status="error",
                                step="pip failed",
                                append_log=f"\npip exited with code {code}.\n")
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
    progress.init("env", task_id, step="installing from requirements.txt")

    def worker():
        try:
            cmd = [py, "-m", "pip", "install", "--no-input", "-r", req_path]
            progress.update("env", task_id, status="running", progress=5,
                            step="pip install -r",
                            append_log="$ " + " ".join(cmd) + "\n")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                progress.update("env", task_id, append_log=line)
                cur = progress.get("env", task_id)["progress"]
                if cur < 95:
                    progress.update("env", task_id, progress=cur + 1)
            code = proc.wait()
            if code == 0:
                progress.update("env", task_id, progress=100, status="done",
                                step="installed",
                                append_log="\nRequirements installed.\n")
            else:
                progress.update("env", task_id, status="error",
                                step="pip failed",
                                append_log=f"\npip exited with code {code}.\n")
        except Exception as exc:  # noqa: BLE001
            progress.update("env", task_id, status="error", step="failed",
                            append_log=f"ERROR: {exc}\n")

    threading.Thread(target=worker, daemon=True).start()


def pip_freeze(env_id):
    """Return installed packages (list of strings) for an env, or an error."""
    env = get_env(env_id)
    if not env:
        return ["(không tìm thấy môi trường)"]
    py = _python_path(env["path"])
    if not os.path.exists(py):
        return ["(thiếu trình thông dịch)"]
    try:
        out = subprocess.run([py, "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=60)
        lines = [l for l in out.stdout.splitlines() if l.strip()]
        return lines or ["(chưa cài gói nào)"]
    except Exception as exc:  # noqa: BLE001
        return [f"(pip freeze lỗi: {exc})"]


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
