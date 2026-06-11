"""
Central configuration for the ML platform.

All paths live under DATA_DIR. On the target Linux HPC server this defaults to
/data. For local development (e.g. Windows) or shared nodes where /data is not
writable, you can override every path by setting the HOSTA100_DATA_DIR
environment variable. If neither is usable, we fall back to ~/host-a100-data.
"""
import os


def _resolve_data_dir():
    """Pick the data directory.

    Priority:
      1. HOSTA100_DATA_DIR if set (explicit override).
      2. /data if it exists and is writable (the target HPC server).
      3. <app dir>/host-a100-data as a safe fallback that lives INSIDE the
         project directory (e.g. shared nodes where /data is owned by root and
         not writable). Keeping it next to the code makes the deployment
         self-contained instead of scattering data into the user's home dir.
    """
    explicit = os.environ.get("HOSTA100_DATA_DIR")
    if explicit:
        return explicit

    default = "/data"
    # Writable if it already exists and we can write, or its parent lets us
    # create it. os.access on a missing path returns False, so check the
    # existing directory or the closest existing ancestor.
    probe = default
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    if os.path.isdir(probe) and os.access(probe, os.W_OK):
        return default

    # Fallback next to this file (the project root), not the home directory.
    app_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(app_dir, "host-a100-data")


# Root data directory. Override with HOSTA100_DATA_DIR for local testing.
DATA_DIR = _resolve_data_dir()

MODELS_DIR = os.path.join(DATA_DIR, "models")
ENVS_DIR = os.path.join(DATA_DIR, "envs")
JOBS_DIR = os.path.join(DATA_DIR, "jobs")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
PROJECTS_DIR = os.path.join(DATA_DIR, "projects")

# Shared pip download/wheel cache. Living under DATA_DIR (same filesystem as the
# envs) means a package is downloaded once and reused across every venv, which
# is the single biggest speedup for repeated installs on a slow shared FS.
PIP_CACHE_DIR = os.path.join(DATA_DIR, "pip-cache")

# SQLite database file.
DB_PATH = os.path.join(DATA_DIR, "platform.db")

# Upload limits. Large model files are streamed to disk, so this is generous.
# 200 GB hard cap to avoid filling the shared filesystem accidentally.
MAX_CONTENT_LENGTH = 200 * 1024 * 1024 * 1024

# Allowed model file extensions (defense in depth; validated again server-side).
ALLOWED_MODEL_EXTENSIONS = {
    ".bin", ".pt", ".pth", ".onnx", ".safetensors", ".ckpt",
    ".h5", ".pb", ".gguf", ".ggml", ".tar", ".gz", ".zip", ".npz",
}

# Code / config files a user may paste or upload into a project. Empty string
# covers extensionless files (e.g. "Dockerfile", "LICENSE").
ALLOWED_CODE_EXTENSIONS = {
    "", ".py", ".txt", ".json", ".yaml", ".yml", ".cfg", ".ini", ".toml",
    ".md", ".csv", ".tsv", ".sh", ".env", ".in", ".requirements", ".gitignore",
}

# Archive extensions accepted by the project zip-upload path.
ALLOWED_ARCHIVE_EXTENSIONS = {".zip"}

# Hard cap on a single pasted/uploaded code file (10 MB) — code is small.
MAX_CODE_FILE_BYTES = 10 * 1024 * 1024

# Network binding. HPC nodes are typically reached over an internal network.
HOST = os.environ.get("HOSTA100_HOST", "0.0.0.0")
PORT = int(os.environ.get("HOSTA100_PORT", "8198"))

# =========================================================================== #
# SLURM dispatch (Cách B): each run job is launched through `srun` so it lands
# on a GPU compute node, while the Flask app stays on the login node. Only takes
# effect if `srun` is on PATH — otherwise jobs run locally (Windows dev, or when
# the app already runs inside an allocation).
#
# >>> SỬA THẲNG Ở ĐÂY <<<  These ARE the defaults baked into the code — you do
# NOT need to pass any env var; plain `python3 app.py` already uses them. Just
# edit the values below if you want different ones. (The HOSTA100_* env vars are
# only an optional override, e.g. for a one-off run without editing the file.)
# =========================================================================== #
DEFAULT_USE_SLURM = True
DEFAULT_SLURM_GRES = "gpu:ampere:1"   # arch family, NOT a model: ampere = A100 OR A40.
                                      # Pick the exact model per-job via the UI's
                                      # "Loại GPU" dropdown (adds --constraint=<model>).
DEFAULT_SLURM_TIME = "04:00:00"       # max wall time per job (HH:MM:SS); "" = cluster default
# The GPU partition this account can actually submit to (AllowAccounts=ALL).
# `main` is the catch-all (CPU+GPU) where GPU jobs queue behind everything; the
# idle GPUs (debug / research-group partitions) are NOT accessible to us. Pinning
# to main-gpu targets the right partition AND lets the dashboard show only the
# GPUs we can really get. "" = don't pass --partition (cluster default).
DEFAULT_SLURM_PARTITION = "main-gpu"
DEFAULT_SLURM_ACCOUNT = ""            # set if it requires -A <account>
DEFAULT_SLURM_CPUS = ""               # e.g. "4"
DEFAULT_SLURM_MEM = ""                # e.g. "16G"


def _env_bool(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# Effective values = the baked-in default, optionally overridden by an env var.
USE_SLURM = _env_bool("HOSTA100_USE_SLURM", DEFAULT_USE_SLURM)
SLURM_GRES = os.environ.get("HOSTA100_SLURM_GRES", DEFAULT_SLURM_GRES)
SLURM_TIME = os.environ.get("HOSTA100_SLURM_TIME", DEFAULT_SLURM_TIME)
SLURM_PARTITION = os.environ.get("HOSTA100_SLURM_PARTITION", DEFAULT_SLURM_PARTITION)
SLURM_ACCOUNT = os.environ.get("HOSTA100_SLURM_ACCOUNT", DEFAULT_SLURM_ACCOUNT)
SLURM_CPUS = os.environ.get("HOSTA100_SLURM_CPUS", DEFAULT_SLURM_CPUS)
SLURM_MEM = os.environ.get("HOSTA100_SLURM_MEM", DEFAULT_SLURM_MEM)

# Flask secret key (sessions / flash messages).
SECRET_KEY = os.environ.get("HOSTA100_SECRET", "change-me-on-the-hpc-server")

# --------------------------------------------------------------------------- #
# Access control. A single shared password gates the whole app. REQUIRED before
# exposing the app publicly (e.g. via a cloudflare tunnel), because "run my
# project code" executes arbitrary Python under your account.
#
# >>> SỬA THẲNG Ở ĐÂY <<< đặt mật khẩu vào DEFAULT_PASSWORD (hoặc biến môi
# trường HOSTA100_PASSWORD). Để rỗng = TẮT đăng nhập (chỉ dùng khi test nội bộ,
# KHÔNG public). Khi tắt, app sẽ in cảnh báo lúc khởi động.
# --------------------------------------------------------------------------- #
DEFAULT_PASSWORD = ""
APP_PASSWORD = os.environ.get("HOSTA100_PASSWORD", DEFAULT_PASSWORD)
AUTH_ENABLED = bool(APP_PASSWORD)


def ensure_dirs():
    """Create all required data directories. Safe to call repeatedly."""
    print(f"[host-a100] Using data directory: {DATA_DIR}")
    for path in (DATA_DIR, MODELS_DIR, ENVS_DIR, JOBS_DIR, RESULTS_DIR,
                 PROJECTS_DIR, PIP_CACHE_DIR):
        os.makedirs(path, exist_ok=True)
