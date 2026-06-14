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
# One subdir per API-farm server (sbatch script, slurm.out, endpoint.json, log).
SERVERS_DIR = os.path.join(DATA_DIR, "servers")

# --------------------------------------------------------------------------- #
# Tools → Clone giọng nói (OmniVoice). Mirrors the API farm: a long-running
# OmniVoice TTS server runs on a GPU compute node via sbatch and the login-node
# app talks to it for synthesis. All paths live on the shared FS so the compute
# node (no internet) can read the model weights + voice references the login node
# wrote.
# --------------------------------------------------------------------------- #
# One subdir per OmniVoice server (run.sh, slurm.out, endpoint.json, server.log).
VOICE_SERVERS_DIR = os.path.join(DATA_DIR, "voice-servers")
# Per-job working dir: per-chunk wavs + the final MP3 + SRT.
VOICE_OUTPUTS_DIR = os.path.join(DATA_DIR, "voice-outputs")
# Cloned-voice profiles (one subdir per voice: reference.wav + profile.json).
VOICES_DIR = os.path.join(DATA_DIR, "voices")
# Shared HuggingFace cache. The OmniVoice weights MUST be downloaded here on the
# login node first (compute nodes have no internet); the server script exports
# HF_HOME=this so it loads from cache. Override with HOSTA100_HF_HOME.
OMNI_HF_HOME = os.environ.get("HOSTA100_HF_HOME", os.path.join(DATA_DIR, "hf-cache"))
# Default OmniVoice model id (resolved from the shared HF cache above).
OMNI_MODEL_ID = os.environ.get("HOSTA100_OMNI_MODEL", "k2-fsa/OmniVoice")
# True GPU batching: how many utterances to feed the model in ONE generate() call
# (the login node also sends this many chunks per request). Bigger = more VRAM
# used + higher throughput on the L40S, but risks OOM if too large. The server
# tries a batched call and falls back to per-utterance if the model won't batch.
# "1" effectively disables batching. Raising this REQUIRES recreating the server.
OMNI_MAX_BATCH = os.environ.get("HOSTA100_OMNI_MAX_BATCH", "8")
OMNI_BATCH_ENABLED = os.environ.get("HOSTA100_OMNI_BATCH", "1")  # "0" = off
# The GPU-side server script (lives in the repo, shipped by git). The batch
# script runs it with the env's python on the compute node.
OMNI_SERVER_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scripts", "omnivoice_server.py")

# Native llama.cpp server binary (`llama-server` from ggml-org). Used by the
# API-farm "llama-server (native, --jinja)" engine, which has a built-in
# tool-call parser + reads the GGUF's embedded chat template — so models like
# Qwen3-Coder actually emit OpenAI tool_calls (Claude Code can edit files). The
# Python `llama_cpp.server` engine does NOT ship this binary. It must live on the
# shared FS so compute nodes can exec it. Override with HOSTA100_LLAMA_SERVER_BIN.
LLAMA_SERVER_BIN = os.environ.get(
    "HOSTA100_LLAMA_SERVER_BIN", os.path.join(DATA_DIR, "bin", "llama-server"))

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

# Optional alert webhook. When an API-farm server dies for real (auto-resubmit
# circuit breaker trips, sbatch fails, or a job ends without ever serving), the
# app POSTs a JSON {"content","text"} message here — works with a Discord/Slack
# incoming webhook URL or any generic endpoint. "" = alerts disabled (default).
# >>> Dán URL webhook Discord/Slack vào đây (hoặc đặt env HOSTA100_ALERT_WEBHOOK).
DEFAULT_ALERT_WEBHOOK = ""
ALERT_WEBHOOK = os.environ.get("HOSTA100_ALERT_WEBHOOK", DEFAULT_ALERT_WEBHOOK)

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
                 PROJECTS_DIR, PIP_CACHE_DIR, SERVERS_DIR,
                 VOICE_SERVERS_DIR, VOICE_OUTPUTS_DIR, VOICES_DIR, OMNI_HF_HOME):
        os.makedirs(path, exist_ok=True)


# --------------------------------------------------------------------------- #
# ffmpeg / ffprobe discovery. The voice tool stitches the per-chunk audio,
# adjusts speed, denoises and exports MP3 entirely with ffmpeg on the LOGIN node
# (no GPU needed for any of that). Prefer an explicit override, then a `ffmpeg/`
# folder next to the app, then PATH.
# --------------------------------------------------------------------------- #
import shutil as _shutil  # noqa: E402  (kept local to these helpers)

_APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_bin(name, env_var):
    explicit = os.environ.get(env_var)
    if explicit and os.path.isfile(explicit):
        return explicit
    local = os.path.join(_APP_DIR, "ffmpeg", name)
    if os.path.isfile(local):
        return local
    if os.name == "nt":
        local_exe = local + ".exe"
        if os.path.isfile(local_exe):
            return local_exe
    return _shutil.which(name) or name


def ffmpeg_path():
    return _find_bin("ffmpeg", "HOSTA100_FFMPEG")


def ffprobe_path():
    return _find_bin("ffprobe", "HOSTA100_FFPROBE")


def ffmpeg_available():
    return bool(_shutil.which(ffmpeg_path()) or os.path.isfile(ffmpeg_path()))
