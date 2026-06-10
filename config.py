"""
Central configuration for the ML platform.

All paths live under DATA_DIR. On the target Linux HPC server this defaults to
/data. For local development (e.g. Windows) you can override every path by
setting the HOSTA100_DATA_DIR environment variable.
"""
import os

# Root data directory. Override with HOSTA100_DATA_DIR for local testing.
DATA_DIR = os.environ.get("HOSTA100_DATA_DIR", "/data")

MODELS_DIR = os.path.join(DATA_DIR, "models")
ENVS_DIR = os.path.join(DATA_DIR, "envs")
JOBS_DIR = os.path.join(DATA_DIR, "jobs")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
PROJECTS_DIR = os.path.join(DATA_DIR, "projects")

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

# Flask secret key (sessions / flash messages).
SECRET_KEY = os.environ.get("HOSTA100_SECRET", "change-me-on-the-hpc-server")


def ensure_dirs():
    """Create all required data directories. Safe to call repeatedly."""
    for path in (DATA_DIR, MODELS_DIR, ENVS_DIR, JOBS_DIR, RESULTS_DIR,
                 PROJECTS_DIR):
        os.makedirs(path, exist_ok=True)
