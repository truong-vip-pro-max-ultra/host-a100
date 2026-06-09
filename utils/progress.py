"""
In-memory progress registry for background tasks (uploads & jobs).

Because the app runs as a SINGLE Flask process (python3 app.py) with background
threads, a module-level dict guarded by a lock is a perfectly safe and simple
shared store. The polling endpoints read from here.

Each entry:
    {
        "progress": int 0-100,
        "step": str,         # short human-readable phase
        "log": str,          # rolling tail of log lines
        "status": str,       # running | done | error
        "updated_at": float, # epoch seconds
    }
"""
import threading
import time

_lock = threading.Lock()
_registry = {}

# Keep only the last N characters of log per task to bound memory.
_MAX_LOG_CHARS = 20000


def _key(kind, task_id):
    return f"{kind}:{task_id}"


def init(kind, task_id, step="queued"):
    """Register a new task in the 'queued' state."""
    with _lock:
        _registry[_key(kind, task_id)] = {
            "progress": 0,
            "step": step,
            "log": "",
            "status": "queued",
            "updated_at": time.time(),
        }


def update(kind, task_id, progress=None, step=None, status=None, append_log=None):
    """Update one or more fields of a task's progress entry."""
    k = _key(kind, task_id)
    with _lock:
        entry = _registry.get(k)
        if entry is None:
            entry = {
                "progress": 0, "step": "", "log": "",
                "status": "running", "updated_at": time.time(),
            }
            _registry[k] = entry
        if progress is not None:
            entry["progress"] = max(0, min(100, int(progress)))
        if step is not None:
            entry["step"] = step
        if status is not None:
            entry["status"] = status
        if append_log:
            entry["log"] = (entry["log"] + append_log)[-_MAX_LOG_CHARS:]
        entry["updated_at"] = time.time()


def get(kind, task_id):
    """Return a snapshot copy of a task's progress, or a default if unknown."""
    with _lock:
        entry = _registry.get(_key(kind, task_id))
        if entry is None:
            return {
                "progress": 0, "step": "unknown", "log": "",
                "status": "unknown", "updated_at": 0,
            }
        return dict(entry)


def active():
    """Return list of (kind, id, entry) for tasks currently running/queued."""
    out = []
    with _lock:
        for k, v in _registry.items():
            if v["status"] in ("running", "queued"):
                kind, task_id = k.split(":", 1)
                out.append((kind, task_id, dict(v)))
    return out
