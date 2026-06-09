"""
Model management: streaming uploads, listing, deletion.

Uploads run in a background thread so the Flask request thread is never blocked.
The route hands us an already-open temp file (Werkzeug streams large uploads to
a SpooledTemporaryFile/disk), and we copy it into /data/models/<name>/ while
reporting progress through utils.progress.
"""
import os
import shutil
import threading

import config
from services import storage_service as db
from utils import file_utils, progress

_CHUNK = 8 * 1024 * 1024  # 8 MB copy chunks


def list_models():
    rows = db.execute("SELECT * FROM models ORDER BY created_at DESC",
                      fetch="all") or []
    models = []
    for r in rows:
        d = dict(r)
        d["size_h"] = file_utils.human_size(d["size"])
        d["exists"] = os.path.exists(d["path"])
        models.append(d)
    return models


def get_model(model_id):
    row = db.execute("SELECT * FROM models WHERE id=?", (model_id,),
                     fetch="one")
    return dict(row) if row else None


def name_taken(name):
    return db.execute("SELECT 1 FROM models WHERE name=?", (name,),
                      fetch="one") is not None


def start_upload(upload_id, model_name, temp_path, filename):
    """
    Launch a background thread that moves an already-received temp file into the
    model directory and registers it in the DB.

    WSGI note: the browser->server transfer is read by the request thread (the
    request stream is closed once the view returns, so it cannot be read here).
    The route streams the upload into `temp_path` and reports that phase via the
    browser's XHR upload progress. This background worker then performs the
    server-side finalize (move/copy + DB insert) and reports it through polling.
    """
    model_name = file_utils.validate_name(model_name)
    filename = file_utils.safe_filename(filename)

    def worker():
        try:
            model_dir = file_utils.safe_join(config.MODELS_DIR, model_name)
            os.makedirs(model_dir, exist_ok=True)
            dest = file_utils.safe_join(model_dir, filename)

            progress.update("upload", upload_id, status="running", progress=0,
                            step="finalizing", append_log=f"-> {dest}\n")
            total = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0

            try:
                # Fast path: same filesystem -> instantaneous rename.
                os.replace(temp_path, dest)
                progress.update("upload", upload_id, progress=95,
                                step="stored (moved)")
            except OSError:
                # Cross-filesystem: copy in chunks with progress.
                copied = 0
                with open(temp_path, "rb") as src, open(dest, "wb") as out:
                    while True:
                        chunk = src.read(_CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
                        copied += len(chunk)
                        pct = min(95, int(copied * 100 / total)) if total else 50
                        progress.update("upload", upload_id, progress=pct,
                                        step=f"copied {file_utils.human_size(copied)}")
                os.remove(temp_path)

            size = file_utils.dir_size(model_dir)
            db.execute(
                "INSERT INTO models (name, path, size, created_at) "
                "VALUES (?, ?, ?, ?)",
                (model_name, model_dir, size, db.now()),
                commit=True,
            )
            progress.update("upload", upload_id, progress=100, status="done",
                            step="complete", append_log="Upload registered.\n")
        except Exception as exc:  # noqa: BLE001 - surface to UI
            progress.update("upload", upload_id, status="error",
                            step="failed", append_log=f"ERROR: {exc}\n")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    threading.Thread(target=worker, daemon=True).start()


def delete_model(model_id):
    """Delete a model's files (confined to /data/models) and its DB row."""
    model = get_model(model_id)
    if not model:
        return False
    path = model["path"]
    if path and file_utils.is_within(config.MODELS_DIR, path) \
            and os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    db.execute("DELETE FROM models WHERE id=?", (model_id,), commit=True)
    return True


def model_file(model_id):
    """
    Return (directory, filename) of the single primary file for download, or
    None. If the model dir has multiple files we return the directory only so
    the route can zip or refuse.
    """
    model = get_model(model_id)
    if not model or not os.path.isdir(model["path"]):
        return None
    files = [f for f in os.listdir(model["path"])
             if os.path.isfile(os.path.join(model["path"], f))]
    if len(files) == 1:
        return model["path"], files[0]
    return model["path"], None
