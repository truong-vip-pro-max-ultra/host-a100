"""
Project (code) management.

A project is a directory under /data/projects/<name>/ holding the user's own
code + config files. Users populate it by pasting file contents in the browser
or by uploading individual files or a .zip archive. A job later runs the chosen
main file inside a selected virtual environment.

All file paths are validated through file_utils.safe_relpath so nothing can be
written outside the project directory (path traversal / zip-slip proof).
"""
import os
import shutil

import config
from services import storage_service as db
from utils import file_utils


def list_projects():
    rows = db.execute("SELECT * FROM projects ORDER BY created_at DESC",
                      fetch="all") or []
    projects = []
    for r in rows:
        d = dict(r)
        d["exists"] = os.path.isdir(d["path"])
        d["size"] = file_utils.dir_size(d["path"])
        d["file_count"] = len(list_files(d["path"]))
        projects.append(d)
    return projects


def get_project(project_id):
    row = db.execute("SELECT * FROM projects WHERE id=?", (project_id,),
                     fetch="one")
    return dict(row) if row else None


def name_taken(name):
    return db.execute("SELECT 1 FROM projects WHERE name=?", (name,),
                      fetch="one") is not None


def create_project(name):
    name = file_utils.validate_name(name)
    project_dir = file_utils.safe_join(config.PROJECTS_DIR, name)
    os.makedirs(project_dir, exist_ok=True)
    return db.execute(
        "INSERT INTO projects (name, path, created_at) VALUES (?, ?, ?)",
        (name, project_dir, db.now()), commit=True,
    )


def list_files(project_dir):
    """Return a sorted list of {rel, size} for every file in the project."""
    out = []
    if not os.path.isdir(project_dir):
        return out
    for root, _dirs, files in os.walk(project_dir):
        for f in files:
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, project_dir).replace("\\", "/")
            try:
                size = os.path.getsize(fp)
            except OSError:
                size = 0
            out.append({"rel": rel, "size": size})
    out.sort(key=lambda x: x["rel"])
    return out


def python_files(project_dir):
    return [f["rel"] for f in list_files(project_dir)
            if f["rel"].lower().endswith(".py")]


def _check_ext(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in config.ALLOWED_CODE_EXTENSIONS:
        raise ValueError(f"Loại file '{ext or '(không có)'}' không được phép.")


def write_text_file(project_id, relpath, content):
    """Create/overwrite a text file (pasted in the browser) inside a project."""
    project = get_project(project_id)
    if not project:
        raise ValueError("Không tìm thấy dự án.")
    if content is None:
        content = ""
    if len(content.encode("utf-8")) > config.MAX_CODE_FILE_BYTES:
        raise ValueError("File quá lớn.")
    abspath, rel = file_utils.safe_relpath(project["path"], relpath)
    _check_ext(rel)
    os.makedirs(os.path.dirname(abspath), exist_ok=True)
    with open(abspath, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
    return rel


def read_text_file(project_id, relpath):
    project = get_project(project_id)
    if not project:
        raise ValueError("Không tìm thấy dự án.")
    abspath, _rel = file_utils.safe_relpath(project["path"], relpath)
    if not os.path.isfile(abspath):
        raise ValueError("Không tìm thấy file.")
    with open(abspath, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def save_uploaded_file(project_id, filename, stream, subdir=""):
    """Save an uploaded code file (FileStorage stream) into the project."""
    project = get_project(project_id)
    if not project:
        raise ValueError("Không tìm thấy dự án.")
    safe_name = file_utils.safe_filename(filename)
    rel = f"{subdir.strip('/')}/{safe_name}" if subdir.strip("/") else safe_name
    abspath, rel = file_utils.safe_relpath(project["path"], rel)
    _check_ext(rel)
    os.makedirs(os.path.dirname(abspath), exist_ok=True)
    size = 0
    with open(abspath, "wb") as out:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > config.MAX_CODE_FILE_BYTES:
                out.close()
                os.remove(abspath)
                raise ValueError(f"{safe_name} vượt quá giới hạn dung lượng file code.")
            out.write(chunk)
    return rel


def extract_zip(project_id, zip_temp_path):
    """Extract an uploaded .zip into the project (zip-slip safe)."""
    project = get_project(project_id)
    if not project:
        raise ValueError("Không tìm thấy dự án.")
    try:
        extracted = file_utils.extract_zip_safely(zip_temp_path, project["path"])
    finally:
        if os.path.exists(zip_temp_path):
            try:
                os.remove(zip_temp_path)
            except OSError:
                pass
    return extracted


def delete_file(project_id, relpath):
    project = get_project(project_id)
    if not project:
        return False
    abspath, _rel = file_utils.safe_relpath(project["path"], relpath)
    if os.path.isfile(abspath):
        os.remove(abspath)
        return True
    return False


def set_main_file(project_id, relpath):
    project = get_project(project_id)
    if not project:
        raise ValueError("Không tìm thấy dự án.")
    abspath, rel = file_utils.safe_relpath(project["path"], relpath)
    if not os.path.isfile(abspath):
        raise ValueError("File chính không tồn tại trong dự án.")
    db.execute("UPDATE projects SET main_file=? WHERE id=?", (rel, project_id),
               commit=True)
    return rel


def delete_project(project_id):
    project = get_project(project_id)
    if not project:
        return False
    path = project["path"]
    if path and file_utils.is_within(config.PROJECTS_DIR, path) \
            and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    db.execute("DELETE FROM projects WHERE id=?", (project_id,), commit=True)
    return True


def find_requirements(project_dir):
    """Return the relpath of a requirements.txt in the project, if present."""
    for f in list_files(project_dir):
        if os.path.basename(f["rel"]).lower() == "requirements.txt":
            return f["rel"]
    return None
