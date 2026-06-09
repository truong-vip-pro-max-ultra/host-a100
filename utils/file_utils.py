"""
Filesystem safety helpers.

These functions are the single source of truth for validating user-supplied
names and for confining every path inside the configured /data tree. They are
the main defense against path traversal and command injection via filenames.
"""
import os
import re

# A conservative whitelist: letters, digits, dot, dash, underscore.
# No slashes, no spaces, no shell metacharacters.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class UnsafeName(ValueError):
    """Raised when a user-supplied name fails validation."""


def validate_name(name):
    """
    Validate a model / environment / job name.

    Returns the cleaned name or raises UnsafeName. Rejects path separators,
    parent references, leading dots and anything outside the whitelist.
    """
    if not name or not isinstance(name, str):
        raise UnsafeName("Tên là bắt buộc.")
    name = name.strip()
    if name in (".", "..") or "/" in name or "\\" in name:
        raise UnsafeName("Tên không được chứa ký tự phân cách đường dẫn.")
    if not _NAME_RE.match(name):
        raise UnsafeName(
            "Tên chỉ được gồm chữ, số, '.', '-', '_' và phải bắt đầu bằng "
            "chữ hoặc số (tối đa 128 ký tự)."
        )
    return name


def safe_filename(filename):
    """
    Reduce an uploaded filename to a safe basename.

    Strips any directory components and keeps only a whitelisted basename.
    Raises UnsafeName if nothing usable remains.
    """
    if not filename:
        raise UnsafeName("Tên file là bắt buộc.")
    # Take basename only — defeats ../ and absolute paths.
    base = os.path.basename(filename.replace("\\", "/"))
    base = base.strip()
    if base in ("", ".", ".."):
        raise UnsafeName("Tên file không hợp lệ.")
    # Collapse anything not whitelisted into underscores, keep the extension.
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    if not cleaned or cleaned in (".", ".."):
        raise UnsafeName("Tên file không hợp lệ.")
    return cleaned


# A single path component inside a project (filename or directory name).
_PART_RE = re.compile(r"^[A-Za-z0-9._][A-Za-z0-9._-]*$")


def safe_relpath(base_dir, relpath):
    """
    Validate a user-supplied RELATIVE path (may contain sub-directories) and
    return (absolute_path, normalized_relpath) confined inside base_dir.

    Allows nested dirs like "src/utils.py" but rejects absolute paths, parent
    references ('..') and any component with unexpected characters. This is the
    anti zip-slip / anti path-traversal gate for project code files.
    """
    if not relpath or not isinstance(relpath, str):
        raise UnsafeName("Đường dẫn là bắt buộc.")
    cleaned = relpath.strip().replace("\\", "/").lstrip("/")
    parts = [p for p in cleaned.split("/") if p not in ("", ".")]
    if not parts:
        raise UnsafeName("Đường dẫn không hợp lệ.")
    for p in parts:
        if p == ".." or not _PART_RE.match(p):
            raise UnsafeName(f"Thành phần đường dẫn không hợp lệ: {p!r}")
    abspath = safe_join(base_dir, *parts)  # double-checks confinement
    return abspath, "/".join(parts)


def extract_zip_safely(zip_path, dest_dir):
    """
    Extract a zip into dest_dir, skipping any member that would escape it
    (zip-slip) or that is a symlink. Returns the list of extracted relpaths.
    """
    import zipfile
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            name = member.filename
            if not name or name.endswith("/"):
                continue  # directory entry
            # Reject absolute / traversal members outright.
            norm = name.replace("\\", "/")
            if norm.startswith("/") or ".." in norm.split("/"):
                continue
            try:
                target, rel = safe_relpath(dest_dir, norm)
            except UnsafeName:
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as out:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            extracted.append(rel)
    return extracted


def is_within(base_dir, target_path):
    """Return True iff target_path resolves to a location inside base_dir."""
    base = os.path.realpath(base_dir)
    target = os.path.realpath(target_path)
    try:
        return os.path.commonpath([base, target]) == base
    except ValueError:
        # Different drives on Windows, etc.
        return False


def safe_join(base_dir, *parts):
    """
    Join parts under base_dir and guarantee the result stays inside base_dir.

    Raises UnsafeName on any traversal attempt.
    """
    candidate = os.path.join(base_dir, *parts)
    if not is_within(base_dir, candidate):
        raise UnsafeName("Đường dẫn vượt ra ngoài thư mục dữ liệu cho phép.")
    return candidate


def dir_size(path):
    """Total size in bytes of a file or directory tree. Missing -> 0."""
    if not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def human_size(num_bytes):
    """Format a byte count as a human-readable string."""
    num = float(num_bytes or 0)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if num < 1024.0:
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024.0
    return f"{num:.1f} EB"
