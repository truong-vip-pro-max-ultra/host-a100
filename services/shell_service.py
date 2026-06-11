"""
Interactive shell tab — runs commands on the host like an SSH session.

The whole app is gated by a shared password and the jobs feature already runs
arbitrary user Python, so giving the authenticated OWNER a command console is
consistent with the existing trust model. Each call is request/response, exactly
like `ssh host "<command>"`: there is no persistent TTY, so long-running
interactive programs (top, vim, a server in the foreground) are not supported —
use the Jobs page for long work.

By default commands run on the login node (where the Flask app lives). An
optional toggle dispatches them to a compute node through `srun`, reusing the
same SLURM mechanism as run jobs.

IMPORTANT: this is a REAL shell for the trusted owner (shell=True), NOT a
sandbox. Do not expose the app without a password (config.AUTH_ENABLED).
"""
import os
import re
import shlex
import subprocess

import config
from services import job_service

# Timeout for a single command, in seconds. This is a one-shot request/response
# console, so a command that needs longer should be submitted as a job instead.
DEFAULT_TIMEOUT = 60

# Cap the returned output so a runaway `cat hugefile` can't blow up the page /
# the response. Keep the TAIL, which is usually the interesting part.
MAX_OUTPUT_CHARS = 200_000


def initial_cwd():
    """Where a fresh terminal session starts."""
    return config.DATA_DIR if os.path.isdir(config.DATA_DIR) else os.path.expanduser("~")


def _pure_cd_target(command):
    """If `command` is just a bare `cd [path]`, return its target path.

    A `cd` is handled in Python (not the shell) because every shell=True call is
    a brand-new shell with its own cwd — without this, `cd` would never persist
    across requests. Returns None when the command is anything other than a
    standalone `cd` (so e.g. `cd foo && ls` still runs normally in the shell).
    """
    stripped = command.strip()
    if stripped == "cd" or stripped == "cd ~":
        return os.path.expanduser("~")
    if stripped.startswith("cd ") and not any(c in stripped for c in ("&", "|", ";", "\n")):
        arg = stripped[3:].strip()
        try:
            parts = shlex.split(arg)
        except ValueError:
            parts = [arg]
        if len(parts) == 1:
            return os.path.expanduser(parts[0])
    return None


def _clip(text):
    if len(text) > MAX_OUTPUT_CHARS:
        cut = len(text) - MAX_OUTPUT_CHARS
        return (f"[... đã cắt bớt {cut} ký tự ở đầu ...]\n"
                + text[-MAX_OUTPUT_CHARS:])
    return text


# Interactive programs that can't run without a TTY. Editors/pagers are refused
# with a hint; top/htop are rewritten to a one-shot batch snapshot.
_NO_TTY_HINT = {
    "vi": "sửa file bằng trang Dự án, hoặc xem bằng `cat`/`sed -n '1,40p'`",
    "vim": "sửa file bằng trang Dự án, hoặc xem bằng `cat`/`sed -n '1,40p'`",
    "nano": "sửa file bằng trang Dự án, hoặc xem bằng `cat`",
    "emacs": "sửa file bằng trang Dự án",
    "less": "dùng `cat`, hoặc `sed -n '1,100p' <file>`",
    "more": "dùng `cat`, hoặc `sed -n '1,100p' <file>`",
    "man": "dùng `<lệnh> --help`",
    "watch": "bỏ `watch`, chạy thẳng lệnh một lần (trang tự không lặp lại)",
}


def _handle_interactive(command):
    """Adjust commands that assume a TTY. Returns (note, command).

    * note=None: nothing special, run `command` as given.
    * note set, command set: run the rewritten `command`, prefix output with note.
    * note set, command=None: do NOT run — `note` is the message to show.

    Only simple commands are touched; anything with a pipe/redirect/chain runs
    untouched so the user's own pipelines are never rewritten.
    """
    if any(c in command for c in ("|", "&", ";", "<", ">", "`", "\n", "$(")):
        return None, command
    parts = command.split()
    if not parts:
        return None, command
    first = parts[0]

    if first in ("top", "htop") and "-b" not in parts:
        # No TTY → run a single batch iteration that prints and exits. htop has
        # no batch mode, so fall back to top for it.
        args = [] if first == "htop" else parts[1:]
        new = " ".join(["top", "-b", "-n", "1"] + args)
        return (f"[{first} cần TTY — đang hiển thị ảnh chụp một lần: `{new}`]\n", new)

    if first == "tail" and "-f" in parts:
        return ("[`tail -f` theo dõi liên tục nên sẽ treo ở terminal này. "
                "Bỏ `-f` để xem phần cuối file một lần.]\n", None)

    if first in _NO_TTY_HINT:
        return (f"[`{first}` là chương trình tương tác cần TTY — terminal này là "
                f"một-lệnh-một-kết-quả nên không hỗ trợ. Gợi ý: "
                f"{_NO_TTY_HINT[first]}.]\n", None)

    return None, command


def run_command(command, cwd, use_gpu=False, on_compute=False,
                timeout=DEFAULT_TIMEOUT):
    """Run one command and return a dict the JSON endpoint hands to the UI.

    Keys: cwd (possibly updated by a `cd`), output (combined stdout+stderr),
    exit_code, and where ("login" or "compute") the command actually ran.
    """
    command = (command or "").strip()
    if not cwd or not os.path.isdir(cwd):
        cwd = initial_cwd()
    if not command:
        return {"cwd": cwd, "output": "", "exit_code": 0, "where": "login"}

    # Handle a standalone `cd` ourselves so navigation sticks between commands.
    target = _pure_cd_target(command)
    if target is not None:
        new = target if os.path.isabs(target) \
            else os.path.normpath(os.path.join(cwd, target))
        if os.path.isdir(new):
            return {"cwd": new, "output": "", "exit_code": 0, "where": "login"}
        return {"cwd": cwd, "exit_code": 1, "where": "login",
                "output": f"cd: {target}: Không có thư mục như vậy\n"}

    # Handle programs that need a TTY. This is a one-shot request/response with
    # no terminal, so `top`/`htop` would just run forever until the timeout and
    # show nothing, and editors/pagers can't work at all. We only touch SIMPLE
    # commands (no pipe/redirect/chain) so real pipelines are never mangled.
    note, command = _handle_interactive(command)
    if note and command is None:
        return {"cwd": cwd, "output": note, "exit_code": 1, "where": "login"}

    # Decide where to run. `srun` wants an argv vector, not a shell string, so on
    # a compute node we wrap the command in `bash -lc "<command>"`. _srun_prefix
    # returns [] when SLURM is unavailable/off, so we transparently fall back to
    # local execution.
    prefix = job_service._srun_prefix("shell", cwd, use_gpu=use_gpu) if on_compute else []
    if prefix:
        argv, use_shell, where = prefix + ["bash", "-lc", command], False, "compute"
    else:
        argv, use_shell, where = command, True, "login"

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.run(
            argv, shell=use_shell, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, text=True, encoding="utf-8", errors="replace",
        )
        output, code = proc.stdout or "", proc.returncode
        if note:
            output = note + output
    except subprocess.TimeoutExpired as exc:
        partial = exc.output or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", "replace")
        output = partial + f"\n[Lệnh vượt quá {timeout}s và đã bị huỷ]\n"
        code = 124
    except FileNotFoundError as exc:
        output, code = f"[Không tìm thấy lệnh: {exc}]\n", 127
    except Exception as exc:  # noqa: BLE001
        output, code = f"[Lỗi khi chạy lệnh: {exc}]\n", 1

    return {"cwd": cwd, "output": _clip(output), "exit_code": code, "where": where}


# Cap how many completion candidates we return, so a huge directory can't flood
# the response / the console.
MAX_COMPLETIONS = 200


def complete(text, cwd):
    """Filesystem tab-completion for the LAST token of `text`, relative to cwd.

    Returns {"matches": [...], "token": "<last token>"}. Each match is a ready
    drop-in replacement for the token — it keeps whatever directory prefix the
    user already typed (including `~/` or an absolute path) and appends a `/` to
    directories so the user can keep tabbing into them.
    """
    if not cwd or not os.path.isdir(cwd):
        cwd = initial_cwd()
    # Last whitespace-separated token (the thing being completed).
    token = re.split(r"\s+", text)[-1] if text else ""

    # Split the token into the directory part the user typed (kept verbatim) and
    # the prefix to match against.
    cut = token.rfind("/")
    dir_part = token[:cut + 1] if cut >= 0 else ""   # keeps trailing slash
    prefix = token[cut + 1:]

    search = os.path.expanduser(dir_part) if dir_part else ""
    base = search if os.path.isabs(search) else os.path.join(cwd, search or ".")
    try:
        entries = os.listdir(base)
    except OSError:
        return {"matches": [], "token": token}

    matches = []
    for name in sorted(entries):
        if not name.startswith(prefix):
            continue
        # Hide dotfiles unless the user explicitly started the prefix with a dot.
        if name.startswith(".") and not prefix.startswith("."):
            continue
        suffix = "/" if os.path.isdir(os.path.join(base, name)) else ""
        matches.append(dir_part + name + suffix)
        if len(matches) >= MAX_COMPLETIONS:
            break
    return {"matches": matches, "token": token}
