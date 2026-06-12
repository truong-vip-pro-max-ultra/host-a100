"""
Interactive PTY terminal backing the WebSocket at /terminal/pty.

Unlike shell_service (one-shot, no TTY), this spawns a real login shell inside a
pseudo-terminal so curses/raw-mode programs work: vim, nano, htop, less, the
python REPL, interactive prompts, native tab-completion, etc. A reader thread
pumps PTY output to the browser (xterm.js) and the WebSocket handler writes the
browser's keystrokes back into the PTY. Optionally the shell is launched on a
compute node via `srun --pty` (reusing the job dispatch flags).

Trust model is identical to shell_service: this is an OWNER-ONLY power tool that
runs commands as the app's user — NOT a sandbox. The WebSocket route is gated by
the same auth as everything else.

Linux only (needs os/pty fork). On Windows dev `available()` returns False and
the terminal page shows a notice instead.
"""
import json
import os
import select
import shlex
import struct
import subprocess
import sys
import tempfile
import threading

import config
from services import job_service

try:
    import pty
    import fcntl
    import termios
except ImportError:  # pragma: no cover - non-POSIX (Windows dev)
    pty = None


def available():
    """True when a PTY shell can actually be spawned (POSIX with pty)."""
    return pty is not None and sys.platform != "win32" and hasattr(pty, "openpty")


def _set_winsize(fd, rows, cols):
    """Tell the PTY its window size so full-screen apps lay out correctly."""
    try:
        fcntl.ioctl(fd, termios.TIOCSWINSZ,
                    struct.pack("HHHH", rows, cols, 0, 0))
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Terminal guard.
#
# The PTY is a REAL interactive shell, so the cleanest way to "filter at the
# backend" is to launch bash with an rcfile WE write. The rcfile first rebuilds
# the normal login environment (so conda/modules/PATH still work), then installs
# shell functions that:
#   * `ls` — when listing the platform's own source directory, reveal ONLY the
#     data directory (host-a100-data); the code dirs and app.py stay hidden.
#   * `kill`/`pkill`/`killall` — refuse to signal the app's own process (PID /
#     process group) or the cloudflare tunnel.
# This is defence for the common case on an OWNER tool, not an airtight sandbox
# (`/bin/ls`, `command kill` still bypass it) — consistent with this terminal
# already being a non-sandboxed power tool.
# --------------------------------------------------------------------------- #


def _app_dir():
    """Absolute path of the project root (the directory holding config.py)."""
    return os.path.dirname(os.path.abspath(config.__file__))


def _visible_in_app_dir():
    """Names to reveal when listing the app dir — just the data dir, if it
    actually lives inside the project tree (otherwise nothing to show)."""
    app = os.path.realpath(_app_dir())
    data = os.path.realpath(config.DATA_DIR)
    if os.path.dirname(data) == app:
        return os.path.basename(data)
    return ""


def _cloudflared_pids():
    """Best-effort PIDs of any running cloudflare tunnel, so `kill <pid>` (even
    via `$(pgrep cloudflared)`) is caught, not just `pkill cloudflared`."""
    pids = set()
    try:
        import psutil  # optional; same dependency the dashboard already uses
    except Exception:  # noqa: BLE001
        psutil = None
    if psutil is not None:
        try:
            for p in psutil.process_iter(["pid", "name", "cmdline"]):
                name = (p.info.get("name") or "").lower()
                cmd = " ".join(p.info.get("cmdline") or []).lower()
                if "cloudflared" in name or "cloudflared" in cmd:
                    pids.add(str(p.info["pid"]))
        except Exception:  # noqa: BLE001
            pass
    else:  # /proc fallback (Linux, no psutil)
        try:
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    with open(f"/proc/{pid}/cmdline", "rb") as fh:
                        cmd = fh.read().replace(b"\0", b" ").decode(
                            "utf-8", "replace").lower()
                except OSError:
                    continue
                if "cloudflared" in cmd:
                    pids.add(pid)
        except OSError:
            pass
    return pids


def _protected_pids():
    """PIDs the terminal must never let the user kill: the app process, its
    process group, and the cloudflare tunnel."""
    pids = {str(os.getpid())}
    try:
        pids.add(str(os.getpgrp()))  # POSIX only; missing on Windows dev
    except (OSError, AttributeError):
        pass
    pids |= _cloudflared_pids()
    return pids


# File-reading/editing tools that must not open the platform's own source. The
# guard wraps each as a shell function; flags are skipped, only path operands are
# checked, so e.g. `cat -n file` still works on allowed files.
_READ_TOOLS = ("cat", "tac", "head", "tail", "nl", "less", "more", "view",
               "vi", "vim", "nano", "strings", "od", "xxd", "bat")


def _guard_rcfile_text():
    """Build the bash rcfile that enforces the terminal guard."""
    app_dir = shlex.quote(os.path.realpath(_app_dir()))
    data_dir = shlex.quote(os.path.realpath(config.DATA_DIR))
    visible = shlex.quote(_visible_in_app_dir())
    pid_list = " ".join(shlex.quote(p) for p in sorted(_protected_pids()))
    read_tools = " ".join(_READ_TOOLS)
    read_defs = "\n".join(
        f'{t}() {{ __ha_guard_read {t} "$@"; }}' for t in _READ_TOOLS)
    return f"""\
# host-a100 terminal guard — generated by services/pty_service.py. Do not edit;
# it is rewritten on every terminal session and removed when the session ends.

# 1) Reproduce a login shell's environment (we run with --rcfile, not -l).
[ -r /etc/profile ] && . /etc/profile
if   [ -r "$HOME/.bash_profile" ]; then . "$HOME/.bash_profile"
elif [ -r "$HOME/.bash_login" ];   then . "$HOME/.bash_login"
elif [ -r "$HOME/.profile" ];      then . "$HOME/.profile"
fi
[ -r "$HOME/.bashrc" ] && . "$HOME/.bashrc"

# 2) Guard configuration (substituted by the backend).
__HA_APP_DIR={app_dir}
__HA_DATA_DIR={data_dir}               # the only readable subtree in the app dir
__HA_VISIBLE={visible}                 # names allowed when listing the app dir
__HA_PIDS=({pid_list})                 # protected PIDs (app + cloudflare tunnel)
__HA_NAMES='cloudflared|app\\.py|host-a100'   # protected process-name patterns

unalias ls cd kill pkill killall {read_tools} 2>/dev/null

# 3) ls: when listing the platform's own directory, reveal only the data dir.
ls() {{
    local target="$PWD" a
    local -a flags=() operands=()
    for a in "$@"; do
        if [ "${{a:0:1}}" = "-" ] && [ "$a" != "-" ]; then flags+=("$a")
        else operands+=("$a"); fi
    done
    if [ "${{#operands[@]}}" -gt 0 ]; then target="${{operands[-1]}}"; fi
    # `builtin cd` so this never re-enters the guarded `cd` defined below.
    local rp; rp=$(builtin cd -- "$target" 2>/dev/null && pwd -P)
    if [ -n "$rp" ] && [ "$rp" = "$__HA_APP_DIR" ]; then
        [ -z "$__HA_VISIBLE" ] && return 0          # nothing to reveal here
        # -d so the data dir shows up as an entry instead of being descended into.
        ( builtin cd -- "$__HA_APP_DIR" 2>/dev/null && command ls -d "${{flags[@]}}" -- $__HA_VISIBLE )
        return $?
    fi
    command ls "$@"
}}

# 4) kill family: refuse the app's own PID/PGID and the cloudflare tunnel.
__ha_is_blocked() {{
    local a p
    for a in "$@"; do
        for p in "${{__HA_PIDS[@]}}"; do
            [ "$a" = "$p" ] || [ "$a" = "-$p" ] && return 0
        done
        printf '%s' "$a" | grep -Eiq "$__HA_NAMES" && return 0
    done
    return 1
}}
__ha_deny() {{
    printf '\\033[31m[host-a100] Bị chặn: không được %s tiến trình của ứng dụng hoặc cloudflare tunnel.\\033[0m\\n' "$1" >&2
    return 1
}}
kill()    {{ if __ha_is_blocked "$@"; then __ha_deny kill;    else command kill "$@";    fi; }}
pkill()   {{ if __ha_is_blocked "$@"; then __ha_deny pkill;   else command pkill "$@";   fi; }}
killall() {{ if __ha_is_blocked "$@"; then __ha_deny killall; else command killall "$@"; fi; }}

# 5) Reading/editing: refuse the platform's own source files (everything under
# the app dir EXCEPT the data subtree). Paths are resolved with pure bash
# builtins (cd + `pwd -P`, resolving `..` and symlinks) — NOT `realpath`, which
# may be missing or lack `-m` on the host and would make the guard fail open.
__ha_resolve() {{
    local f="$1" dir bn rp
    case "$f" in /*) ;; *) f="$PWD/$f";; esac
    # `builtin cd` (never the guarded cd below) in a subshell for physical
    # resolution. If the whole path is a directory, resolve it directly — this
    # collapses a trailing `.`/`..` (e.g. `cd ..`); otherwise resolve the parent
    # and re-attach the basename (for files like `cat ../app.py`).
    rp=$(builtin cd -- "$f" 2>/dev/null && pwd -P) && {{ printf '%s' "$rp"; return; }}
    dir="${{f%/*}}"; bn="${{f##*/}}"; [ -z "$dir" ] && dir=/
    rp=$(builtin cd -- "$dir" 2>/dev/null && pwd -P) && {{ printf '%s/%s' "$rp" "$bn"; return; }}
    printf '%s' "$f"
}}
__ha_is_source() {{
    local rp; rp=$(__ha_resolve "$1")
    [ -z "$rp" ] && return 1
    case "$rp/" in
        "$__HA_DATA_DIR"/*) return 1;;          # data subtree is always allowed
        "$__HA_APP_DIR"/*)  return 0;;          # anything else in the app dir = source
    esac
    return 1
}}
__ha_guard_read() {{
    local tool="$1"; shift
    local a
    for a in "$@"; do
        case "$a" in -*) continue;; esac        # skip flags, only check path operands
        if __ha_is_source "$a"; then
            printf '\\033[31m[host-a100] Bị chặn: không được mở file mã nguồn của ứng dụng (%s).\\033[0m\\n' "$a" >&2
            return 1
        fi
    done
    command "$tool" "$@"
}}
{read_defs}

# 6) cd: refuse to enter the app's own source tree (anything under the app dir
# except the data subtree).
cd() {{
    local a d=""
    for a in "$@"; do case "$a" in -*) ;; *) d="$a";; esac; done
    if [ -n "$d" ] && __ha_is_source "$d"; then
        printf '\\033[31m[host-a100] Bị chặn: không được vào thư mục mã nguồn của ứng dụng (%s).\\033[0m\\n' "$d" >&2
        return 1
    fi
    builtin cd "$@"
}}

# 7) Tab-completion: hide the source tree from `cd` AND from every read/edit
# command, so the names are neither suggested nor reachable by completion.
# `$1` = "dir" to offer only directories (cd), anything else = files + dirs.
__ha_complete() {{
    COMPREPLY=()
    local mode="$1" cur="${{COMP_WORDS[COMP_CWORD]}}" dirpart base searchexp e name
    case "$cur" in
        */*) dirpart="${{cur%/*}}/"; base="${{cur##*/}}";;
        *)   dirpart="";            base="$cur";;
    esac
    eval "searchexp=${{dirpart:-./}}" 2>/dev/null || searchexp="${{dirpart:-./}}"
    shopt -s nullglob
    for e in "$searchexp"*; do
        name="${{e##*/}}"
        case "$name" in "$base"*) ;; *) continue;; esac
        __ha_is_source "$e" && continue
        if [ -d "$e" ]; then COMPREPLY+=( "${{dirpart}}${{name}}/" )
        elif [ "$mode" != "dir" ]; then COMPREPLY+=( "${{dirpart}}${{name}}" ); fi
    done
    shopt -u nullglob
}}
_ha_cd_complete()   {{ __ha_complete dir;  }}
_ha_read_complete() {{ __ha_complete file; }}
complete -F _ha_cd_complete   -o nospace cd
complete -F _ha_read_complete -o nospace -o filenames {read_tools}
"""


def _write_guard_rcfile():
    """Write the guard rcfile under DATA_DIR (shared FS, so it is also readable
    on a compute node via `srun --pty`). Returns the path, or None on any error
    so the caller can fall back to a plain login shell."""
    try:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix=".ha-termguard-", suffix=".bashrc",
                                    dir=config.DATA_DIR)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(_guard_rcfile_text())
        return path
    except Exception:  # noqa: BLE001 - guard is best-effort; never break the shell
        return None


def _build_argv(on_compute, use_gpu, gpu_model, cwd, rcfile=None):
    """The shell command — local interactive shell, or `srun --pty ... bash`.

    When a guard rcfile is available we launch `bash --rcfile <file> -i` (the
    rcfile re-creates the login environment); otherwise we fall back to the
    plain `bash -l` login shell.
    """
    if rcfile:
        shell = ["bash", "--rcfile", rcfile, "-i"]
    else:
        shell = ["bash", "-l"]
    if on_compute:
        prefix = job_service._srun_prefix("term", cwd, use_gpu=use_gpu,
                                          gpu_model=gpu_model)
        if prefix:  # SLURM active → run the shell on a compute node
            return prefix + ["--pty", *shell]
    return shell


def open_session(ws, on_compute=False, use_gpu=False, gpu_model=""):
    """Run one interactive shell for a connected WebSocket until it closes.

    Spawns `bash -l` in a fresh PTY (its own session + controlling tty so job
    control and full-screen apps work), streams PTY output to the browser as
    binary frames, and feeds back JSON control messages: {"type":"input",...}
    for keystrokes and {"type":"resize","cols":..,"rows":..} for window size.
    """
    if not available():
        try:
            ws.send(b"PTY khong kha dung tren node nay (can Linux).\r\n")
        finally:
            return

    cwd = config.DATA_DIR if os.path.isdir(config.DATA_DIR) else os.getcwd()
    rcfile = _write_guard_rcfile()
    argv = _build_argv(on_compute, use_gpu, gpu_model, cwd, rcfile=rcfile)

    env = dict(os.environ)
    env["TERM"] = "xterm-256color"
    env["PYTHONUNBUFFERED"] = "1"

    def _cleanup_rcfile():
        if rcfile:
            try:
                os.unlink(rcfile)
            except OSError:
                pass

    master, slave = pty.openpty()

    def _child_setup():
        # New session + make the slave our controlling terminal so signals
        # (Ctrl-C), job control and curses apps behave like a real tty.
        os.setsid()
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)

    try:
        proc = subprocess.Popen(
            argv, cwd=cwd, env=env,
            stdin=slave, stdout=slave, stderr=slave,
            preexec_fn=_child_setup, close_fds=True,
        )
    except Exception as exc:  # noqa: BLE001
        os.close(master)
        os.close(slave)
        _cleanup_rcfile()
        try:
            ws.send(f"Khong khoi dong duoc shell: {exc}\r\n".encode())
        finally:
            return
    os.close(slave)  # the child holds its own copy now

    stop = threading.Event()

    def reader():
        """PTY -> browser. Ends on EOF (shell exit) or a closed socket."""
        try:
            while not stop.is_set():
                r, _, _ = select.select([master], [], [], 0.2)
                if master in r:
                    data = os.read(master, 65536)
                    if not data:
                        break
                    ws.send(data)
        except Exception:  # noqa: BLE001 - socket/pty gone
            pass
        finally:
            stop.set()
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        while not stop.is_set():
            msg = ws.receive(timeout=0.5)
            if msg is None:
                if proc.poll() is not None:  # shell exited
                    break
                continue
            try:
                obj = json.loads(msg) if isinstance(msg, str) else None
            except ValueError:
                obj = None
            if not isinstance(obj, dict):
                continue
            kind = obj.get("type")
            if kind == "input":
                os.write(master, obj.get("data", "").encode("utf-8"))
            elif kind == "resize":
                _set_winsize(master, int(obj.get("rows", 24)),
                             int(obj.get("cols", 80)))
    except Exception:  # noqa: BLE001 - client vanished mid-stream
        pass
    finally:
        stop.set()
        # Kill the whole shell process group, then reap it.
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except (OSError, ProcessLookupError):
            pass
        try:
            proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            pass
        try:
            os.close(master)
        except OSError:
            pass
        t.join(timeout=1)
        _cleanup_rcfile()
