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
import struct
import subprocess
import sys
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


def _build_argv(on_compute, use_gpu, gpu_model, cwd):
    """The shell command — local login shell, or `srun --pty ... bash -l`."""
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
    argv = _build_argv(on_compute, use_gpu, gpu_model, cwd)

    env = dict(os.environ)
    env["TERM"] = "xterm-256color"
    env["PYTHONUNBUFFERED"] = "1"

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
