"""
CPU and RAM inspection for the dashboard.

Prefers psutil when installed; otherwise falls back to reading /proc on Linux
(the deployment target). On platforms where neither works (e.g. local Windows
dev without psutil) the helpers degrade gracefully, returning name=None and
pct=0 so the UI can show "không khả dụng" instead of crashing.
"""
import os
import platform
import time

try:
    import psutil  # type: ignore
except Exception:  # noqa: BLE001 - psutil is optional
    psutil = None


def _cpu_name():
    """Best-effort human-readable CPU model name."""
    # Linux: /proc/cpuinfo "model name".
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    # Fallbacks for non-Linux.
    return platform.processor() or platform.machine() or None


def _cpu_count():
    try:
        return os.cpu_count() or 0
    except Exception:  # noqa: BLE001
        return 0


def _read_proc_stat():
    """Return (idle, total) jiffies from the aggregate /proc/stat cpu line."""
    with open("/proc/stat", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("cpu "):
                parts = [float(x) for x in line.split()[1:]]
                idle = parts[3] + (parts[4] if len(parts) > 4 else 0.0)
                return idle, sum(parts)
    return 0.0, 0.0


def cpu_info():
    """Return {name, cores, pct} for the host CPU."""
    name, cores = _cpu_name(), _cpu_count()
    pct = 0.0
    if psutil is not None:
        try:
            pct = float(psutil.cpu_percent(interval=0.3))
        except Exception:  # noqa: BLE001
            pct = 0.0
    else:
        # Sample /proc/stat over a short window.
        try:
            idle1, total1 = _read_proc_stat()
            time.sleep(0.3)
            idle2, total2 = _read_proc_stat()
            dt = total2 - total1
            if dt > 0:
                pct = max(0.0, min(100.0, (1 - (idle2 - idle1) / dt) * 100))
        except OSError:
            pct = 0.0
    return {"name": name, "cores": cores, "pct": round(pct, 1)}


def ram_info():
    """Return {total, used, pct} in bytes for system RAM."""
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            return {"total": vm.total, "used": vm.used,
                    "pct": round(float(vm.percent), 1)}
        except Exception:  # noqa: BLE001
            pass
    # Linux fallback: /proc/meminfo (values are in kB).
    try:
        info = {}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                info[key.strip()] = int(rest.split()[0]) * 1024
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable",
                         info.get("MemFree", 0) + info.get("Cached", 0))
        used = max(0, total - avail)
        pct = round(used * 100 / total, 1) if total else 0.0
        return {"total": total, "used": used, "pct": pct}
    except OSError:
        return {"total": 0, "used": 0, "pct": 0.0}


# --- running processes (for the dashboard CPU/RAM drill-down) --------------- #
# Per-process %CPU needs two samples. Both backends keep state BETWEEN calls
# (psutil caches its Process objects; the /proc path keeps _PROC_CPU below), so
# the value is the average since the previous dashboard poll — no sleep needed.
# The first call after start reports 0 for every process (no baseline yet).
_HZ = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_PROC_CPU = {}   # pid -> (proc_jiffies, wall_time) from the previous /proc sample


def _psutil_processes():
    out = []
    for p in psutil.process_iter(["pid", "name", "memory_info", "memory_percent"]):
        try:
            cpu = p.cpu_percent(None)          # since the previous call (cached)
            info = p.info
            mi = info.get("memory_info")
            out.append({
                "pid": info["pid"],
                "name": (info.get("name") or "?")[:40],
                "cpu": round(cpu, 1),
                "mem_pct": round(info.get("memory_percent") or 0.0, 1),
                "mem_bytes": mi.rss if mi else 0,
            })
        except Exception:  # noqa: BLE001 - process vanished / access denied
            continue
    return out


def _proc_processes():
    """psutil-free fallback: read /proc directly (Linux only)."""
    out, seen, now = [], {}, time.time()
    total_mem = 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    total_mem = int(line.split()[1]) * 1024
                    break
    except OSError:
        return out
    page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    try:
        pids = [d for d in os.listdir("/proc") if d.isdigit()]
    except OSError:
        return out
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as fh:
                stat = fh.read()
            rp = stat.rfind(")")
            name = stat[stat.find("(") + 1:rp]
            f = stat[rp + 2:].split()
            tj = int(f[11]) + int(f[12])               # utime + stime
            with open(f"/proc/{pid}/statm", encoding="utf-8") as fh:
                rss = int(fh.read().split()[1]) * page
            cpu, prev = 0.0, _PROC_CPU.get(pid)
            if prev:
                dt = now - prev[1]
                if dt > 0:
                    cpu = 100.0 * (tj - prev[0]) / _HZ / dt
            seen[pid] = (tj, now)
            out.append({
                "pid": int(pid), "name": name[:40],
                "cpu": round(max(0.0, cpu), 1),
                "mem_pct": round(rss * 100 / total_mem, 1) if total_mem else 0.0,
                "mem_bytes": rss,
            })
        except (OSError, ValueError, IndexError):
            continue
    _PROC_CPU.clear()
    _PROC_CPU.update(seen)
    return out


def top_processes(limit=8):
    """Return {'cpu': [...], 'mem': [...]} — the top processes by %CPU and by
    memory. Each item: {pid, name, cpu, mem_pct, mem_bytes}. Empty lists when
    neither psutil nor /proc is available (e.g. Windows dev without psutil)."""
    try:
        procs = _psutil_processes() if psutil is not None else _proc_processes()
    except Exception:  # noqa: BLE001
        procs = []
    by_cpu = sorted(procs, key=lambda x: x["cpu"], reverse=True)[:limit]
    by_mem = sorted(procs, key=lambda x: x["mem_bytes"], reverse=True)[:limit]
    return {"cpu": by_cpu, "mem": by_mem}
