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
