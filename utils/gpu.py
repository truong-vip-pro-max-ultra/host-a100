"""
GPU inspection helpers built around nvidia-smi.

All calls use a fixed argument list (never shell=True) so there is no way for
user input to influence the command. If nvidia-smi is missing (e.g. local dev
on a non-GPU box) the helpers degrade gracefully.
"""
import os
import shutil
import subprocess

# Common absolute locations for nvidia-smi. On many HPC nodes the binary is
# installed but NOT on the app process's PATH (it only appears after loading a
# module), so PATH lookup alone reports "not found" even though a GPU is
# present. We probe these as a fallback.
_NVIDIA_SMI_PATHS = (
    "/usr/bin/nvidia-smi",
    "/bin/nvidia-smi",
    "/usr/local/bin/nvidia-smi",
    "/usr/local/nvidia/bin/nvidia-smi",
    "/usr/local/cuda/bin/nvidia-smi",
    "/opt/nvidia/bin/nvidia-smi",
)


def _find_nvidia_smi():
    """Return a usable nvidia-smi path (PATH first, then known locations)."""
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for path in _NVIDIA_SMI_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def nvidia_smi_available():
    return _find_nvidia_smi() is not None


def raw_nvidia_smi():
    """Return the plain textual nvidia-smi output, or an explanatory message."""
    smi = _find_nvidia_smi()
    if not smi:
        return "nvidia-smi not found on this host (no NVIDIA driver detected)."
    try:
        out = subprocess.run(
            [smi],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout or out.stderr or "(no output)"
    except Exception as exc:  # noqa: BLE001 - report any failure to the UI
        return f"Failed to run nvidia-smi: {exc}"


def gpu_summary():
    """
    Return a structured list of GPUs via nvidia-smi --query-gpu.

    Each item: name, mem_used_mb, mem_total_mb, util_pct, temp_c.
    Returns an empty list when no GPU / driver is available.
    """
    smi = _find_nvidia_smi()
    if not smi:
        return []
    query = "name,memory.used,memory.total,utilization.gpu,temperature.gpu"
    try:
        out = subprocess.run(
            [smi, f"--query-gpu={query}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:  # noqa: BLE001
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        name, used, total, util, temp = parts

        def _num(v):
            try:
                return float(v)
            except ValueError:
                return 0.0

        gpus.append({
            "name": name,
            "mem_used_mb": _num(used),
            "mem_total_mb": _num(total),
            "util_pct": _num(util),
            "temp_c": _num(temp),
        })
    return gpus
