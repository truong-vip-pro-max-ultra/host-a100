"""
GPU inspection helpers built around nvidia-smi.

All calls use a fixed argument list (never shell=True) so there is no way for
user input to influence the command. If nvidia-smi is missing (e.g. local dev
on a non-GPU box) the helpers degrade gracefully.
"""
import glob
import os
import re
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


def _read_text(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return None


def gpu_diagnostics():
    """Explain WHY no GPU is visible — most often a login node with no GPU.

    Checks the device nodes, kernel driver, PCI bus, and whether SLURM is
    present (which usually means the GPUs live on separate compute nodes you
    must request an allocation on). All checks are read-only and degrade
    quietly on machines where the files/tools are absent (e.g. Windows dev).
    """
    lines = []

    devs = glob.glob("/dev/nvidia*")
    lines.append(f"Thiết bị /dev/nvidia*: {', '.join(devs) if devs else 'không có'}")

    drv = _read_text("/proc/driver/nvidia/version")
    lines.append(f"Driver NVIDIA: {drv.splitlines()[0] if drv else 'chưa nạp'}")

    lspci = shutil.which("lspci")
    if lspci:
        try:
            out = subprocess.run([lspci], capture_output=True, text=True,
                                 timeout=10).stdout
            nv = [l for l in out.splitlines() if "nvidia" in l.lower()]
            lines.append("Thiết bị PCI NVIDIA: "
                         + (nv[0].strip() if nv else "không thấy trên node này"))
        except Exception:  # noqa: BLE001
            pass

    if shutil.which("srun") or shutil.which("sinfo"):
        lines.append(
            "Phát hiện SLURM → đây nhiều khả năng là NODE ĐĂNG NHẬP (không gắn "
            "GPU). GPU A100 nằm ở compute node. Xin một GPU rồi chạy ở đó, ví dụ:\n"
            "    srun --gres=gpu:1 --pty bash   # vào node có GPU\n"
            "    # sau đó chạy: python3 app.py\n"
            "Hoặc xem các node có GPU: sinfo -o '%n %G'")

    return "\n".join(lines)


def slurm_gpu_raw(arch="ampere"):
    """A read-only snapshot of GPU availability on the SLURM cluster.

    The dashboard runs on the login node, which has no GPU, so nvidia-smi there
    is misleading. When jobs are dispatched via SLURM the relevant question is
    "how many <arch> GPUs are free on the cluster" — sinfo answers that without
    consuming an allocation. Returns text (filtered to the arch) or None.
    """
    sinfo = shutil.which("sinfo")
    if not sinfo:
        return None
    try:
        out = subprocess.run(
            [sinfo, "-h", "-N",
             "-O", "NodeHost:14,Gres:22,GresUsed:24,StateLong:12"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:  # noqa: BLE001
        return None
    rows = [l.rstrip() for l in out.stdout.splitlines()
            if arch in l.lower() and "gpu" in l.lower()]
    if not rows:
        return None
    header = f"{'NODE':<14}{'GRES (tổng)':<22}{'GRES đã dùng':<24}TRẠNG THÁI"
    return header + "\n" + "\n".join(rows)


def _count_gpus(field, arch):
    """Sum the GPU counts of `arch` in a SLURM GRES / GresUsed field.

    Handles `gpu:ampere:4`, `gpu:ampere:2(IDX:0-1)` and comma-separated lists.
    """
    return sum(int(n) for n in re.findall(rf"gpu:{re.escape(arch)}:(\d+)", field))


def slurm_gpu_counts(arch="ampere"):
    """Aggregate how many `arch` GPUs the cluster has, is using, and has free.

    Parses `sinfo` (read-only, no allocation) and returns
    {total, used, free, nodes} or None when sinfo is unavailable / reports no
    GPUs of this arch. `free` only counts GPUs on nodes that can actually accept
    work — nodes in a down/drain/reserved/etc. state are excluded from free even
    if their GPUs look idle, because you can't be scheduled onto them.

    StateLong is queried FIRST and GresUsed LAST so a long IDX list in GresUsed
    can't bleed into and corrupt the columns we slice for state/total.
    """
    sinfo = shutil.which("sinfo")
    if not sinfo:
        return None
    try:
        out = subprocess.run(
            [sinfo, "-h", "-N", "-O", "StateLong:20,Gres:80,GresUsed:200"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:  # noqa: BLE001
        return None

    _UNUSABLE = ("down", "drain", "drng", "resv", "maint", "inval", "unk",
                 "fail", "boot", "power", "plnd")
    total = used = free = nodes = 0
    for line in out.stdout.splitlines():
        if "gpu" not in line.lower() or arch not in line.lower():
            continue
        state = line[0:20].strip().lower()
        gres = line[20:100]
        gres_used = line[100:]
        node_total = _count_gpus(gres, arch)
        if node_total == 0:
            continue
        node_used = _count_gpus(gres_used, arch)
        total += node_total
        used += node_used
        nodes += 1
        if not any(bad in state for bad in _UNUSABLE):
            free += max(0, node_total - node_used)
    if total == 0:
        return None
    return {"total": total, "used": used, "free": free, "nodes": nodes}


def raw_nvidia_smi():
    """Return the plain textual nvidia-smi output, or an explanatory message."""
    smi = _find_nvidia_smi()
    if not smi:
        return ("Không tìm thấy nvidia-smi trên node này.\n\n"
                "Chẩn đoán:\n" + gpu_diagnostics())
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
