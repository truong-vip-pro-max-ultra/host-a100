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


def _gpu_count_any(field):
    """Sum the GPU counts of ANY type in a SLURM GRES / GresUsed field.

    Handles `gpu:ampere:4`, `gpu:ampere:2(IDX:0-1)` and comma-separated mixed
    lists like `gpu:turing:2,gpu:ampere:1`.
    """
    return sum(int(n) for n in re.findall(r"gpu:[^:]+:(\d+)", field))


# SLURM node "states" that mean the node can't accept new work right now, so its
# GPUs must NOT be counted as free even if they look idle.
_UNUSABLE = ("down", "drain", "drng", "resv", "maint", "inval", "unk",
             "fail", "boot", "power", "plnd")

# Node FEATURE tokens that are NOT a GPU model (CPU vendor / instruction sets /
# GPU vendor line). Whatever feature remains after removing these is the GPU
# model (a100, a40, l40, l40s, t4, h200nvl, rtx-6000, rtx6000pro, ...).
_NON_MODEL_FEATS = frozenset({
    "intel", "amd", "avx", "avx2", "avx512", "avx512f",
    "tesla", "quadro", "geforce", "nvidia",
})


def _model_from_features(feats):
    """Pick the GPU-model token out of a comma-separated AVAIL_FEATURES string."""
    for tok in feats.split(","):
        tok = tok.strip().lower()
        if tok and tok not in _NON_MODEL_FEATS:
            return tok
    return None


def slurm_gpu_models():
    """Per-model GPU availability across the cluster — the honest answer to
    "how many <model> GPUs are free".

    CRUCIAL: on this cluster the SLURM GRES *type* is the architecture FAMILY
    (ampere, turing, lovelace, hopper, blackwell), which is NOT the GPU model —
    e.g. `gpu:ampere` spans both A100 and A40. The real model lives in the node
    FEATURE list, so we key the counts off that. `sinfo -N` lists a node once per
    partition, so we dedup by node host. `free` excludes down/drain/reserved
    nodes (you can't be scheduled onto them).

    Parses `sinfo` (read-only, no allocation). Returns a list of
    {model, arch, total, used, free, down, nodes} sorted by free-desc then model,
    or None when sinfo is unavailable / no GPU nodes exist. `total = used + free
    + down`; `down` = GPUs on unschedulable (down/drain/...) nodes. Generous
    column widths keep any single field from overflowing into the next.
    """
    sinfo = shutil.which("sinfo")
    if not sinfo:
        return None
    try:
        out = subprocess.run(
            [sinfo, "-h", "-N", "-O",
             "NodeHost:20,StateLong:14,Gres:40,GresUsed:150,Features:160"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:  # noqa: BLE001
        return None

    seen = set()
    models = {}  # model -> {arch, total, used, free, down, nodes}
    for line in out.stdout.splitlines():
        node = line[0:20].strip()
        if not node or node in seen:
            continue
        gres = line[34:74]
        node_total = _gpu_count_any(gres)
        if node_total == 0:
            continue
        seen.add(node)
        state = line[20:34].strip().lower()
        gres_used = line[74:224]
        feats = line[224:]
        model = _model_from_features(feats)
        if not model:
            continue
        arch_m = re.search(r"gpu:([^:]+):", gres)
        node_used = _gpu_count_any(gres_used)
        m = models.setdefault(model, {"arch": arch_m.group(1) if arch_m else "",
                                      "total": 0, "used": 0, "free": 0,
                                      "down": 0, "nodes": 0})
        m["total"] += node_total
        m["nodes"] += 1
        if any(bad in state for bad in _UNUSABLE):
            # Node can't be scheduled onto — its GPUs are neither free nor really
            # "in use" (GresUsed is unreliable when down), so bucket them as down.
            m["down"] += node_total
        else:
            m["used"] += node_used
            m["free"] += max(0, node_total - node_used)
    if not models:
        return None
    return [dict(model=k, **v) for k, v in
            sorted(models.items(), key=lambda kv: (-kv[1]["free"], kv[0]))]


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
