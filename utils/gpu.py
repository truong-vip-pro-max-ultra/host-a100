"""
GPU inspection helpers built around nvidia-smi.

All calls use a fixed argument list (never shell=True) so there is no way for
user input to influence the command. If nvidia-smi is missing (e.g. local dev
on a non-GPU box) the helpers degrade gracefully.
"""
import getpass
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


def slurm_gpu_models(partition=None):
    """Per-model GPU availability — the honest answer to "how many <model> GPUs
    can I actually get".

    CRUCIAL: on this cluster the SLURM GRES *type* is the architecture FAMILY
    (ampere, turing, lovelace, hopper, blackwell), which is NOT the GPU model —
    e.g. `gpu:ampere` spans both A100 and A40. The real model lives in the node
    FEATURE list, so we key the counts off that. `sinfo -N` lists a node once per
    partition, so we dedup by node host. `free` excludes down/drain/reserved
    nodes (you can't be scheduled onto them).

    `partition` scopes the query (sinfo -p) so counts reflect only nodes you can
    submit to — the idle GPUs in admin/research-group partitions are unreachable,
    so counting them cluster-wide is misleading. None = whole cluster.

    Parses `sinfo` (read-only, no allocation). Returns a list of
    {model, arch, total, used, free, down, nodes} sorted by free-desc then model,
    or None when sinfo is unavailable / no GPU nodes exist. `total = used + free
    + down`; `down` = GPUs on unschedulable (down/drain/...) nodes. Generous
    column widths keep any single field from overflowing into the next.
    """
    sinfo = shutil.which("sinfo")
    if not sinfo:
        return None
    cmd = [sinfo, "-h", "-N", "-O",
           "NodeHost:20,StateLong:14,Gres:40,GresUsed:150,Features:160"]
    if partition:
        cmd += ["-p", partition]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
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


# squeue Reason → a short Vietnamese gloss for the dashboard. Unknown reasons are
# shown verbatim (accuracy over guessing). The long "reserved for higher priority
# partitions" message and the "ReqNodeNotAvail,..." variants are matched by prefix.
_REASON_VI = {
    "Priority": "Chờ tới lượt (xếp hàng — cụm đông)",
    "Resources": "Chờ GPU/tài nguyên trống",
    "QOSMaxGRESPerUser": "CHẠM TRẦN GPU của bạn (quota 2 GPU)",
    "QOSMaxJobsPerUserLimit": "Chạm trần số job của bạn",
    "QOSGrpGRES": "Chạm trần GPU của nhóm",
    "AssocGrpGRES": "Chạm trần GPU của nhóm",
    "Dependency": "Chờ job phụ thuộc chạy xong",
    "JobArrayTaskLimit": "Giới hạn job array",
    "BeginTime": "Chờ tới giờ hẹn chạy",
    "None": "",
    "": "",
}


def _gpu_count_tres(field):
    """Sum GPU counts in a squeue TRES-per-node field (`%b`), which — unlike the
    sinfo Gres field `_gpu_count_any` handles — may be UNTYPED (`gres:gpu:1`) as
    well as typed (`gres:gpu:ampere:2`), comma-separated."""
    return sum(int(n) for n in re.findall(r"gpu:(?:[^:,]+:)?(\d+)", field or ""))


def _reason_vi(reason):
    """Translate a squeue Reason to Vietnamese; fall back to the raw string."""
    r = (reason or "").strip()
    if r in _REASON_VI:
        return _REASON_VI[r]
    if r.startswith("ReqNodeNotAvail"):
        return "Node yêu cầu chưa sẵn sàng (down/bận)"
    if "reserved for jobs in higher priority" in r or "DOWN, DRAINED" in r:
        return "Node cần dùng đang down/được giữ cho job khác"
    return r


def slurm_queue(partition=None, me=None):
    """Detailed SLURM queue for the dashboard's "Hàng đợi" card.

    ONE read-only `squeue` call (fixed argv, never shell=True) over `partition`.
    We split out the current user's jobs (detailed — for monitoring our GPU
    servers) and summarise the rest as congestion context. Returns None when
    `squeue` is unavailable (e.g. local dev box).

    Each job dict: id, name, user, state ('RUNNING'/'PENDING'/…), st ('R'/'PD'/…),
    reason (raw), reason_vi (gloss), gpu (int), is_gpu, time_used, time_limit,
    node, start_est (estimated start for PD; '' when N/A), mine (bool).

    ACCURACY NOTE (matches what bit us in practice): `start_est` is SLURM's
    PESSIMISTIC worst-case — it assumes every running job uses its FULL walltime,
    so it routinely reads days out while the job actually starts far sooner. The
    template labels it as a "muộn nhất" upper bound; never present it as a real ETA.
    """
    squeue = shutil.which("squeue")
    if not squeue:
        return None
    if me is None:
        try:
            me = getpass.getuser()
        except Exception:  # noqa: BLE001
            me = os.environ.get("USER", "")
    # %S = expected start time (estimate for PENDING jobs), %b = TRES_PER_NODE.
    fmt = "%i|%j|%u|%T|%r|%b|%M|%l|%N|%S"
    cmd = [squeue, "-h", "-o", fmt]
    if partition:
        cmd += ["-p", partition]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except Exception:  # noqa: BLE001
        return None

    mine = []
    pending = running = pending_gpu = running_gpu = 0
    for line in out.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 10:
            continue
        jid, name, user, state, reason, gres, tused, tlimit, node, start = \
            (p.strip() for p in parts[:10])
        if not jid:
            continue
        gpu_n = _gpu_count_tres(gres)
        is_gpu = gpu_n > 0
        if state == "PENDING":
            pending += 1
            pending_gpu += is_gpu
        elif state == "RUNNING":
            running += 1
            running_gpu += is_gpu
        if user != me:
            continue  # only keep our own jobs in detail; others feed the counts
        mine.append(dict(
            id=jid, name=name, user=user, state=state,
            st={"RUNNING": "R", "PENDING": "PD"}.get(state, (state[:2] or "?")),
            reason=reason, reason_vi=_reason_vi(reason),
            gpu=gpu_n, is_gpu=is_gpu,
            time_used=tused, time_limit=tlimit,
            node=node if node and node != "(null)" else "",
            start_est="" if start in ("", "N/A", "(null)") else start,
            mine=True,
        ))

    # Running first, then pending, then by job id — the order you want to scan.
    mine.sort(key=lambda j: (j["state"] != "RUNNING", j["id"]))
    return dict(
        me=me, mine=mine,
        pending_total=pending, running_total=running,
        pending_gpu=pending_gpu, running_gpu=running_gpu,
        partition=partition or "",
    )


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
