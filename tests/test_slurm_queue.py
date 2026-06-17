"""Tests for the dashboard SLURM queue view (utils.gpu.slurm_queue + helpers).
Pure stdlib — fakes `squeue` via monkeypatched shutil.which/subprocess.run, no
SLURM/HPC needed. Run:
    python3 tests/test_slurm_queue.py

Verifies: (1) one squeue call is parsed into our own jobs (detailed) + cluster
counts; (2) GPU counts parse BOTH untyped (`gres:gpu:1`) and typed
(`gres:gpu:ampere:2`) TRES fields; (3) Reason is glossed to Vietnamese, with the
quota reason flagged and unknown reasons passed through verbatim; (4) the pending
StartTime estimate is carried through (template labels it as a worst-case bound);
(5) graceful None when squeue is missing.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil  # noqa: E402

from utils import gpu  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def _fake_squeue(rows):
    """Install a fake `squeue` returning the given pipe-delimited rows."""
    out = "\n".join(rows)

    class R:
        stdout = out
        returncode = 0

    real_which = shutil.which
    shutil.which = lambda n: "/usr/bin/squeue" if n == "squeue" else real_which(n)
    subprocess.run = lambda *a, **k: R()


ROWS = [
    # id|name|user|state|reason|tres_per_node|time_used|time_limit|node|start
    "516495|hosta100-voice-19|s3002152|PENDING|Priority|gres:gpu:1|0:00|6:00:00|(null)|2026-06-20T12:03:49",
    "515814|hosta100-srv-20|s3002152|RUNNING|None|gres:gpu:1|20:03:48|1-00:00:00|ctit092|2026-06-16T13:00:00",
    "516600|hog|s3002152|PENDING|QOSMaxGRESPerUser|gres:gpu:1|0:00|2:00:00|(null)|N/A",
    "516316|train|hongm|PENDING|Resources|gres:gpu:2|0:00|1-16:00:00|(null)|N/A",
    "516443|x|wuh1|RUNNING|None|gres:gpu:ampere:1|39:27|2-00:00:00|ctit094|2026-06-16T00:00:00",
    "600001|cpujob|someone|PENDING|Priority|N/A|0:00|1:00:00|(null)|N/A",
]


def run():
    print("test_slurm_queue")

    _orig_run = subprocess.run
    _orig_which = shutil.which
    try:
        _fake_squeue(ROWS)
        q = gpu.slurm_queue("main-gpu", me="s3002152")

        check("returns dict", isinstance(q, dict))
        # cluster counts span ALL users; GPU counts include untyped + typed.
        # PENDING rows: 516495, 516600, 516316 (gpu:2), 600001 (cpu) = 4 total,
        # of which 3 request a GPU. RUNNING: 515814, 516443 = 2, both GPU.
        check("running_total", q["running_total"] == 2, str(q["running_total"]))
        check("pending_total", q["pending_total"] == 4, str(q["pending_total"]))
        check("running_gpu (untyped+typed)", q["running_gpu"] == 2, str(q["running_gpu"]))
        check("pending_gpu", q["pending_gpu"] == 3, str(q["pending_gpu"]))

        # only our jobs kept in detail, running sorted before pending
        mine = q["mine"]
        check("mine count", len(mine) == 3, str(len(mine)))
        check("running first", mine[0]["st"] == "R" and mine[0]["id"] == "515814")
        ids_pending = {j["id"] for j in mine if j["st"] == "PD"}
        check("pending ids", ids_pending == {"516495", "516600"}, str(ids_pending))

        byid = {j["id"]: j for j in mine}
        check("untyped gpu count", byid["516495"]["gpu"] == 1)
        check("priority gloss", byid["516495"]["reason_vi"].startswith("Chờ tới lượt"))
        check("quota gloss flagged", "TRẦN" in byid["516600"]["reason_vi"].upper())
        check("start estimate carried", byid["516495"]["start_est"] == "2026-06-20T12:03:49")
        check("N/A start blanked", byid["516600"]["start_est"] == "")
        check("(null) node blanked", byid["516495"]["node"] == "")
        check("running node kept", byid["515814"]["node"] == "ctit092")
    finally:
        subprocess.run = _orig_run
        shutil.which = _orig_which

    # helper-level checks
    check("tres untyped", gpu._gpu_count_tres("gres:gpu:1") == 1)
    check("tres typed", gpu._gpu_count_tres("gres:gpu:ampere:2") == 2)
    check("tres mixed", gpu._gpu_count_tres("gres:gpu:turing:2,gpu:ampere:1") == 3)
    check("tres none", gpu._gpu_count_tres("N/A") == 0)
    check("unknown reason verbatim", gpu._reason_vi("SomeNewReason") == "SomeNewReason")
    check("reqnode prefix", "down" in gpu._reason_vi("ReqNodeNotAvail, UnavailableNodes:ctit088").lower())

    # squeue missing → None (local dev box)
    _orig_which2 = shutil.which
    shutil.which = lambda n: None
    try:
        check("None when squeue missing", gpu.slurm_queue("main-gpu") is None)
    finally:
        shutil.which = _orig_which2

    print(f"\n{'ALL PASSED' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(run())
