"""
Inference / run job management.

Two run modes:

  * 'runner'  — runs the trusted, platform-shipped inference_runner.py inside a
                selected venv. User supplies only model + env + JSON params.
  * 'project' — runs the MAIN FILE of a user project (their own code) inside a
                selected venv. The project is copied into the job directory so a
                run is reproducible and never mutates the source project.

In both modes everything is confined under /data. No subprocess uses shell=True
and the user never supplies a raw command line — only choices (which file, which
env, which model) and a JSON params blob.

Outputs the user code writes to $OUTPUT_DIR (/data/jobs/<id>/output) are
browsable and downloadable from the Jobs page.
"""
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import zipfile

# Resolve srun/sbatch once so we can decide whether SLURM dispatch is possible.
# srun = interactive (used by the Terminal/PTY); sbatch = detached batch jobs.
_SRUN = shutil.which("srun")
_SBATCH = shutil.which("sbatch")

# How often we poll squeue for a running job's state, and the file in which we
# stash its SLURM job id (so delete_job can scancel a still-running job).
_POLL_SECS = 4
_SLURM_ID_FILE = "slurm_job_id"

import config
from services import env_service, model_service, project_service
from services import storage_service as db
from utils import file_utils, gpu, progress

# Absolute path to the runner that ships with the platform.
_RUNNER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "inference_runner.py")


def list_jobs():
    rows = db.execute(
        """
        SELECT jobs.*, models.name AS model_name, envs.name AS env_name,
               projects.name AS project_name
        FROM jobs
        LEFT JOIN models   ON jobs.model_id   = models.id
        LEFT JOIN envs     ON jobs.env_id     = envs.id
        LEFT JOIN projects ON jobs.project_id = projects.id
        ORDER BY jobs.created_at DESC
        """,
        fetch="all",
    ) or []
    return [dict(r) for r in rows]


def get_job(job_id):
    row = db.execute(
        """
        SELECT jobs.*, models.name AS model_name, models.path AS model_path,
               envs.name AS env_name, envs.path AS env_path,
               projects.name AS project_name, projects.path AS project_path
        FROM jobs
        LEFT JOIN models   ON jobs.model_id   = models.id
        LEFT JOIN envs     ON jobs.env_id     = envs.id
        LEFT JOIN projects ON jobs.project_id = projects.id
        WHERE jobs.id = ?
        """,
        (job_id,), fetch="one",
    )
    return dict(row) if row else None


def read_log(job_id):
    job = get_job(job_id)
    if not job or not job.get("logs_path"):
        return ""
    log_file = os.path.join(job["logs_path"], "job.log")
    if not os.path.exists(log_file):
        return ""
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()[-20000:]
    except OSError:
        return ""


def submit_job(model_id, env_id, job_name, params_json,
               run_mode="runner", project_id=None, main_file=None,
               use_gpu=True, gpu_model="", run_local=False):
    """
    Validate inputs, create the DB row + job dirs, and launch the run thread.

    Returns the new job_id. Raises ValueError on bad input.
    """
    env = env_service.get_env(env_id)
    if not env:
        raise ValueError("Môi trường đã chọn không tồn tại.")
    py = env_service._python_path(env["path"])
    if not os.path.exists(py):
        raise ValueError("Thiếu trình thông dịch của môi trường trên đĩa.")

    # Model is required for the runner mode, optional for project mode.
    model = None
    if model_id:
        model = model_service.get_model(model_id)
        if not model:
            raise ValueError("Mô hình đã chọn không tồn tại.")
    if run_mode == "runner" and not model:
        raise ValueError("Runner mặc định cần một mô hình.")

    project = None
    if run_mode == "project":
        if not project_id:
            raise ValueError("Hãy chọn một dự án để chạy.")
        project = project_service.get_project(project_id)
        if not project:
            raise ValueError("Dự án đã chọn không tồn tại.")
        if not main_file:
            main_file = project.get("main_file")
        if not main_file:
            raise ValueError("Hãy chọn file chính để chạy.")
        # Confirm the main file exists inside the project.
        abspath, main_file = file_utils.safe_relpath(project["path"], main_file)
        if not os.path.isfile(abspath):
            raise ValueError("File chính không tồn tại trong dự án.")

    job_name = (job_name or run_mode).strip()[:128]

    params = {}
    if params_json and params_json.strip():
        try:
            params = json.loads(params_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Tham số phải là JSON hợp lệ: {exc}")
        if not isinstance(params, dict):
            raise ValueError("Tham số JSON phải là một object.")

    created = db.now()
    job_id = db.execute(
        "INSERT INTO jobs (model_id, env_id, project_id, main_file, run_mode, "
        "name, status, progress, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 'queued', 0, ?)",
        (model_id or None, env_id, project_id or None, main_file, run_mode,
         job_name, created),
        commit=True,
    )

    job_dir = file_utils.safe_join(config.JOBS_DIR, str(job_id))
    output_dir = file_utils.safe_join(job_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    result_path = file_utils.safe_join(config.RESULTS_DIR, f"{job_id}.json")
    db.execute(
        "UPDATE jobs SET logs_path=?, result_path=?, output_path=? WHERE id=?",
        (job_dir, result_path, output_dir, job_id), commit=True,
    )

    params_file = os.path.join(job_dir, "params.json")
    with open(params_file, "w") as fh:
        json.dump(params, fh)

    progress.init("job", job_id, step="queued")

    if run_mode == "project":
        # Copy the project into the job dir for a reproducible, isolated run.
        code_dir = file_utils.safe_join(job_dir, "code")
        shutil.copytree(project["path"], code_dir, dirs_exist_ok=True)
        # Snapshot the files we just copied in. Anything the user code writes
        # into its working directory (relative-path outputs like love.pdf) is
        # then detectable as "new" and surfaced as a downloadable result.
        _snapshot_code_dir(job_dir, code_dir)
        cmd = [py, main_file]
        cwd = code_dir
    else:
        cmd = [py, _RUNNER,
               "--model-path", model["path"],
               "--params", params_file,
               "--output", result_path]
        cwd = job_dir

    # Cách B (sbatch detached): when SLURM is active the job is submitted to a
    # compute node via `sbatch`, which writes its output to a file on the shared
    # filesystem. There is NO live srun back-channel to time out ("Socket timed
    # out on send/recv"), and the job survives an app restart. _launch picks the
    # sbatch path when SLURM+sbatch are available, else runs the command locally.
    _launch(job_id, cmd, cwd, job_dir, output_dir, params_file,
            model_path=model["path"] if model else "",
            use_gpu=use_gpu, gpu_model=gpu_model, run_local=run_local)
    return job_id


def slurm_active():
    """True when jobs will actually be dispatched to a GPU node via srun."""
    return bool(config.USE_SLURM and _SRUN)


def _gres_count():
    """The GPU count from config.SLURM_GRES (the trailing `:N`), default 1."""
    m = re.search(r":(\d+)\s*$", config.SLURM_GRES or "")
    return m.group(1) if m else "1"


def _gpu_flags(gpu_model=""):
    """The srun GPU request flags.

    gpu_model="" → ANY GPU type (untyped `--gres=gpu:N`). This is the "cấp nhanh
    nhất" choice, so it must NOT pin to the arch family — `gpu:ampere:N` would
    still queue behind the busy A100/A40 nodes even when idle L40/H200/blackwell
    nodes are sitting free. Untyped lets SLURM grab whatever is free first.
    gpu_model set (e.g. "a100") → pin to that MODEL via --constraint, because the
    GRES type is only the arch family (ampere = A100 AND A40), so the constraint
    is what actually selects A100 vs A40. We resolve the arch from sinfo to keep
    a typed --gres (known-good on this cluster); fall back to an untyped count.
    """
    if not gpu_model:
        return [f"--gres=gpu:{_gres_count()}"]
    arch = ""
    models = gpu.slurm_gpu_models() or []
    for m in models:
        if m["model"] == gpu_model:
            arch = m.get("arch", "")
            break
    gres = f"gpu:{arch}:{_gres_count()}" if arch else f"gpu:{_gres_count()}"
    return [f"--gres={gres}", f"--constraint={gpu_model}"]


def _srun_prefix(job_id, cwd, use_gpu=True, gpu_model=""):
    """Build the `srun ...` prefix that sends a job to a compute node.

    Returns [] when SLURM dispatch is off or srun is unavailable, in which case
    the job runs locally (the previous behaviour). srun streams the remote
    stdout/stderr back to us, so log capture works exactly as before; --chdir
    runs the command in the job's working dir on the (shared-filesystem) node,
    and --export=ALL forwards JOB_DIR/OUTPUT_DIR/MODEL_PATH/etc. to the job.

    use_gpu=False omits --gres so a CPU-only task doesn't tie up a GPU and is
    scheduled on any free node (usually much sooner than a GPU one). gpu_model
    optionally pins the run to a specific GPU model (e.g. "a100") — see
    _gpu_flags.
    """
    if not slurm_active():
        return []
    pre = [_SRUN,
           "--job-name", f"hosta100-{job_id}",
           "--chdir", cwd,
           "--export=ALL",
           "--unbuffered"]
    if use_gpu:
        for i, flag in enumerate(_gpu_flags(gpu_model), start=1):
            pre.insert(i, flag)
    if config.SLURM_PARTITION:
        pre += ["--partition", config.SLURM_PARTITION]
    if config.SLURM_ACCOUNT:
        pre += ["--account", config.SLURM_ACCOUNT]
    if config.SLURM_TIME:
        pre += ["--time", config.SLURM_TIME]
    if config.SLURM_CPUS:
        pre += ["--cpus-per-task", config.SLURM_CPUS]
    if config.SLURM_MEM:
        pre += ["--mem", config.SLURM_MEM]
    return pre


def _sbatch_flags(job_id, cwd, slurm_out, use_gpu=True, gpu_model=""):
    """The `sbatch` flags for a detached job (mirrors _srun_prefix's options).

    --output sends the job's combined stdout/stderr to a file on the shared FS,
    so we tail that file instead of streaming over a live connection. --export=ALL
    forwards JOB_DIR/OUTPUT_DIR/MODEL_PATH/PYTHONIOENCODING/etc. to the job.
    """
    flags = ["--job-name", f"hosta100-{job_id}",
             "--chdir", cwd,
             "--export=ALL",
             "--output", slurm_out]
    if use_gpu:
        flags += _gpu_flags(gpu_model)
    if config.SLURM_PARTITION:
        flags += ["--partition", config.SLURM_PARTITION]
    if config.SLURM_ACCOUNT:
        flags += ["--account", config.SLURM_ACCOUNT]
    if config.SLURM_TIME:
        flags += ["--time", config.SLURM_TIME]
    if config.SLURM_CPUS:
        flags += ["--cpus-per-task", config.SLURM_CPUS]
    if config.SLURM_MEM:
        flags += ["--mem", config.SLURM_MEM]
    return flags


def _set_status(job_id, status=None, prog=None):
    sets, vals = [], []
    if status is not None:
        sets.append("status=?")
        vals.append(status)
    if prog is not None:
        sets.append("progress=?")
        vals.append(int(prog))
    if sets:
        vals.append(job_id)
        db.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=?",
                   tuple(vals), commit=True)


def _launch(job_id, cmd, cwd, job_dir, output_dir, params_file, model_path="",
            use_gpu=True, gpu_model="", run_local=False):
    log_file = os.path.join(job_dir, "job.log")

    # Environment for the child process. User code reads these to find the model,
    # write outputs, and read params — without any shell exposure.
    env_vars = dict(os.environ)
    env_vars.update({
        "PYTHONUNBUFFERED": "1",
        # Compute nodes often have a latin-1 locale, so the job's stdout defaults
        # to latin-1 and any non-ASCII print() (e.g. a Vietnamese transcript)
        # raises UnicodeEncodeError. Force UTF-8 stdio; we read the stream/file
        # as UTF-8 below, so this round-trips cleanly.
        "PYTHONIOENCODING": "utf-8",
        "JOB_ID": str(job_id),
        "JOB_DIR": job_dir,
        "OUTPUT_DIR": output_dir,
        "PARAMS_FILE": params_file,
    })
    if model_path:
        env_vars["MODEL_PATH"] = model_path

    # run_local=True forces the job to run right here on the login node (a local
    # subprocess) instead of being dispatched to a compute node via SLURM. The
    # login node is the ONLY node with internet, so this is the way to run code
    # that fetches at runtime (requests/HuggingFace download/etc.). There is no
    # GPU on the login node, so use_gpu/gpu_model are irrelevant in this mode.
    if slurm_active() and _SBATCH and not run_local:
        target = lambda: _run_via_sbatch(job_id, cmd, cwd, job_dir, log_file,
                                         env_vars, use_gpu, gpu_model)
    else:
        target = lambda: _run_local(job_id, cmd, cwd, log_file, env_vars)
    threading.Thread(target=target, daemon=True).start()


def _finish(job_id, code):
    """Mark a job done (code 0) or error, with a closing log line."""
    if code == 0:
        progress.update("job", job_id, progress=100, status="done",
                        step="completed",
                        append_log="\nJob finished successfully.\n")
        _set_status(job_id, status="done", prog=100)
    else:
        progress.update("job", job_id, status="error", step="process failed",
                        append_log=f"\nProcess exited with code {code}.\n")
        _set_status(job_id, status="error")


def _scan_progress(job_id, line):
    """Honour a `PROGRESS <n>` line the job prints to drive the progress bar."""
    s = line.strip()
    if s.startswith("PROGRESS"):
        try:
            val = int(s.split()[1])
            progress.update("job", job_id, progress=val)
            _set_status(job_id, prog=val)
        except (IndexError, ValueError):
            pass


def _run_local(job_id, cmd, cwd, log_file, env_vars):
    """Run the job as a local subprocess (no SLURM — e.g. Windows dev)."""
    progress.update("job", job_id, status="running", progress=5, step="launching")
    _set_status(job_id, status="running", prog=5)
    try:
        with open(log_file, "w", buffering=1,
                  encoding="utf-8", errors="replace") as logf:
            logf.write("$ " + " ".join(cmd) + "\n")
            logf.write(f"(cwd={cwd})\n\n")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace",
                env=env_vars, cwd=cwd,
            )
            for line in proc.stdout:
                logf.write(line)
                progress.update("job", job_id, append_log=line)
                _scan_progress(job_id, line)
            code = proc.wait()
        _finish(job_id, code)
    except Exception as exc:  # noqa: BLE001
        progress.update("job", job_id, status="error", step="failed",
                        append_log=f"ERROR: {exc}\n")
        _set_status(job_id, status="error")


def _write_batch_script(job_dir, cwd, cmd):
    """Write the run.sh wrapper that sbatch executes on the compute node."""
    script = os.path.join(job_dir, "run.sh")
    body = ("#!/bin/bash\n"
            f"cd {shlex.quote(cwd)}\n"
            f"exec {' '.join(shlex.quote(c) for c in cmd)}\n")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(body)
    try:
        os.chmod(script, 0o755)
    except OSError:
        pass
    return script


def _record_slurm_id(job_dir, slurm_id):
    try:
        with open(os.path.join(job_dir, _SLURM_ID_FILE), "w") as fh:
            fh.write(slurm_id)
    except OSError:
        pass


def _run_via_sbatch(job_id, cmd, cwd, job_dir, log_file, env_vars,
                    use_gpu, gpu_model):
    """Submit the job to a compute node via `sbatch` and tail its output file."""
    progress.update("job", job_id, status="running", progress=5,
                    step="đang gửi tới SLURM, chờ cấp GPU…")
    _set_status(job_id, status="running", prog=5)

    slurm_out = os.path.join(job_dir, "slurm.out")
    open(slurm_out, "w").close()      # fresh file so the tail starts empty
    script = _write_batch_script(job_dir, cwd, cmd)
    submit = ([_SBATCH or "sbatch", "--parsable"]
              + _sbatch_flags(job_id, cwd, slurm_out, use_gpu, gpu_model)
              + [script])

    try:
        with open(log_file, "w", buffering=1,
                  encoding="utf-8", errors="replace") as logf:
            logf.write("$ " + " ".join(submit) + "\n")
            logf.write(f"(cwd={cwd})\n\n")

            res = subprocess.run(
                submit, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                env=env_vars, cwd=cwd, timeout=60,
            )
            out = (res.stdout or "").strip()
            # --parsable prints "<jobid>" or "<jobid>;<cluster>".
            slurm_id = out.split(";")[0].strip() if out else ""
            if res.returncode != 0 or not slurm_id.isdigit():
                logf.write(out + "\n")
                progress.update("job", job_id, status="error", step="sbatch lỗi",
                                append_log=out +
                                f"\nsbatch thất bại (mã {res.returncode}).\n")
                _set_status(job_id, status="error")
                return

            _record_slurm_id(job_dir, slurm_id)
            msg = f"Đã gửi SLURM job {slurm_id} (sbatch detached).\n"
            logf.write(msg)
            progress.update("job", job_id, append_log=msg,
                            step="đã gửi, chờ SLURM cấp tài nguyên…")
            _monitor_sbatch(job_id, slurm_id, slurm_out, logf)
    except subprocess.TimeoutExpired:
        progress.update("job", job_id, status="error", step="sbatch treo",
                        append_log="sbatch không phản hồi trong 60s.\n")
        _set_status(job_id, status="error")
    except Exception as exc:  # noqa: BLE001
        progress.update("job", job_id, status="error", step="failed",
                        append_log=f"ERROR: {exc}\n")
        _set_status(job_id, status="error")


def _stream_slurm_out(job_id, slurm_out, logf, pos):
    """Copy newly-appended slurm.out bytes into job.log + the live log. Returns
    the new read position."""
    if not os.path.exists(slurm_out):
        return pos
    try:
        with open(slurm_out, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(pos)
            chunk = fh.read()
            pos = fh.tell()
    except OSError:
        return pos
    if chunk:
        logf.write(chunk)
        for line in chunk.splitlines(keepends=True):
            progress.update("job", job_id, append_log=line)
            _scan_progress(job_id, line)
    return pos


def _squeue_state(slurm_id):
    """The job's squeue state (PENDING/RUNNING/…), None if it's gone from the
    queue (finished), or 'UNKNOWN' on a transient squeue error."""
    try:
        res = subprocess.run(
            ["squeue", "-h", "-j", slurm_id, "-o", "%T"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=30,
        )
    except Exception:  # noqa: BLE001
        return "UNKNOWN"
    out = (res.stdout or "").strip()
    if not out:
        return None
    return out.splitlines()[0].strip()


def _sacct_final(slurm_id):
    """Final (State, ExitCode) from accounting once the job leaves the queue.
    Retries briefly because sacct can lag a couple seconds behind squeue."""
    last_state = None
    for _ in range(6):
        try:
            res = subprocess.run(
                ["sacct", "-j", slurm_id, "-X", "-n", "-P", "-o", "State,ExitCode"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, timeout=30,
            )
        except Exception:  # noqa: BLE001
            return None, ""
        out = (res.stdout or "").strip()
        if out:
            state, _, exit_code = out.splitlines()[0].partition("|")
            state, exit_code = state.strip(), exit_code.strip()
            last_state = state
            if state and not state.endswith("ING"):   # terminal state reached
                return state, exit_code
        time.sleep(2)
    return last_state, ""


def _monitor_sbatch(job_id, slurm_id, slurm_out, logf):
    """Poll squeue and tail slurm.out until the job ends, then settle status."""
    pos = 0
    seen_running = False
    unknown = 0
    while True:
        pos = _stream_slurm_out(job_id, slurm_out, logf, pos)
        state = _squeue_state(slurm_id)
        if state is None:
            break
        if state == "UNKNOWN":
            unknown += 1
            if unknown > 30:        # ~2 min of squeue failures → fall to sacct
                break
        else:
            unknown = 0
            if state in ("PENDING", "CONFIGURING"):
                progress.update("job", job_id, step="đang chờ SLURM cấp GPU…")
            elif state in ("RUNNING", "COMPLETING") and not seen_running:
                seen_running = True
                progress.update("job", job_id, progress=8,
                                step="đã được cấp tài nguyên, đang chạy…")
        time.sleep(_POLL_SECS)

    time.sleep(1)               # let SLURM flush the final lines to slurm.out
    _stream_slurm_out(job_id, slurm_out, logf, pos)

    state, exit_code = _sacct_final(slurm_id)
    if state and state.startswith("COMPLETED"):
        _finish(job_id, 0)
    else:
        detail = (f" trạng thái={state}" if state else "") + \
                 (f" exit={exit_code}" if exit_code else "")
        progress.update("job", job_id, status="error", step="job thất bại",
                        append_log=f"\nSLURM job {slurm_id} kết thúc:{detail}\n")
        _set_status(job_id, status="error")


# --------------------------------------------------------------------------- #
# Output browsing / download
# --------------------------------------------------------------------------- #
_SNAPSHOT_FILE = ".code_snapshot.json"


def _snapshot_code_dir(job_dir, code_dir):
    """Record the relpaths present in code_dir right after the project copy."""
    rels = []
    for root, _dirs, names in os.walk(code_dir):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), code_dir)
            rels.append(rel.replace("\\", "/"))
    try:
        with open(os.path.join(job_dir, _SNAPSHOT_FILE), "w") as fh:
            json.dump(rels, fh)
    except OSError:
        pass


def _output_bases(job):
    """Yield (base_dir, [relpaths]) holding this job's downloadable files.

    Always includes the dedicated output dir ($OUTPUT_DIR). For project runs it
    also includes files NEWLY created in the code/working dir during the run —
    i.e. relative-path outputs the user code wrote next to itself — by diffing
    against the snapshot taken before launch.
    """
    out_dir = job.get("output_path")
    if out_dir and os.path.isdir(out_dir):
        rels = []
        for root, _dirs, names in os.walk(out_dir):
            for n in names:
                rels.append(os.path.relpath(os.path.join(root, n), out_dir)
                            .replace("\\", "/"))
        yield out_dir, rels

    if job.get("run_mode") == "project" and job.get("logs_path"):
        code_dir = file_utils.safe_join(job["logs_path"], "code")
        if os.path.isdir(code_dir):
            baseline = set()
            snap = os.path.join(job["logs_path"], _SNAPSHOT_FILE)
            if os.path.exists(snap):
                try:
                    with open(snap) as fh:
                        baseline = set(json.load(fh))
                except (OSError, ValueError):
                    baseline = set()
            new = []
            for root, _dirs, names in os.walk(code_dir):
                for n in names:
                    rel = os.path.relpath(os.path.join(root, n), code_dir) \
                        .replace("\\", "/")
                    if rel not in baseline:
                        new.append(rel)
            yield code_dir, new


def list_outputs(job_id):
    """Return [{rel, size}] of every downloadable file the job produced."""
    job = get_job(job_id)
    if not job:
        return []
    files, seen = [], set()
    for base, rels in _output_bases(job):
        for rel in rels:
            if rel in seen:
                continue
            seen.add(rel)
            try:
                size = os.path.getsize(os.path.join(base, rel))
            except OSError:
                size = 0
            files.append({"rel": rel, "size": size})
    files.sort(key=lambda x: x["rel"])
    return files


def resolve_output(job_id, relpath):
    """Return (base_dir, normalized_relpath) for one confined output file, or None."""
    job = get_job(job_id)
    if not job:
        return None
    for base, _rels in _output_bases(job):
        try:
            abspath, rel = file_utils.safe_relpath(base, relpath)
        except file_utils.UnsafeName:
            continue
        if os.path.isfile(abspath):
            return base, rel
    return None


def zip_outputs(job_id):
    """Build an in-memory zip of all output files. Returns (BytesIO, name)."""
    files = list_outputs(job_id)
    if not files:
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            resolved = resolve_output(job_id, f["rel"])
            if resolved:
                base, rel = resolved
                zf.write(os.path.join(base, rel), f["rel"])
    buf.seek(0)
    return buf, f"job_{job_id}_outputs.zip"


def _scancel_job(job):
    """Best-effort cancel of a still-running SLURM job before its dirs go away."""
    job_dir = job.get("logs_path")
    if not job_dir:
        return
    try:
        with open(os.path.join(job_dir, _SLURM_ID_FILE)) as fh:
            slurm_id = fh.read().strip()
    except OSError:
        return
    if slurm_id.isdigit() and shutil.which("scancel"):
        try:
            subprocess.run(["scancel", slurm_id], timeout=20,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001
            pass


def delete_job(job_id):
    job = get_job(job_id)
    if not job:
        return False
    _scancel_job(job)
    for path, base in ((job.get("logs_path"), config.JOBS_DIR),
                       (job.get("result_path"), config.RESULTS_DIR)):
        if path and file_utils.is_within(base, path) and os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except OSError:
                    pass
    db.execute("DELETE FROM jobs WHERE id=?", (job_id,), commit=True)
    return True
