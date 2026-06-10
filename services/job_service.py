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
import shutil
import subprocess
import threading
import zipfile

# Resolve srun once so we can decide whether SLURM dispatch is even possible.
_SRUN = shutil.which("srun")

import config
from services import env_service, model_service, project_service
from services import storage_service as db
from utils import file_utils, progress

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
        with open(log_file, "r", errors="replace") as fh:
            return fh.read()[-20000:]
    except OSError:
        return ""


def submit_job(model_id, env_id, job_name, params_json,
               run_mode="runner", project_id=None, main_file=None):
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

    # Cách B: prepend srun so the job runs on a GPU (A100/ampere) compute node.
    cmd = _srun_prefix(job_id, cwd) + cmd

    _launch(job_id, cmd, cwd, job_dir, output_dir, params_file,
            model_path=model["path"] if model else "")
    return job_id


def slurm_active():
    """True when jobs will actually be dispatched to a GPU node via srun."""
    return bool(config.USE_SLURM and _SRUN)


def _srun_prefix(job_id, cwd):
    """Build the `srun ...` prefix that sends a job to a GPU compute node.

    Returns [] when SLURM dispatch is off or srun is unavailable, in which case
    the job runs locally (the previous behaviour). srun streams the remote
    stdout/stderr back to us, so log capture works exactly as before; --chdir
    runs the command in the job's working dir on the (shared-filesystem) node,
    and --export=ALL forwards JOB_DIR/OUTPUT_DIR/MODEL_PATH/etc. to the job.
    """
    if not slurm_active():
        return []
    pre = [_SRUN,
           f"--gres={config.SLURM_GRES}",
           "--job-name", f"hosta100-{job_id}",
           "--chdir", cwd,
           "--export=ALL",
           "--unbuffered"]
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


def _launch(job_id, cmd, cwd, job_dir, output_dir, params_file, model_path=""):
    log_file = os.path.join(job_dir, "job.log")

    def worker():
        launch_step = ("đang gửi tới SLURM, chờ cấp GPU…" if slurm_active()
                       else "launching")
        progress.update("job", job_id, status="running", progress=5,
                        step=launch_step)
        _set_status(job_id, status="running", prog=5)

        # Environment for the child process. User code reads these to find the
        # model, write outputs, and read params — without any shell exposure.
        env_vars = dict(os.environ)
        env_vars.update({
            "PYTHONUNBUFFERED": "1",
            "JOB_ID": str(job_id),
            "JOB_DIR": job_dir,
            "OUTPUT_DIR": output_dir,
            "PARAMS_FILE": params_file,
        })
        if model_path:
            env_vars["MODEL_PATH"] = model_path

        try:
            with open(log_file, "w", buffering=1) as logf:
                logf.write("$ " + " ".join(cmd) + "\n")
                logf.write(f"(cwd={cwd})\n\n")
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, env=env_vars, cwd=cwd,
                )
                for line in proc.stdout:
                    logf.write(line)
                    progress.update("job", job_id, append_log=line)
                    stripped = line.strip()
                    low = stripped.lower()
                    # srun status lines (sent to stderr, merged here).
                    if low.startswith("srun:"):
                        if "queued and waiting" in low:
                            progress.update("job", job_id,
                                            step="đang chờ SLURM cấp GPU…")
                        elif "has been allocated" in low:
                            progress.update("job", job_id, progress=8,
                                            step="đã được cấp GPU, đang chạy…")
                    if stripped.startswith("PROGRESS"):
                        try:
                            val = int(stripped.split()[1])
                            progress.update("job", job_id, progress=val)
                            _set_status(job_id, prog=val)
                        except (IndexError, ValueError):
                            pass
                code = proc.wait()
            if code == 0:
                progress.update("job", job_id, progress=100, status="done",
                                step="completed",
                                append_log="\nJob finished successfully.\n")
                _set_status(job_id, status="done", prog=100)
            else:
                progress.update("job", job_id, status="error",
                                step="process failed",
                                append_log=f"\nProcess exited with code {code}.\n")
                _set_status(job_id, status="error")
        except Exception as exc:  # noqa: BLE001
            progress.update("job", job_id, status="error", step="failed",
                            append_log=f"ERROR: {exc}\n")
            _set_status(job_id, status="error")

    threading.Thread(target=worker, daemon=True).start()


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


def delete_job(job_id):
    job = get_job(job_id)
    if not job:
        return False
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
