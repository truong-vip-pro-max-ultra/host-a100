# host-a100 — lightweight Flask ML platform for an A100 HPC node

A single-process Flask app to upload models, manage virtual environments,
install pip packages, run GPU inference jobs on an A100, and monitor everything
in real time via polling. No Celery, no Gunicorn, no WebSockets required.

## Why single-process?

It is started with:

```bash
python3 app.py
```

All long-running work runs in **daemon threads**; Flask runs with
`threaded=True` so the upload POST and the polling GETs are served concurrently
within the one process. Progress lives in an in-memory registry
(`utils/progress.py`) plus the SQLite `jobs` table.

## Layout

```
app.py                  Flask app + all routes
inference_runner.py     Trusted runner executed inside a user-selected venv
config.py               Paths & limits (override root with HOSTA100_DATA_DIR)
services/
  storage_service.py    SQLite layer + disk usage
  model_service.py      Upload finalize, list, delete, download
  env_service.py        venv creation, pip install, pip freeze, requirements.txt
  project_service.py    User code projects: paste/upload/zip, main file
  job_service.py        Job lifecycle (runner mode + project mode) + outputs
utils/
  progress.py           In-memory progress registry (thread-safe)
  gpu.py                nvidia-smi helpers
  file_utils.py         Name/relpath validation, path confinement, safe unzip
templates/              Bootstrap UI (dashboard, upload, models, envs,
                        projects, jobs)
```

Data lives under `/data` on the server:
`/data/{models,envs,jobs,results,projects}` and `/data/platform.db`.

## Running your own code (Projects)

1. **Projects** page → create a project, then add code by **pasting** files
   (supports sub-paths like `src/model.py`) and/or **uploading** files or a
   **.zip** (auto-extracted, zip-slip safe). Pick a **main file**.
2. **Environments** page → create a venv and install deps by typing package
   names *or* uploading a `requirements.txt` (`pip install -r`).
3. **Jobs** page → choose **"Run my project code"**, select the project + main
   file + env (+ optional model + JSON params) → **Run on A100**.

Your main file is executed as `python <main_file>` inside the chosen venv, in an
isolated per-job copy of the project. These environment variables are provided:

| Var | Meaning |
|-----|---------|
| `MODEL_PATH`  | Path to the selected model dir (if any) |
| `OUTPUT_DIR`  | Write your results here → they become browsable/downloadable |
| `PARAMS_FILE` | Path to the JSON params blob you submitted |
| `JOB_DIR` / `JOB_ID` | The job's working directory / id |

Print `PROGRESS <0-100>` lines to drive the progress bar. Anything you write
under `$OUTPUT_DIR` is listed on the Jobs page for **per-file download** or
**download-all as zip**.

> Note: "Run my project code" executes arbitrary Python you provide, on the
> server. This is intended for a trusted HPC node used by you / your team. It is
> still confined to `/data` and never uses a shell, but it is **not** a sandbox
> against deliberately hostile code — add authentication before exposing it.

## Run

```bash
pip install -r requirements.txt
python3 app.py            # serves on 0.0.0.0:5000
```

Local dev on a machine without `/data` write access or without a GPU:

```bash
# Linux/macOS
HOSTA100_DATA_DIR=./_data python3 app.py
```

```powershell
# Windows PowerShell
$env:HOSTA100_DATA_DIR = "$PWD\_data"; python app.py
```

The GPU panel and CUDA checks degrade gracefully when `nvidia-smi` / PyTorch
are absent.

## Security model

- Every model/env/job name is validated against a strict whitelist; uploaded
  filenames are reduced to safe basenames.
- All filesystem access is confined inside `/data` via `file_utils.safe_join`
  / `is_within` (path-traversal proof).
- No subprocess uses `shell=True`; pip package specs are validated so users
  cannot inject flags or shell metacharacters.
- Users never supply a command line. Inference always runs the trusted
  `inference_runner.py`; users only pick model + env + JSON params.
- Destructive actions require a confirm dialog.

## API farm (LLM serving)

The **API farm** tab serves an OpenAI- and Anthropic-compatible LLM API off a
GPU node (native `llama-server`), usable from Claude Code, Cline, aider, the
OpenAI SDK, or the built-in Chat tab. Build/setup, the Anthropic bridge,
performance tuning, tests, and deploy/ops are documented in
**[docs/API_FARM.md](docs/API_FARM.md)**.

## Tools — Clone giọng nói (OmniVoice TTS)

The **Công cụ** tab hosts GPU-backed utilities. The first one, **Clone giọng
nói**, reads a script into one seamless MP3 using **OmniVoice** on a GPU node
(same sbatch lifecycle as the API farm): clone a voice from a 5–15 s sample,
adjust speed (pitch-preserving), denoise, and export a matching `.SRT`. Setup
(env, weights download, HF cache), ops and tests are in
**[docs/TOOLS_VOICE.md](docs/TOOLS_VOICE.md)**.

## Inference runner

`inference_runner.py` ships with the platform and is the only code a job
executes. It detects CUDA via PyTorch (if installed in the chosen env),
inspects the model files, emits `PROGRESS <n>` lines (parsed for the progress
bar) and writes a JSON result to `/data/results/<job_id>.json`. Replace the
`run_inference()` body with your real model loading + prediction logic while
keeping the PROGRESS + JSON contract intact.
```
