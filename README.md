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

### Deploy + when to recreate a GPU server

Deploy = commit to `main` → push → `git pull` on the login node → **DETACHED**
app restart. The GPU servers (OmniVoice TTS / image gen, API-farm LLM) are
**separate SLURM processes**, so:

- **Login-side change** (`app.py`, `services/*`, `templates/*`, ffmpeg) → just
  `git pull` + restart the app.
- **GPU-side change** — editing `scripts/omnivoice_server.py` (the TTS/image
  engine), or changing the image model / quant / env → **recreate the server**
  from the Clone giọng nói tab (stop the old one, start a new one). A running
  server has already loaded its model into VRAM and will **not** pick up the new
  code on `git pull`; only a fresh server runs the updated `omnivoice_server.py`.

## Dashboard

The dashboard live-refreshes (background AJAX of `/dashboard/fragment`, no
full-page reload) and shows GPU/CPU/RAM/disk, per-model free-GPU counts scoped to
the `main-gpu` partition, top processes, and recent jobs (clickable → live log
popup). Under SLURM it also renders a **"Hàng đợi SLURM"** card from one
read-only `squeue` call (`utils/gpu.slurm_queue()`): your own jobs in detail
(state, GPU count, time used/limit, node, a Vietnamese-glossed wait reason) plus
cluster congestion counts (running/pending and how many request a GPU). Pending
start times are labelled a pessimistic worst-case ("muộn nhất ≤") — they read
days out while the job usually starts far sooner. The card disappears when
`squeue` is absent (local dev).

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
adjust speed (pitch-preserving), auto-trim the per-chunk onset/edge noise,
in-page audio preview, and a matching `.SRT`. It maxes the
L40S with fp16 + TF32 and **true GPU batching** (`HOSTA100_OMNI_MAX_BATCH`,
defensive fallback to per-utterance).

**Fast model load (important):** the venv lives on the shared NFS, where importing
torch/omnivoice cold is brutally slow (~9 min — an NFS metadata "stat storm"),
which made the server look like it "hung at loading". So `run.sh` **stages the env
to the node's local disk** (extracts a compressed tarball `host-a100-data/envs/<env>.tar.gz`
to `$TMPDIR`, ~4 s import, reused across launches) and falls back to the NFS python
if it can't (`HOSTA100_OMNI_STAGE_LOCAL`, default on). Build the tarball once on the
login node (`tar -cf <env>.tar <env>` then `pigz`/`gzip` it). The server defaults to
a small **2 CPU / 16G** SLURM request so it can slot onto GPU nodes that have a free
GPU but only 0–2 idle CPUs (asking 8 CPU made it queue for hours with `Priority`
even with GPUs free). Compute nodes may have a latin-1 console, so `run.sh` forces
`PYTHONUTF8=1` and keep `omnivoice_server.py` prints ASCII (a stray `…` in a print
once crashed the load thread and stuck `/health` at `ready:false`).

Full setup (env via the requirements file, weights warmup for the offline HF cache,
ffmpeg), all env/config knobs, a troubleshooting table and tests are in
**[docs/TOOLS_VOICE.md](docs/TOOLS_VOICE.md)**.

## Tools — Gen video từ kịch bản

The second **Công cụ** tool turns a **script into a narrated video** (AI images +
OmniVoice narration + burned subtitles + Ken-Burns slow-zoom). It **reuses the
SAME GPU server as Clone giọng nói** — `scripts/omnivoice_server.py` now also
lazy-loads a diffusers **SDXL** image model, so one SLURM job / one GPU serves
both TTS and image generation (no extra GPU slot). Image prompts are written by
the **API-farm LLM** when one is running (falls back to a rule-based prompt +
VN→EN translation). The login node does scene splitting, prompt writing and all
ffmpeg assembly. Quick setup:

1. Add `diffusers` to the OmniVoice env (re-pins `transformers==5.3.0`, leaves
   torch untouched) — upload `scripts/requirements-video-image.txt` via the
   Môi trường tab, or `pip install "diffusers>=0.30.0" "transformers==5.3.0"`.
2. Pre-download the image model into the shared HF cache on the login node
   (default = `SG161222/RealVisXL_V5.0_Lightning`, SDXL realism — far fewer
   duplicated-character / bad-hand artifacts than SDXL-Turbo):
   `snapshot_download("SG161222/RealVisXL_V5.0_Lightning")` with
   `HF_HOME=$PWD/host-a100-data/hf-cache`. **The login node runs a C locale**, so
   set `PYTHONIOENCODING=utf-8` and keep any `print()` ASCII-only or it crashes
   with `UnicodeEncodeError: latin-1` before downloading (full command in
   [docs/TOOLS_VIDEO.md](docs/TOOLS_VIDEO.md)).
3. **Recreate** the voice server (the `--image-model` branch is new; changing the
   model also needs a recreate). `ImageEngine` auto-detects turbo (no negative) vs
   lightning/lcm (few-step, CFG ~1.5, negative ON) vs full SDXL, and snaps SDXL to
   its trained aspect buckets (16:9 → 1344×768) to avoid the twin artifact.
4. *(Optional)* `pip install -U yt-dlp` in the app's python (login node) to enable
   the **"Lấy kịch bản từ YouTube"** box — paste a video URL and its subtitles drop
   into the script editor.

Full architecture, all env/config knobs (`HOSTA100_IMAGE_MODEL`,
`HOSTA100_IMAGE_MAX_BATCH`), a troubleshooting table and tests are in
**[docs/TOOLS_VIDEO.md](docs/TOOLS_VIDEO.md)**.

## Inference runner

`inference_runner.py` ships with the platform and is the only code a job
executes. It detects CUDA via PyTorch (if installed in the chosen env),
inspects the model files, emits `PROGRESS <n>` lines (parsed for the progress
bar) and writes a JSON result to `/data/results/<job_id>.json`. Replace the
`run_inference()` body with your real model loading + prediction logic while
keeping the PROGRESS + JSON contract intact.
```
