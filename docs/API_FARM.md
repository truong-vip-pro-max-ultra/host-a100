# API farm — OpenAI/Anthropic-compatible LLM API off the HPC GPU

The **API farm** tab turns a GPU compute node into a public, OpenAI- *and*
Anthropic-compatible LLM API (usable from Claude Code, Cline, aider, the OpenAI
SDK, or the built-in Chat tab). This doc is the operational guide: architecture,
how to build/run it, how to test it, and the gotchas learned in production.

## Architecture

```
client ──HTTPS──> Cloudflare tunnel ──> Flask app (login node hpc-head1)
                                            │  reverse-proxy / translate
                                            ▼
                                   llama-server (GPU node, via sbatch)
```

- The Flask app runs on the **login node**; the GPU is on a **compute node**
  reachable only via `sbatch`. The server writes `<node>:<port>` into
  `host-a100-data/servers/<id>/endpoint.json` on the shared FS; the app
  reverse-proxies `/v1/*` to it. External clients reach only the login node
  through the Cloudflare tunnel.
- Key files: `services/serve_service.py` (launch + monitor + auto-resubmit +
  `resolve_endpoint`), `services/apikey_service.py` (Bearer keys),
  `services/anthropic_bridge.py` (Anthropic↔OpenAI translation), proxy + `/api`
  + `/v1/*` routes in `app.py`, `templates/api.html` (+ `chat.html`). DB tables
  `servers` + `api_keys` in `storage_service`.

### Three API surfaces — only ONE goes through the bridge

| Surface | Route | Goes through `anthropic_bridge`? | Clients |
|---|---|---|---|
| OpenAI chat | `/v1/chat/completions`, `/v1/models`, … | **No** — proxied straight to llama-server | OpenAI SDK, Cline, aider |
| Chat tab | `/chat/send` (session-gated) | **No** — relays the OpenAI SSE | the built-in web Chat UI |
| Anthropic | `/v1/messages`, `/v1/messages/count_tokens` | **Yes** — full translation | Claude Code |

So changes in `anthropic_bridge.to_openai_request` / the `/v1/messages` route
affect **Claude Code only**. The Chat tab and OpenAI-SDK clients speak OpenAI
directly to the server and are unaffected by bridge changes.

`/v1/*` is exempt from the session-password gate; it requires a **Bearer API
key** (managed in the API tab; a default key is minted at startup).

## Engine: native `llama-server` (required)

The server is the **native ggml-org `llama-server --jinja`** binary (NOT
`llama-cpp-python`). `--jinja` makes it use the GGUF's own chat template + its
built-in tool-call parser, so Qwen3-Coder emits **real OpenAI `tool_calls`**.
The bridge forwards the OpenAI `tools` array, so the chatml `llamacpp` engine
(which ignores that array) is **no longer offered in the UI** — do not use it.

`config.LLAMA_SERVER_BIN` (default `host-a100-data/bin/llama-server`, override
`HOSTA100_LLAMA_SERVER_BIN`) is where the app looks for the binary.

### Building the binary (one-time, on the login node)

ggml-org ships no prebuilt Linux-CUDA binary, so build from source. The cluster
has CUDA via env modules (namespaced — `module avail | grep cuda`):

```bash
cd ~/LeeHoang_/ollama/app/host-a100/host-a100-data
module load nvidia/cuda-12.4 cmake/3.31.11
nvcc --version                          # expect release 12.4

git clone --depth=1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;89" -DLLAMA_CURL=OFF
cmake --build build --config Release -j4 --target llama-server   # ~15 min

# place binary + ALL shared libs where the app expects them
mkdir -p ../bin
cp build/bin/llama-server ../bin/
find build -name '*.so*' -exec cp -av {} ../bin/ \;
# CUDA runtime libs must travel with the binary (compute node has no module load)
CUDA_LIB=$(dirname $(dirname $(which nvcc)))/lib64
cp -av $CUDA_LIB/libcudart.so.12* $CUDA_LIB/libcublas.so.12* $CUDA_LIB/libcublasLt.so.12* ../bin/
chmod +x ../bin/llama-server
```

`CMAKE_CUDA_ARCHITECTURES="80;89"` = A100 (sm80) + L40/L40S (sm89); add `86` for
A40. The login node has no GPU, so `llama-server --help` fails with
`libcuda.so.1: cannot open` — that is EXPECTED (the driver is only on GPU nodes);
`ldd ../bin/llama-server | grep "not found"` should show *only* `libcuda.so.1`.

### Registering a GGUF model

The UI uploader takes one file, but a GGUF can also be placed on disk + a DB row
inserted manually. Download on the login node (has internet), then:

```bash
cd ~/LeeHoang_/ollama/app/host-a100/host-a100-data/models
hf download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF \
  --include "*Q6_K*" --local-dir qwen3-coder-30b-q6k
# register (RUN FROM models/ — ../platform.db; running it elsewhere makes a stray empty db)
python3 - <<'PY'
import sqlite3, os, time, glob
d = os.path.abspath('qwen3-coder-30b-q6k')
assert glob.glob(os.path.join(d, '*.gguf')), "no .gguf in dir"
con = sqlite3.connect('../platform.db')
con.execute("INSERT INTO models (name,path,size,created_at) VALUES (?,?,?,?)",
            ('qwen3-coder-30b-q6k', d, 0, time.time()))
con.commit(); con.close(); print("registered ->", d)
PY
```

Schema: `models(name UNIQUE, path=DIR, size, created_at)`; the loader walks the
dir for the first `.gguf`.

### Creating a server (API farm tab)

- Engine: **llama-server** (only option)
- GPU: pick **A100** (this cluster's A100 is `A100-PCIE-40GB`, 40 GB) or L40/L40S
- Model: e.g. `qwen3-coder-30b-q6k`
- `n_ctx = 65536`, `n_gpu_layers = 99`
- **`extra_args = --parallel 1 --cache-reuse 256`** (see Performance below)

There is no edit route — to change engine/model/n_ctx/extra_args you Stop+Delete
and Create a new server (the `models` row survives).

## Performance notes (learned the hard way)

- **Decode** is fast (~95 tok/s on A100 for Qwen3-Coder-30B-A3B, a 3B-active MoE).
- **Prefill** reprocesses the prompt; the win comes from **cross-turn KV prefix
  reuse**, which needs a **single slot**: the default `n_parallel=auto=4` bounces
  each request to a different slot (LRU) → no reuse. Always set
  **`--parallel 1`** (and `--cache-reuse 256`).
- With `--parallel 1`, the **Chat tab** reuses perfectly (follow-up prefill drops
  from ~24 s to ~0.3 s). **Claude Code does NOT** reuse: it rebuilds its system
  prompt every turn with volatile content (git/file state, `<system-reminder>`s,
  env) near the FRONT, and llama.cpp matches prefixes *positionally* — any early
  change kills reuse. This is a client/design mismatch, **not fixable
  server-side**. For Claude Code, use `/compact` + short sessions to keep the
  absolute context small (10 k ctx → ~5 s prefill instead of ~24 s at 45 k).
- VRAM on the 40 GB A100: Q6_K (~25 GB) + 64 k KV (~6 GB) ≈ 34 GB, safe. Don't
  fill past ~36 GB or the load OOMs (job dies < 60 s → resubmit loop, which stops
  after 3 quick failures).

## The Anthropic bridge (Claude Code path)

`anthropic_bridge.to_openai_request` forwards the OpenAI `tools` array +
`tool_choice` and expresses tool history as `tool_calls`/`tool`-role messages —
llama-server parses tools natively into real `tool_calls`. `app.py /v1/messages`
streams token-by-token even with tools (`anthropic_bridge.stream` turns native
`tool_calls` into Anthropic `input_json_delta`). Non-stream clients get a
blocking read + `openai_response_to_anthropic` (which also falls back to parsing
Qwen `<tool_call>` text if a server ever emits it).

`config.ALERT_WEBHOOK` (env `HOSTA100_ALERT_WEBHOOK`) — if set, the app POSTs a
JSON `{content,text}` (Discord/Slack/generic) when a server dies for real
(auto-resubmit circuit breaker trips, sbatch fails, job ends never-ready).

### Pointing Claude Code here

```bash
export ANTHROPIC_BASE_URL="https://<your-tunnel-host>"   # host root, NO /v1
export ANTHROPIC_AUTH_TOKEN="<your-bearer-api-key>"
claude
```

Add a project `CLAUDE.md` to steer the weaker model, e.g.:

```
- Luôn trả lời bằng tiếng Việt.
- Before editing a file, Read it first and copy old_string EXACTLY (a weak model
  hallucinates the existing text → "Error editing file").
- To test a dev server, run it in the BACKGROUND (python3 app.py &; curl; kill %1),
  never foreground (it blocks the turn).
```

## Tests

Pure-stdlib, run from the repo root (no HPC env needed — the bridge imports only
json/re/secrets/time):

```bash
python3 tests/test_native_tools.py      # request shape, tool_choice, native tool_calls (stream + non-stream)
python3 tests/test_buffered_stream.py   # legacy text-tool_call streaming (chatml fallback path)
```

## Deploy / ops

- **Deploy** = edit + commit on the dev machine → push → `git pull` on the server
  → **DETACHED restart** of `app.py` (the web terminal is served BY the app, so a
  plain kill drops the terminal):
  ```bash
  unset -f cd ls kill pkill cat tail head   # bypass the in-app terminal guard
  APP=~/LeeHoang_/ollama/app/host-a100
  PID=$(pgrep -f "python3 app.py" | head -1)
  setsid bash -c "sleep 1; kill $PID; sleep 3; cd $APP; nohup python3 app.py >> host-a100-data/app.log 2>&1" &
  ```
  Keep `debug=False` / reloader OFF (single process + daemon threads; a reload
  mid-job orphans sbatch jobs). Jinja templates are cached because debug is off,
  so a template change ALSO needs a restart.
- **Bridge / app changes** (anthropic_bridge.py, app.py, templates) only need
  `git pull` + restart — **NOT** a server recreate. Only model/quant/n_ctx/engine/
  extra_args changes need Stop+Delete+Create.
- **Auto-resubmit**: each server runs a submit→monitor→resubmit loop (default
  walltime 24 h). It only resubmits a run that became `ready` (walltime/preempt);
  the circuit breaker stops after 3 consecutive ready-but-died-<60 s runs
  (measured from the ready moment, not from submit, so a long queue wait can't
  mask a fast crash loop).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Invalid tool parameters` in Claude Code | Old bridge didn't mark required params. Fixed by native tool_calls — ensure engine is **llama-server** and the app is restarted after pull. |
| Claude Code "loads forever" / hangs | The model ran a foreground server (`python3 app.py`) → the turn blocks. Esc; add the background rule to CLAUDE.md. |
| Slow / full-prompt prefill every turn | `n_parallel` default = 4 bounces slots → set `--parallel 1 --cache-reuse 256`. For Claude Code specifically, prefix reuse is limited by its volatile prompt → `/compact`. |
| `llama-cpp-python --cache true` made it hang | That engine serializes the WHOLE state to RAM per request (`save_state`); huge at n_ctx 65536. Don't use; the native llama-server prompt cache is fine. |
| Cloudflare Error 524 on long gens | Old buffered path went silent; the live `stream()` emits tokens continuously so the tunnel never idles. |
| Server dies < 60 s repeatedly | Usually VRAM OOM — lower `n_ctx`/`n_gpu_layers`/quant. Breaker stops after 3. |

## History

Built from the chatml `llama-cpp-python` engine + a text-`<tool_call>` parsing
bridge; migrated to the native `llama-server` engine + native `tool_calls`
(commits `0b5565d`, `e5f4133`). Earlier reliability/UX commits: `07c6621` (524
ping), `0d3f81c` (circuit breaker + alert), `99b852f` (Chat live highlight).
