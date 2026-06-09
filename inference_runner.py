#!/usr/bin/env python3
"""
Trusted inference runner — executed by job_service inside a user-selected venv.

This script is shipped WITH the platform and is never authored by end users.
Users only choose a model + environment + JSON params; this code is what runs.

It deliberately keeps a generic shape:

  1. Report the Python / environment it is running in.
  2. Detect CUDA via PyTorch if PyTorch is installed in the env (A100 check).
  3. Inspect the model files.
  4. Emit "PROGRESS <n>" lines that job_service parses for the progress bar.
  5. Write a JSON result document.

Extend the run_inference() function with your real model-loading and inference
logic. Keeping the contract (PROGRESS lines + JSON output) intact means the web
UI continues to work unchanged.
"""
import argparse
import json
import os
import sys
import time


def log(msg):
    print(msg, flush=True)


def progress(pct):
    print(f"PROGRESS {int(pct)}", flush=True)


def detect_cuda():
    """Return a dict describing GPU availability as seen from inside the env."""
    info = {"torch_installed": False, "cuda_available": False, "devices": []}
    try:
        import torch  # noqa: WPS433 - optional dependency inside the venv
        info["torch_installed"] = True
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_mb": round(props.total_memory / (1024 ** 2)),
                    "capability": f"{props.major}.{props.minor}",
                })
    except ImportError:
        log("PyTorch not installed in this environment — skipping CUDA check.")
    except Exception as exc:  # noqa: BLE001
        log(f"CUDA detection error: {exc}")
    return info


def inspect_model(model_path):
    files = []
    if os.path.isdir(model_path):
        for root, _dirs, names in os.walk(model_path):
            for n in names:
                fp = os.path.join(root, n)
                try:
                    files.append({"file": os.path.relpath(fp, model_path),
                                  "size": os.path.getsize(fp)})
                except OSError:
                    pass
    elif os.path.isfile(model_path):
        files.append({"file": os.path.basename(model_path),
                      "size": os.path.getsize(model_path)})
    return files


def run_inference(model_path, params, cuda_info):
    """
    Placeholder inference body. Replace with real model loading + prediction.

    Emits progress while doing simulated work so the UI shows a moving bar.
    """
    steps = int(params.get("steps", 5))
    steps = max(1, min(steps, 100))
    outputs = []
    for i in range(steps):
        time.sleep(0.2)  # stand-in for real compute
        progress(10 + int(85 * (i + 1) / steps))
        outputs.append({"step": i + 1, "note": "simulated inference step"})
        log(f"Completed step {i + 1}/{steps}")
    return {"steps_run": steps, "outputs": outputs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    progress(5)
    log(f"Runner Python: {sys.executable}")
    log(f"Python version: {sys.version.splitlines()[0]}")

    try:
        with open(args.params) as fh:
            params = json.load(fh)
    except Exception:  # noqa: BLE001
        params = {}
    log(f"Parameters: {params}")

    progress(8)
    cuda_info = detect_cuda()
    log(f"CUDA info: {json.dumps(cuda_info)}")

    model_files = inspect_model(args.model_path)
    log(f"Model has {len(model_files)} file(s).")
    progress(10)

    result = {
        "status": "ok",
        "model_path": args.model_path,
        "cuda": cuda_info,
        "model_files": model_files,
        "inference": run_inference(args.model_path, params, cuda_info),
    }

    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=2)
    progress(100)
    log(f"Result written to {args.output}")


if __name__ == "__main__":
    main()
