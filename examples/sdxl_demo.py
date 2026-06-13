#!/usr/bin/env python3
"""
Demo: sinh ảnh bằng Stable Diffusion XL trên nền tảng host-a100
(chế độ "Run my project code").

Đặt file này làm MAIN FILE của một project, rồi ở tab Jobs chọn:
  * Run my project code
  * Project = project chứa file này, Main file = sdxl_demo.py
  * Env    = môi trường ĐÃ cài `torch diffusers transformers accelerate safetensors`
  * Model  = thư mục SDXL (diffusers format) bạn đã thêm ở tab Models
  * Bật GPU (chạy trên compute node A100)
  * Params (JSON), ví dụ:
        {"prompt": "a cozy cabin in a pine forest, golden hour, highly detailed",
         "negative_prompt": "blurry, low quality",
         "steps": 30, "guidance_scale": 7.0,
         "width": 1024, "height": 1024,
         "seed": 1234, "num_images": 1}

Nền tảng cung cấp sẵn các biến môi trường:
  MODEL_PATH  -> thư mục model (đưa thẳng vào from_pretrained)
  PARAMS_FILE -> file JSON tham số bạn nhập ở ô Params
  OUTPUT_DIR  -> ghi ảnh PNG vào đây để tải về từ tab Jobs
  JOB_ID / JOB_DIR

In ra dòng `PROGRESS <0-100>` để đẩy thanh tiến trình trên UI.

LƯU Ý về MODEL_PATH:
  SDXL ở định dạng "diffusers" là một THƯ MỤC (có model_index.json,
  unet/, vae/, text_encoder/...). Hãy upload nguyên thư mục đó lên tab Models,
  KHÔNG phải một file .safetensors đơn lẻ. Nếu bạn chỉ có 1 file
  sdxl.safetensors single-file thì cần from_single_file (xem ghi chú cuối).
"""
import json
import os
import sys

# Compute node (A100) KHÔNG có internet. Model đã nằm sẵn ở MODEL_PATH nên ép
# diffusers/huggingface_hub chạy offline để không treo khi cố gọi mạng.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def log(msg):
    """Ghi ra stdout (đi vào job.log) và flush ngay để UI thấy realtime."""
    print(msg, flush=True)


def progress(pct):
    """Đẩy thanh tiến trình trên trang Jobs."""
    print(f"PROGRESS {int(pct)}", flush=True)


def main():
    model_path = os.environ.get("MODEL_PATH", "")
    output_dir = os.environ.get("OUTPUT_DIR", ".")
    params_file = os.environ.get("PARAMS_FILE", "")

    if not model_path or not os.path.isdir(model_path):
        log(f"ERROR: MODEL_PATH không hợp lệ: {model_path!r}")
        sys.exit(1)

    # ---- Đọc tham số người dùng nhập ở ô Params (JSON) ----
    params = {}
    if params_file and os.path.isfile(params_file):
        with open(params_file, "r", encoding="utf-8") as fh:
            params = json.load(fh) or {}

    prompt = params.get("prompt") or "a scenic mountain lake at sunrise, highly detailed"
    negative_prompt = params.get("negative_prompt") or ""
    steps = int(params.get("steps", 30))
    guidance_scale = float(params.get("guidance_scale", 7.0))
    width = int(params.get("width", 1024))
    height = int(params.get("height", 1024))
    num_images = int(params.get("num_images", 1))
    seed = params.get("seed", None)  # None = ngẫu nhiên mỗi lần
    variant = params.get("variant", "fp16")  # bản fp16 đã tải; "" nếu model full

    progress(2)
    log(f"Model: {model_path}")
    log(f"Prompt: {prompt}")
    log(f"steps={steps} guidance={guidance_scale} size={width}x{height} "
        f"num_images={num_images} seed={seed}")

    # ---- Nạp thư viện (chậm lần đầu) ----
    import torch
    from diffusers import StableDiffusionXLPipeline

    has_cuda = torch.cuda.is_available()
    log(f"CUDA available: {has_cuda} "
        f"({torch.cuda.get_device_name(0) if has_cuda else 'CPU'})")
    progress(8)

    # ---- Nạp pipeline ----
    log("Đang nạp pipeline SDXL (lần đầu sẽ lâu)…")
    # float16 chạy tốt trên cả A100 lẫn RTX 6000 (Turing). variant="fp16" để
    # diffusers đọc các file *.fp16.safetensors mình đã tải. Nếu model là bản
    # full (không có file fp16) thì truyền {"variant": ""} ở Params để bỏ qua.
    dtype = torch.float16 if has_cuda else torch.float32
    load_kwargs = dict(torch_dtype=dtype, use_safetensors=True)
    if variant:
        load_kwargs["variant"] = variant
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(model_path, **load_kwargs)
    except (OSError, ValueError) as exc:
        # Thư mục không có file đúng variant -> thử lại không variant.
        log(f"Không nạp được variant={variant!r} ({exc}); thử lại không variant…")
        load_kwargs.pop("variant", None)
        pipe = StableDiffusionXLPipeline.from_pretrained(model_path, **load_kwargs)
    pipe = pipe.to("cuda" if has_cuda else "cpu")
    # A100 80GB thừa sức giữ cả model trên GPU; nếu báo OOM thì bật dòng dưới:
    # pipe.enable_model_cpu_offload()
    progress(45)

    # ---- Seed để tái lập kết quả ----
    generator = None
    if seed is not None:
        generator = torch.Generator(
            device="cuda" if has_cuda else "cpu"
        ).manual_seed(int(seed))

    # ---- Callback đẩy tiến trình theo từng bước denoise (45% -> 92%) ----
    def on_step(pipe_, step, timestep, cbk):
        progress(45 + int((step + 1) / max(steps, 1) * 47))
        return cbk

    # ---- Sinh ảnh ----
    log("Đang sinh ảnh…")
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        num_images_per_prompt=num_images,
        generator=generator,
        callback_on_step_end=on_step,
    )
    images = result.images
    progress(92)

    # ---- Ghi ảnh ra OUTPUT_DIR để tải về từ tab Jobs ----
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for i, img in enumerate(images):
        name = "image.png" if len(images) == 1 else f"image_{i + 1}.png"
        path = os.path.join(output_dir, name)
        img.save(path)
        saved.append(name)
        log(f"Đã lưu: {name}")

    with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {"prompt": prompt, "negative_prompt": negative_prompt,
             "steps": steps, "guidance_scale": guidance_scale,
             "width": width, "height": height, "seed": seed,
             "images": saved},
            fh, ensure_ascii=False, indent=2,
        )
    progress(100)
    log("Xong. Xem image.png ở phần Outputs của job.")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# Nếu model của bạn là MỘT file .safetensors single-file (không phải thư mục
# diffusers), thay đoạn from_pretrained ở trên bằng:
#
#   pipe = StableDiffusionXLPipeline.from_single_file(
#       os.path.join(model_path, "sd_xl_base_1.0.safetensors"),
#       torch_dtype=dtype, use_safetensors=True,
#   )
# -----------------------------------------------------------------------------
