#!/usr/bin/env python3
"""
Demo: chạy Qwen2.5-Instruct trên nền tảng host-a100 (chế độ "Run my project code").

Đặt file này làm MAIN FILE của một project, rồi ở tab Jobs chọn:
  * Run my project code
  * Project = project chứa file này, Main file = qwen_demo.py
  * Env    = môi trường ĐÃ cài `torch transformers accelerate`
  * Model  = thư mục Qwen2.5 bạn đã thêm ở tab Models
  * Bật GPU (chạy trên compute node A100)
  * Params (JSON), ví dụ:
        {"prompt": "Giới thiệu ngắn về thành phố Đà Lạt.",
         "max_new_tokens": 256, "temperature": 0.7}

Nền tảng cung cấp sẵn các biến môi trường:
  MODEL_PATH  -> thư mục model (đưa thẳng vào from_pretrained)
  PARAMS_FILE -> file JSON tham số bạn nhập ở ô Params
  OUTPUT_DIR  -> ghi kết quả vào đây để tải về từ tab Jobs
  JOB_ID / JOB_DIR

In ra dòng `PROGRESS <0-100>` để đẩy thanh tiến trình trên UI.
"""
import json
import os
import sys

# Compute node (A100) KHÔNG có internet. Model đã nằm sẵn ở MODEL_PATH nên ép
# transformers/huggingface_hub chạy offline để không treo khi cố gọi mạng.
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

    prompt = params.get("prompt") or "Xin chào! Hãy giới thiệu về chính bạn."
    system = params.get("system") or "You are Qwen, a helpful assistant."
    max_new_tokens = int(params.get("max_new_tokens", 256))
    temperature = float(params.get("temperature", 0.7))
    top_p = float(params.get("top_p", 0.8))

    progress(2)
    log(f"Model: {model_path}")
    log(f"Prompt: {prompt}")

    # ---- Nạp thư viện (chậm lần đầu) ----
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    has_cuda = torch.cuda.is_available()
    log(f"CUDA available: {has_cuda} "
        f"({torch.cuda.get_device_name(0) if has_cuda else 'CPU'})")
    progress(8)

    # ---- Nạp tokenizer + model ----
    log("Đang nạp tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    progress(20)

    log("Đang nạp model (lần đầu sẽ lâu)…")
    # bfloat16 trên A100 cho tốc độ/độ chính xác tốt; device_map="auto" để
    # transformers/accelerate tự đặt model lên GPU.
    dtype = torch.bfloat16 if has_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if has_cuda else None,
        trust_remote_code=True,
    )
    model.eval()
    progress(60)

    # ---- Dựng chat template của Qwen ----
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    progress(70)

    # ---- Sinh văn bản ----
    log("Đang sinh kết quả…")
    do_sample = temperature > 0
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
        )
    # Chỉ lấy phần token mới sinh ra (bỏ phần prompt đầu vào).
    new_tokens = generated[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    progress(92)

    log("\n===== KẾT QUẢ =====")
    log(answer)
    log("===================\n")

    # ---- Ghi kết quả ra OUTPUT_DIR để tải về từ tab Jobs ----
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "answer.txt"), "w", encoding="utf-8") as fh:
        fh.write(answer + "\n")
    with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {"prompt": prompt, "system": system, "answer": answer,
             "max_new_tokens": max_new_tokens, "temperature": temperature},
            fh, ensure_ascii=False, indent=2,
        )
    progress(100)
    log("Xong. Xem answer.txt / result.json ở phần Outputs của job.")


if __name__ == "__main__":
    main()
