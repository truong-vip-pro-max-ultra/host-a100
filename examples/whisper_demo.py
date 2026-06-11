"""
Demo faster-whisper trên host-a100 — chạy OFFLINE trên compute node (không mạng).

CÁCH DÙNG (toàn bộ qua web, không cần SSH):
  1. Tab "Môi trường": cài  faster-whisper  (thêm  nvidia-cublas-cu12 nvidia-cudnn-cu12
     nếu định chạy GPU).
  2. Tab "Môi trường" → "Tải sẵn model Whisper": chọn model (tiny/base cho nhanh) → Tải về.
     (Tải trên node đăng nhập có mạng, vào HF cache dùng chung → compute node đọc offline.)
  3. Upload file này CÙNG THƯ MỤC với audio.mp3 thành 1 project, rồi tab "Jobs" chạy
     project đó (main file = whisper_demo.py). Job chạy ở jobs/<id>/code nên audio.mp3
     phải nằm cạnh script; transcript.txt sinh ra sẽ thành file tải về được.

GPU đang full thì để DEVICE="cpu" (chạy ngay, bỏ chọn "cần GPU" ở trang Jobs).
Khi có GPU trống: đổi DEVICE="cuda", bật "cần GPU", chọn "Loại GPU" = A40/A100.
"""
import os
# PHẢI đặt TRƯỚC khi import faster_whisper/huggingface — nếu không nó sẽ thử kiểm
# tra online rồi TREO vì compute node không có mạng.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import glob
import ctypes

# >>> ĐỔI Ở ĐÂY <<<
MODEL = "base"      # tiny / base / small / medium / large-v3 (khớp model đã tải sẵn)
DEVICE = "cpu"      # "cpu" = chạy ngay; "cuda" khi có GPU trống
COMPUTE = "int8" if DEVICE == "cpu" else "float16"
AUDIO = "audio.mp3"


def preload_cuda_libs():
    """Nạp sẵn .so cuBLAS/cuDNN từ venv (chỉ cần cho GPU).

    pip cài nvidia-*-cu12 nhưng .so nằm trong site-packages/nvidia/*/lib, KHÔNG có
    trên LD_LIBRARY_PATH nên CTranslate2 không tìm thấy. Sửa env lúc chạy vô ích
    (loader chỉ đọc khi khởi động) → nạp trực tiếp bằng ctypes trước khi import.
    """
    try:
        import nvidia
    except ImportError:
        return
    # `nvidia` là namespace package (không có __init__.py) → __file__ là None.
    # Phải duyệt __path__ (danh sách thư mục), KHÔNG dùng __file__.
    sos = []
    for root in list(getattr(nvidia, "__path__", [])):
        sos += glob.glob(os.path.join(root, "*", "lib", "*.so*"))
    for _ in range(2):  # lặp 2 lượt để lib phụ thuộc nhau vẫn resolve được
        for so in sorted(sos):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def main():
    print("PROGRESS 5")
    if DEVICE == "cuda":
        preload_cuda_libs()

    from faster_whisper import WhisperModel
    print(f"[whisper] model={MODEL} device={DEVICE} compute={COMPUTE}", flush=True)
    model = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE)
    print("PROGRESS 15")

    if not os.path.isfile(AUDIO):
        raise SystemExit(f"Không thấy file '{AUDIO}' cạnh script — hãy upload kèm.")

    segments, info = model.transcribe(AUDIO, beam_size=5)
    print(f"[whisper] ngon ngu={info.language} ({info.language_probability:.0%}), "
          f"dai={info.duration:.1f}s", flush=True)

    dur = info.duration or 0
    with open("transcript.txt", "w", encoding="utf-8") as fh:
        for seg in segments:
            print(f"[{seg.start:6.2f} -> {seg.end:6.2f}] {seg.text.strip()}", flush=True)
            fh.write(seg.text.strip() + "\n")
            if dur:
                print(f"PROGRESS {15 + int(80 * min(seg.end / dur, 1.0))}")
    print("PROGRESS 100")
    print("[whisper] xong -> transcript.txt")


if __name__ == "__main__":
    main()
