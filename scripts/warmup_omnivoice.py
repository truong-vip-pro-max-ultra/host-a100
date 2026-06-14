#!/usr/bin/env python3
"""Warm the shared HF cache for OmniVoice so the GPU node (which has NO internet
and runs with HF_HUB_OFFLINE=1) finds EVERY repo it needs at runtime.

`snapshot_download('k2-fsa/OmniVoice')` only fetches the MAIN model — OmniVoice
then lazily downloads more repos when it actually runs: a vocoder/aux model on
the first generate, AND a Whisper ASR model when you clone a voice WITHOUT typing
the sample's transcript (it transcribes the clip itself). Those never reach the
cache from a plain snapshot_download, so the offline GPU server fails exactly on
those paths ("Cannot find an appropriate cached snapshot folder … offline").

Run ONCE on the LOGIN node (has internet), pointing HF_HOME at the SAME shared
cache the server uses (config.OMNI_HF_HOME, default <app>/host-a100-data/hf-cache)
with downloads enabled:

    cd ~/LeeHoang_/ollama/app/host-a100
    HF_HOME="$PWD/host-a100-data/hf-cache" HF_HUB_OFFLINE=0 \
        host-a100-data/envs/env-omnivoice/bin/python scripts/warmup_omnivoice.py

It runs on CPU (no GPU on the login node), so generation is slow and MAY raise at
the very end — that's fine as long as the DOWNLOADS finished first (this script
exercises each path in order so every repo is fetched). Re-running is cheap:
already-cached repos are reused. After it finishes, (re)start the voice server.
"""
import os
import sys
import tempfile

MODEL = os.environ.get("OMNI_MODEL", os.environ.get("HOSTA100_OMNI_MODEL",
                                                    "k2-fsa/OmniVoice"))


def main():
    # Make sure downloads are allowed even if the caller forgot the env var.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    hf_home = os.environ.get("HF_HOME", "(unset!)")
    print(f"[warmup] HF_HOME={hf_home}")
    print(f"[warmup] model={MODEL}")
    if hf_home == "(unset!)":
        print("[warmup] WARNING: HF_HOME chưa đặt — cache có thể rơi vào ~/.cache "
              "thay vì FS chung. Đặt HF_HOME=$PWD/host-a100-data/hf-cache rồi chạy lại.")

    import numpy as np
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig

    print("[warmup] 1/3 from_pretrained (tải model chính)…", flush=True)
    m = OmniVoice.from_pretrained(MODEL)
    cfg = OmniVoiceGenerationConfig(num_step=4)

    print("[warmup] 2/3 generate giọng mặc định (tải vocoder/aux)…", flush=True)
    ref_wav = os.path.join(tempfile.gettempdir(), "_omni_warm.wav")
    try:
        audio = m.generate(text="xin chào các bạn", language="vi",
                           generation_config=cfg)
        wav = audio[0] if isinstance(audio, (list, tuple)) else audio
        try:
            import soundfile as sf
            sf.write(ref_wav, np.asarray(wav, dtype=np.float32).reshape(-1), 24000)
        except Exception as exc:  # noqa: BLE001
            print(f"  (không ghi được ref wav: {exc})")
            ref_wav = ""
    except Exception as exc:  # noqa: BLE001
        print(f"  generate lỗi (KHÔNG sao nếu xảy ra SAU khi đã tải xong): {exc}")
        ref_wav = ""

    if ref_wav and os.path.exists(ref_wav):
        print("[warmup] 3/3 clone prompt với ref_text=None (tải Whisper ASR)…",
              flush=True)
        try:
            m.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=None)
            print("[warmup]   ✓ clone/ASR đã cache.")
        except Exception as exc:  # noqa: BLE001
            print(f"  clone-prompt lỗi (KHÔNG sao nếu sau khi đã tải): {exc}")
    else:
        print("[warmup] 3/3 BỎ QUA (không có ref wav) — nếu clone giọng để TRỐNG ô "
              "lời thoại sẽ cần ASR; hãy điền ô lời thoại khi clone, hoặc chạy lại "
              "warmup khi generate ở bước 2 chạy được.")

    print("\n[warmup] XONG. Kiểm tra cache:")
    print(f"    ls -la {os.path.join(os.environ.get('HF_HOME', ''), 'hub')}")
    print("Thấy các thư mục models--... (gồm cả 1 repo whisper) là đủ. "
          "Giờ (khởi động lại) server giọng nói.")


if __name__ == "__main__":
    sys.exit(main())
