"""Tests for the Tools / Clone giọng nói login-side pipeline: Vietnamese script
chunking, the atempo factor builder, the ffmpeg concat-filter graph, SRT cue
scaling, and a real ffmpeg concat+speed+denoise+MP3 round-trip (needs ffmpeg on
PATH; that part is skipped if ffmpeg is missing). Run: python3 tests/test_voice_pipeline.py

No GPU / HPC / OmniVoice needed — voice_pipeline imports only stdlib + ffmpeg.
"""
import os
import sys
import tempfile

# Use a throwaway data dir so importing config/services doesn't touch real data.
os.environ.setdefault("HOSTA100_DATA_DIR", tempfile.mkdtemp())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from services import voice_pipeline as vp  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def t1_chunking():
    print("t1: script chunking")
    chunks = vp.parse_script("")
    check("t1: empty → []", chunks == [])
    s = ("Cảnh 1: Xin chào các bạn.\n\n"
         "Đoạn hai có nhiều câu. Câu này khá dài để kiểm tra việc tách theo mệnh "
         "đề tiếng Việt nhưng vẫn giữ tự nhiên khi đọc thành tiếng.")
    chunks = vp.parse_script(s)
    check("t1: produced chunks", len(chunks) >= 1, str(len(chunks)))
    check("t1: marker stripped", not chunks[0].lower().startswith("cảnh 1"),
          chunks[0])
    # A very long run-on splits into multiple chunks at clause boundaries.
    longrun = "và ".join(["đây là một câu rất dài"] * 30)
    check("t1: long run-on splits", len(vp.parse_script(longrun)) > 1)


def t2_atempo():
    print("t2: atempo chain")
    check("t2: 1.0 → none", vp._atempo_chain(1.0) == [])
    check("t2: 1.25 single", vp._atempo_chain(1.25) == ["atempo=1.250000"])
    # Out-of-stage-range factors chain within [0.5, 2.0] and multiply back.
    f = vp._atempo_chain(0.4)
    prod = 1.0
    for x in f:
        prod *= float(x.split("=")[1])
    check("t2: 0.4 chains to product", abs(prod - 0.4) < 1e-6, str(f))
    f = vp._atempo_chain(3.0)
    prod = 1.0
    for x in f:
        prod *= float(x.split("=")[1])
    check("t2: 3.0 chains to product", abs(prod - 3.0) < 1e-6, str(f))


def t3_concat_filters():
    print("t3: concat filter graph")
    lines = vp._concat_filters(3, speed=1.0)
    check("t3: n+1 lines (concat tail)", len(lines) == 4, str(len(lines)))
    check("t3: declick fade present", "afade=t=in:st=0:d=0.015" in lines[0])
    check("t3: gap on non-last", "apad=pad_dur=0.55" in lines[0])
    check("t3: no gap after last", "apad" not in lines[2])
    check("t3: concat node", "concat=n=3:v=0:a=1[voc]" in lines[-1])
    # speed adds atempo to each chunk chain (gap stays fixed, applied per chunk)
    sped = vp._concat_filters(2, speed=1.5)
    check("t3: atempo injected per chunk", "atempo=1.500000" in sped[0])


def t4_srt():
    print("t4: SRT cue scaling")
    srt = os.path.join(tempfile.gettempdir(), "vp_test.srt")
    # raw durations 2 + 3 = 5; final 6.1 with one 0.55 gap → speech 5.55, scale 1.11
    vp.write_srt([("Câu một", 2.0), ("Câu hai dài hơn", 3.0)], srt, 6.1)
    body = open(srt, encoding="utf-8").read()
    check("t4: two cues", body.count("-->") == 2, body)
    check("t4: starts at 0", body.splitlines()[1].startswith("00:00:00,000"))
    # last cue ends at/just under the measured total (no trailing gap)
    last_end = body.strip().splitlines()[-2].split("-->")[1].strip()
    check("t4: last end near total", last_end.startswith("00:00:06"), last_end)


def t5_ffmpeg_roundtrip():
    print("t5: ffmpeg concat+speed+denoise+MP3 (needs ffmpeg)")
    import shutil
    import subprocess
    if not shutil.which(config.ffmpeg_path()):
        print("  [SKIP] ffmpeg not found on PATH")
        return
    d = tempfile.mkdtemp()
    wavs = []
    for i, freq in enumerate([220, 330, 440]):
        w = os.path.join(d, f"part_{i:04d}.wav")
        subprocess.run([config.ffmpeg_path(), "-y", "-f", "lavfi", "-i",
                        f"sine=frequency={freq}:duration=1.2",
                        "-ar", "24000", "-ac", "1", w],
                       capture_output=True)
        wavs.append(w)
    out = os.path.join(d, "out.mp3")
    vp._export_mp3(wavs, out, speed=1.25, denoise=True)
    check("t5: mp3 written", os.path.exists(out) and os.path.getsize(out) > 256)
    # 3×1.2s ÷ 1.25 = 2.88s speech + 2×0.55 gaps = 3.98s
    dur = vp.probe_duration(out)
    check("t5: duration ~3.98s", abs(dur - 3.98) < 0.5, f"{dur:.2f}s")
    # preprocess_reference → mono 24k
    ref = os.path.join(d, "ref.wav")
    vp.preprocess_reference(wavs[0], ref)
    info = subprocess.run([config.ffprobe_path(), "-v", "error", "-show_entries",
                           "stream=channels,sample_rate", "-of", "csv=p=0", ref],
                          capture_output=True, text=True).stdout.strip()
    check("t5: ref mono/24k", info == "24000,1", info)


if __name__ == "__main__":
    for fn in (t1_chunking, t2_atempo, t3_concat_filters, t4_srt, t5_ffmpeg_roundtrip):
        fn()
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL TESTS PASSED")
