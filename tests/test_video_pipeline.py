"""Tests for the Tools / Gen video login-side code: prompt building (rule-based
path, no LLM/network), the ffmpeg Ken-Burns + audio filter graphs, SRT/ASS
subtitle generation, and a REAL ffmpeg render of fake scenes into an MP4 with a
video + audio stream (skipped if ffmpeg is missing). Includes the missing-image
flat-frame fallback. Run: python3 tests/test_video_pipeline.py

No GPU / HPC / diffusers / OmniVoice needed — the modules under test only import
stdlib (+ ffmpeg for the render); the GPU calls are not exercised here.
"""
import os
import sys
import tempfile

os.environ.setdefault("HOSTA100_DATA_DIR", tempfile.mkdtemp())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from services import video_prompts as vpr  # noqa: E402
from services import video_render as vrd  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def t1_prompts():
    print("t1: prompt building (rule-based, no LLM)")
    chunks = [
        "Vũ trụ bao la với hàng tỉ ngôi sao và thiên hà xa xôi.",
        "Các nhà khoa học khám phá bí ẩn của hành tinh đỏ.",
        "Ngân hà của chúng ta chỉ là một chấm nhỏ trong vũ trụ.",
    ]
    # use_llm=False → never touches serve_service / network.
    out = vpr.build_prompts(chunks, style_preset="cinematic", use_llm=False)
    check("t1: one prompt per chunk", len(out) == len(chunks), str(len(out)))
    check("t1: every prompt non-empty", all(o["prompt"] for o in out))
    check("t1: base style appended", all("cinematic still" in o["prompt"] for o in out))
    check("t1: negative present", all(o["negative"] for o in out))
    # Seeds are stable per chunk text (reproducible reruns).
    out2 = vpr.build_prompts(chunks, use_llm=False)
    check("t1: stable seeds", [o["seed"] for o in out] == [o["seed"] for o in out2])
    # Space-dominant script → space mood tokens fold into the prompt.
    mood_name, _ = vpr.dominant_mood(chunks)
    check("t1: dominant mood = space", mood_name == "space", mood_name)
    # Anchor derivation returns a recurring-subject string.
    anchor = vpr.derive_anchor(chunks)
    check("t1: anchor derived", isinstance(anchor, str))


def t1b_clean_screenplay():
    print("t1b: screenplay cleaning (strip stage directions / labels / headers)")
    from services.video_pipeline import clean_screenplay
    script = ("*TÊN PHIM*\n\n*[MỞ ĐẦU – 0:00-0:10]*\n\n(Cảnh màn hình tối)\n\n"
              "Narrator:\n\n\"Đây là lời thoại một.\nDòng hai của lời thoại.\"\n\n"
              "---\n\nCaster:\n\n\"Lời thoại hai!\"\n\n*ON-SCREEN TEXT*")
    out = clean_screenplay(script)
    check("t1b: keeps narration", "Đây là lời thoại một." in out)
    check("t1b: drops bracket header", "MỞ ĐẦU" not in out)
    check("t1b: drops stage direction", "màn hình tối" not in out)
    check("t1b: drops speaker label", "Narrator" not in out and "Caster" not in out)
    check("t1b: drops bold title/on-screen", "TÊN PHIM" not in out and "ON-SCREEN" not in out)
    check("t1b: strips surrounding quotes", '"Lời thoại hai' not in out
          and "Lời thoại hai!" in out)
    # Plain prose is left intact (nothing matches the screenplay rules).
    prose = "Vũ trụ bao la. Có hàng tỉ ngôi sao ngoài kia."
    check("t1b: prose unchanged", clean_screenplay(prose) == prose)


def t2_render_filters():
    print("t2: ffmpeg filter graphs")
    kb = vrd._ken_burns_filter(0, 90, 1920, 1080, 0.12, 30)
    check("t2: zoompan in filter", "zoompan=" in kb and "fps=30" in kb)
    check("t2: lanczos upscale", "scale=3840:2160:flags=lanczos" in kb)
    scenes = [{"text": "a", "duration": 2.0}, {"text": "b", "duration": 3.0}]
    lines = vrd._voice_concat_filters(scenes, 1)
    check("t2: n+1 filter lines", len(lines) == 3, str(len(lines)))
    check("t2: declick fade", "afade=t=in:st=0:d=0.015" in lines[0])
    check("t2: locked to scene dur", "atrim=0:2.000" in lines[0])
    check("t2: concat tail", "concat=n=2:v=0:a=1[voc]" in lines[-1])


def t3_subtitles():
    print("t3: SRT + ASS subtitles")
    d = tempfile.mkdtemp()
    scenes = [{"text": "Câu một ngắn.", "duration": 2.0},
              {"text": "Câu hai dài hơn một chút để tách thành nhiều cue.", "duration": 4.0}]
    srt = vrd.build_srt(scenes, vrd.DEFAULT_SUBTITLE, os.path.join(d, "x.srt"))
    body = open(srt, encoding="utf-8").read()
    check("t3: srt has cues", body.count("-->") >= 2, body[:80])
    check("t3: srt starts at 0", "00:00:00,000" in body)
    ass = vrd.build_ass(scenes, vrd.DEFAULT_SUBTITLE, 1920, 1080,
                        os.path.join(d, "x.ass"))
    a = open(ass, encoding="utf-8").read()
    check("t3: ass PlayRes pinned", "PlayResY: 1080" in a)
    check("t3: ass dialogue lines", a.count("Dialogue:") >= 2)


def _ffmpeg_ok():
    import shutil
    return bool(shutil.which(config.ffmpeg_path()))


def _mk_png(path, w, h, color):
    import subprocess
    subprocess.run([config.ffmpeg_path(), "-y", "-f", "lavfi", "-i",
                    f"color=c={color}:s={w}x{h}", "-frames:v", "1", path],
                   capture_output=True)


def _mk_wav(path, freq, dur):
    import subprocess
    subprocess.run([config.ffmpeg_path(), "-y", "-f", "lavfi", "-i",
                    f"sine=frequency={freq}:duration={dur}", "-ar", "24000",
                    "-ac", "1", path], capture_output=True)


def t4_render_roundtrip():
    print("t4: real ffmpeg render → MP4 (needs ffmpeg)")
    if not _ffmpeg_ok():
        print("  [SKIP] ffmpeg not found on PATH")
        return
    import subprocess
    d = tempfile.mkdtemp()
    work = os.path.join(d, "work")
    scenes = []
    for i, (color, freq) in enumerate([("red", 220), ("green", 330), ("blue", 440)]):
        img = os.path.join(d, f"img_{i}.png")
        wav = os.path.join(d, f"v_{i}.wav")
        _mk_png(img, 800, 450, color)
        _mk_wav(wav, freq, 1.2)
        scenes.append({"text": f"Cảnh số {i + 1} của video thử nghiệm.",
                       "image_path": img, "audio_path": wav, "duration": 1.2})
    out = os.path.join(d, "out.mp4")
    mp4, srt = vrd.render(scenes, work, out,
                          render={"width": 640, "height": 360, "fps": 24})
    check("t4: mp4 written", os.path.exists(mp4) and os.path.getsize(mp4) > 1024)
    check("t4: srt written", srt and os.path.exists(srt))
    streams = subprocess.run(
        [config.ffprobe_path(), "-v", "error", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", mp4],
        capture_output=True, text=True).stdout
    check("t4: has video stream", "video" in streams, streams)
    check("t4: has audio stream", "audio" in streams, streams)


def t5_missing_image_fallback():
    print("t5: missing-image flat-frame fallback (needs ffmpeg)")
    if not _ffmpeg_ok():
        print("  [SKIP] ffmpeg not found on PATH")
        return
    d = tempfile.mkdtemp()
    work = os.path.join(d, "work")
    # Missing image_path → renderer must synthesise a flat color frame. (Audio is
    # present; an all-silent video is a separate everything-failed edge case.)
    wav = os.path.join(d, "v.wav")
    _mk_wav(wav, 330, 1.5)
    scenes = [{"text": "Cảnh không có ảnh.", "image_path": "", "audio_path": wav,
               "duration": 1.5}]
    out = os.path.join(d, "out.mp4")
    mp4, _srt = vrd.render(scenes, work, out,
                           render={"width": 640, "height": 360, "fps": 24},
                           subtitle={"enabled": False})
    check("t5: mp4 still rendered", os.path.exists(mp4) and os.path.getsize(mp4) > 1024)


if __name__ == "__main__":
    for fn in (t1_prompts, t1b_clean_screenplay, t2_render_filters, t3_subtitles,
               t4_render_roundtrip, t5_missing_image_fallback):
        fn()
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL TESTS PASSED")
