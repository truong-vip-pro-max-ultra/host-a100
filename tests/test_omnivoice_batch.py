"""Tests for the OmniVoice server's TRUE GPU-batching path + its safe fallback.
Uses fake models (no torch/omnivoice/GPU needed; only numpy). Run:
    python3 tests/test_omnivoice_batch.py

Verifies: (1) when the model batches a list of texts, synthesize_batch writes one
correct wav per item via the batched path; (2) when the model does NOT accept a
list (raises) it falls back to per-utterance and still writes every wav; (3) a
batched reply with the WRONG count is rejected → per-item fallback.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
import numpy as np  # noqa: E402
import omnivoice_server as ov  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def _clip(n=ov.SR):
    return (np.ones(n, dtype=np.float32) * 0.1)


class BatchModel:
    """Accepts a list of texts and returns one waveform per text."""
    def __init__(self):
        self.calls = []

    def generate(self, text=None, **kw):
        self.calls.append(text)
        if isinstance(text, list):
            return [_clip() for _ in text]
        return [_clip()]


class NoBatchModel:
    """Only accepts a single string; raises on a list (no batch support)."""
    def __init__(self):
        self.calls = []

    def generate(self, text=None, **kw):
        self.calls.append(text)
        if isinstance(text, list):
            raise TypeError("does not accept batched text")
        return [_clip()]


class WrongCountModel:
    """Accepts a list but returns the WRONG number of clips (1 for N)."""
    def generate(self, text=None, **kw):
        if isinstance(text, list):
            return [_clip()]          # only 1 back for N → must be rejected
        return [_clip()]


def _items(d, n):
    return [{"text": f"đoạn số {i}", "out_path": os.path.join(d, f"part_{i:04d}.wav")}
            for i in range(n)]


def _engine(model):
    e = ov.OmniEngine("fake")
    e._model = model                  # bypass load() → no torch import
    return e


def t1_batched_path():
    print("t1: model that batches → batched path, all wavs written")
    d = tempfile.mkdtemp()
    m = BatchModel()
    e = _engine(m)
    items = _items(d, 5)
    res = e.synthesize_batch(items, seed=-1)
    check("t1: 5 results", len(res) == 5)
    check("t1: all ok", all(r["ok"] for r in res))
    check("t1: all wavs exist", all(os.path.exists(r["out_path"]) for r in res))
    # max_batch defaults to 8, so 5 go in ONE batched call (text passed as list)
    list_calls = [c for c in m.calls if isinstance(c, list)]
    check("t1: one batched generate call", len(list_calls) == 1, str(m.calls))
    check("t1: no per-item str calls", not any(isinstance(c, str) for c in m.calls))


def t2_fallback_on_raise():
    print("t2: model without batch → per-utterance fallback, all wavs written")
    d = tempfile.mkdtemp()
    m = NoBatchModel()
    e = _engine(m)
    items = _items(d, 4)
    res = e.synthesize_batch(items, seed=-1)
    check("t2: 4 results", len(res) == 4)
    check("t2: all ok", all(r["ok"] for r in res))
    check("t2: all wavs exist", all(os.path.exists(r["out_path"]) for r in res))
    # one failed list attempt + 4 per-item string calls
    check("t2: fell back to 4 str calls",
          sum(isinstance(c, str) for c in m.calls) == 4, str(m.calls))


def t3_reject_wrong_count():
    print("t3: batched reply with wrong count → rejected, fallback writes all")
    d = tempfile.mkdtemp()
    e = _engine(WrongCountModel())
    items = _items(d, 3)
    res = e.synthesize_batch(items, seed=-1)
    check("t3: 3 results all ok", len(res) == 3 and all(r["ok"] for r in res))
    check("t3: all wavs exist", all(os.path.exists(r["out_path"]) for r in res))


def t4_batch_disabled():
    print("t4: OMNI_BATCH=0 → never attempts a list call")
    d = tempfile.mkdtemp()
    m = BatchModel()
    e = _engine(m)
    e.use_batch = False
    res = e.synthesize_batch(_items(d, 3), seed=-1)
    check("t4: all ok", len(res) == 3 and all(r["ok"] for r in res))
    check("t4: no list calls", not any(isinstance(c, list) for c in m.calls))


def _silence(secs):
    return np.zeros(int(ov.SR * secs), dtype=np.float32)


def _tone(secs, amp=0.3):
    n = int(ov.SR * secs)
    t = np.arange(n) / float(ov.SR)
    return (amp * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)


def t5_strip_blip_keeps_short_word():
    print("t5: strip_lead_blip drops the blip but KEEPS a short opening word")
    # blip (0.11s) → pause → short word "Và" (0.12s) → pause → rest of speech.
    blip = _tone(0.11)
    short_word = _tone(0.12)
    rest = _tone(1.0)
    wav = np.concatenate([blip, _silence(0.10), short_word,
                          _silence(0.10), rest])
    out = ov.strip_lead_blip(wav)
    # The blip (~0.11s + 0.10s pause ≈ 0.21s) is removed; everything from the
    # short word on (≈ 0.12 + 0.10 + 1.0 ≈ 1.22s) survives.
    secs = len(out) / float(ov.SR)
    check("t5: blip removed (clip got shorter)", len(out) < len(wav), f"{secs:.2f}s")
    check("t5: short word kept (≥1.2s remains)", secs >= 1.2, f"{secs:.2f}s")


def t6_strip_blip_no_blip_untouched():
    print("t6: strip_lead_blip leaves a clip with no leading blip untouched")
    # Speech starts almost immediately (no real silence gap after) → not a blip.
    wav = np.concatenate([_silence(0.02), _tone(1.5)])
    out = ov.strip_lead_blip(wav)
    check("t6: untouched", len(out) == len(wav))


def t7_strip_blip_keeps_quiet_word():
    print("t7: strip_lead_blip keeps a QUIET opening word (the 'À' case)")
    # blip (loud) → true silence → a soft interjection at ~-43 dB (between the
    # -50 LOW and -38 HIGH thresholds) → silence → rest. The quiet word must
    # survive: the old single-threshold version cut straight through it.
    blip = _tone(0.11, amp=0.3)
    quiet = _tone(0.15, amp=0.01)            # ≈ -43 dB: audible but below -38
    rest = _tone(0.8, amp=0.3)
    wav = np.concatenate([blip, _silence(0.08), quiet, _silence(0.05), rest])
    out = ov.strip_lead_blip(wav)
    secs = len(out) / float(ov.SR)
    check("t7: blip removed", len(out) < len(wav), f"{secs:.2f}s")
    # blip(0.11)+gap(0.08) ≈ 0.19s removed; quiet(0.15)+0.05+rest(0.8) ≈ 1.0s kept.
    check("t7: quiet word kept (≥0.95s remains)", secs >= 0.95, f"{secs:.2f}s")
    # And the quiet word really is near the FRONT of the surviving audio (its low
    # energy sits in the first ~0.2s), proving we didn't cut through it.
    head = out[:int(0.05 * ov.SR)]
    check("t7: surviving audio starts with the quiet word (low energy head)",
          float(np.sqrt((head ** 2).mean())) < 0.05)


if __name__ == "__main__":
    for fn in (t1_batched_path, t2_fallback_on_raise, t3_reject_wrong_count,
               t4_batch_disabled, t5_strip_blip_keeps_short_word,
               t6_strip_blip_no_blip_untouched, t7_strip_blip_keeps_quiet_word):
        fn()
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL TESTS PASSED")
