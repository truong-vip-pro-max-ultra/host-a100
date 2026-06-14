#!/usr/bin/env python3
"""OmniVoice TTS server — runs on a GPU compute node, driven by host-a100.

This is the GPU half of the "Clone giọng nói" tool. It is launched on a SLURM
GPU node by ``services/voice_service.py`` (via the env's python), loads OmniVoice
once into VRAM, and serves synthesis over plain HTTP. The login-node Flask app
posts one request per text chunk and does ALL the ffmpeg post-processing
(concat / de-click / speed / denoise / MP3 / SRT) itself — this process only
turns text + an optional reference clip into a raw WAV on the shared filesystem.

Endpoints (JSON in, JSON out unless noted):
    GET  /health      → {"ok": true, "ready": <bool>, "device": "...", ...}
    POST /synthesize  → render ONE chunk to a wav on the shared FS:
        request:  {"text": "...", "out_path": "/shared/.../part_0001.wav",
                   "language": "vi", "num_step": 16, "seed": 1234,
                   "ref_audio": "/shared/voices/.../reference.wav" | "",
                   "ref_text": "..."}
        response: {"ok": true, "out_path": "...", "duration": 3.21}

Design notes (ported from clone-voice's OmniVoiceProvider, which is desktop-only):
  * The heavy model is cached process-wide; clone prompts are cached per
    (ref_audio, mtime) so the Whisper ASR + feature extraction is a one-time cost.
  * The default (zero-shot) voice is locked to one timbre by self-cloning from a
    single seeded clip — OmniVoice samples a fresh speaker from RNG otherwise, so
    a fixed seed alone does NOT reproduce the same voice across chunks.
  * OmniVoice prepends a short onset "blip" to every clip; we strip it at the
    source so the login-node concat doesn't need fragile energy trims.

It is intentionally dependency-light on the HTTP side (stdlib ``http.server``)
so the only third-party imports are torch / omnivoice / numpy / soundfile, which
live in the user's OmniVoice env.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
import threading
import traceback
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_MODEL = "k2-fsa/OmniVoice"
SR = 24000  # OmniVoice output sample rate

# A fixed, phoneme-rich Vietnamese sentence used to mint the default-voice
# reference clip (cross-lingual cloning keeps it usable for other languages too).
_DEFAULT_REF_TEXT = "Xin chào, đây là giọng kể chuyện cho video của chúng ta hôm nay."


# --------------------------------------------------------------------------- #
# Audio helpers (no ffmpeg here — the login node does all post-processing).
# --------------------------------------------------------------------------- #
def write_wav_from_float(out_path, audio, sr=SR):
    """Write a float32 mono waveform to a 16-bit WAV (soundfile, else stdlib)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    try:
        import soundfile as sf  # type: ignore
        sf.write(str(out_path), arr, sr)
        return out_path
    except Exception:
        pass
    pcm = (arr * 32767.0).astype("<i2").tobytes()
    with wave.open(str(out_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return out_path


def wav_duration(path):
    try:
        with wave.open(str(path), "r") as wf:
            return wf.getnframes() / float(wf.getframerate() or SR)
    except Exception:
        return 0.0


def strip_lead_blip(wav, sr=SR):
    """Remove OmniVoice's fixed-position onset blip (~0.11s, <0.18s long).

    Drops short voiced bursts in the first ~0.65s, keeps from the first real
    speech region. Defensive: returns the audio untouched on any oddity so we
    never risk eating speech. (Ported verbatim from clone-voice.)
    """
    try:
        import numpy as np
        a = np.asarray(wav, dtype=np.float32).reshape(-1)
        if a.size < int(0.4 * sr):
            return wav
        hop = max(1, int(0.01 * sr))
        rms = np.array([np.sqrt((a[j:j + hop] ** 2).mean() + 1e-12)
                        for j in range(0, len(a) - hop, hop)])
        db = 20.0 * np.log10(rms + 1e-9)
        voiced = db > -38.0
        regions = []
        k = 0
        while k < len(voiced):
            if voiced[k]:
                st = k
                while k < len(voiced) and voiced[k]:
                    k += 1
                regions.append((st, k))
            else:
                k += 1
        if not regions:
            return wav
        blip_zone = int(0.65 / 0.01)
        blip_max = int(0.18 / 0.01)
        lead = int(0.06 / 0.01)
        last_blip_end = 0
        cut_frame = 0
        for st, en in regions:
            if st < blip_zone and (en - st) < blip_max:
                last_blip_end = en
                continue
            cut_frame = max(last_blip_end, st - lead)
            break
        cut = int(cut_frame * hop)
        if cut <= 0 or cut > int(2.0 * sr):
            return wav
        return a[cut:]
    except Exception:
        return wav


def seed_rng(seed):
    """Pin every RNG OmniVoice might draw from (so a fixed seed = a fixed voice)."""
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# The model wrapper (lazy-loaded, cached, thread-guarded).
# --------------------------------------------------------------------------- #
class OmniEngine:
    def __init__(self, model_id=DEFAULT_MODEL):
        self.model_id = (model_id or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self._model = None
        self._device = "cpu"
        self._dtype_str = "float32"
        self._lock = threading.Lock()         # OmniVoice generate isn't reentrant
        self._prompt_cache = {}               # (ref, mtime) -> clone prompt
        self._default_ref_cache = {}          # (seed,) -> clone prompt
        self.load_error = ""
        # True GPU batching: feed up to N utterances through one generate() call
        # so the GPU does them in parallel (fills the L40S VRAM, big throughput
        # win). OmniVoice's batch API is unverified, so this is DEFENSIVE — we
        # try a batched call and fall back to per-utterance on ANY mismatch.
        self.use_batch = os.environ.get("OMNI_BATCH", "1") != "0"
        try:
            self.max_batch = max(1, int(os.environ.get("OMNI_MAX_BATCH", "8")))
        except ValueError:
            self.max_batch = 8

    # -- load -------------------------------------------------------------- #
    def _pick_device(self):
        import torch
        if torch.cuda.is_available():
            return "cuda:0", torch.float16, "float16"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps", torch.float16, "float16"
        return "cpu", torch.float32, "float32"

    def load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            import torch
            from omnivoice import OmniVoice
            device, dtype, dtype_str = self._pick_device()
            self._device, self._dtype_str = device, dtype_str
            print(f"[omnivoice] loading '{self.model_id}' on {device} ({dtype_str})…",
                  flush=True)
            try:
                model = OmniVoice.from_pretrained(self.model_id, device_map=device,
                                                  dtype=dtype)
            except TypeError:
                try:
                    model = OmniVoice.from_pretrained(self.model_id, device_map=device,
                                                      torch_dtype=dtype)
                except TypeError:
                    model = OmniVoice.from_pretrained(self.model_id)
            try:
                model.eval()
                # Squeeze the L40S: TF32 matmul/conv (Ada has fast TF32) + the
                # cudnn autotuner. fp16 weights are already chosen above. These
                # are inference-only and harmless on CPU/MPS (guarded).
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            self._model = model
            print("[omnivoice] model ready.", flush=True)
            return model

    @property
    def ready(self):
        return self._model is not None

    # -- generation -------------------------------------------------------- #
    def _gen_config(self, num_step):
        try:
            from omnivoice import OmniVoiceGenerationConfig
            return OmniVoiceGenerationConfig(num_step=int(num_step))
        except Exception:
            return None

    def _generate(self, model, text, voice_clone_prompt=None, ref_audio="",
                  ref_text="", language="", gen_config=None):
        kwargs = {"text": text}
        if language:
            kwargs["language"] = language
        if gen_config is not None:
            kwargs["generation_config"] = gen_config
        if voice_clone_prompt is not None:
            kwargs["voice_clone_prompt"] = voice_clone_prompt
        elif ref_audio:
            kwargs["ref_audio"] = ref_audio
            if ref_text:
                kwargs["ref_text"] = ref_text
        try:
            return model.generate(**kwargs)
        except TypeError:
            simple = {"text": text}
            if voice_clone_prompt is not None:
                simple["voice_clone_prompt"] = voice_clone_prompt
            elif ref_audio:
                simple["ref_audio"] = ref_audio
                if ref_text:
                    simple["ref_text"] = ref_text
            return model.generate(**simple)

    def _clone_prompt(self, model, ref_audio, ref_text):
        try:
            mtime = Path(ref_audio).stat().st_mtime
        except OSError:
            mtime = 0.0
        key = (ref_audio, mtime)
        cached = self._prompt_cache.get(key)
        if cached is not None:
            return cached
        prompt = model.create_voice_clone_prompt(ref_audio=ref_audio,
                                                 ref_text=ref_text or None)
        self._prompt_cache[key] = prompt
        return prompt

    def _default_clone_prompt(self, model, seed, language=""):
        key = (int(seed),)
        cached = self._default_ref_cache.get(key)
        if cached is not None:
            return cached
        seed_rng(seed)
        audio = self._generate(model, _DEFAULT_REF_TEXT, language=language or "vi",
                               gen_config=self._gen_config(16))
        wav = audio[0] if isinstance(audio, (list, tuple)) else audio
        ref_path = Path(tempfile.gettempdir()) / f"omni_default_ref_{abs(int(seed))}.wav"
        write_wav_from_float(ref_path, wav, sr=SR)
        prompt = model.create_voice_clone_prompt(ref_audio=str(ref_path),
                                                 ref_text=_DEFAULT_REF_TEXT)
        self._default_ref_cache[key] = prompt
        return prompt

    def _resolve_prompt(self, model, ref_audio, ref_text, seed, language):
        """Resolve the voice-conditioning prompt ONCE (cached). Returns
        (clone_prompt, raw_ref_audio, raw_ref_text)."""
        if ref_audio and Path(ref_audio).exists():
            try:
                return self._clone_prompt(model, ref_audio, ref_text), "", ""
            except Exception as exc:
                print(f"[omnivoice] clone prompt failed ({exc}); raw ref.", flush=True)
                return None, ref_audio, ref_text
        if seed is not None and seed >= 0:
            try:
                return self._default_clone_prompt(model, seed, language), "", ""
            except Exception as exc:
                print(f"[omnivoice] default voice lock failed ({exc}).", flush=True)
        return None, "", ""

    def _render_one(self, model, text, out_path, clone_prompt, raw_ref_audio,
                    raw_ref_text, language, gen_config, seed):
        """Generate + write one wav. Caller must hold self._lock."""
        spoken = (text or "").strip().lower()
        if not spoken:
            raise ValueError("text trống")
        if seed is not None and seed >= 0:
            seed_rng(seed)
        audio = self._generate(model, spoken, voice_clone_prompt=clone_prompt,
                               ref_audio=raw_ref_audio, ref_text=raw_ref_text,
                               language=language, gen_config=gen_config)
        wav = audio[0] if isinstance(audio, (list, tuple)) else audio
        wav = strip_lead_blip(wav, sr=SR)
        out = Path(out_path)
        if out.suffix.lower() != ".wav":
            out = out.with_suffix(".wav")
        write_wav_from_float(out, wav, sr=SR)
        if not out.exists() or out.stat().st_size < 256:
            raise RuntimeError("OmniVoice produced empty audio")
        return str(out), wav_duration(out)

    def synthesize(self, text, out_path, language="vi", num_step=16, seed=-1,
                   ref_audio="", ref_text=""):
        model = self.load()
        gen_config = self._gen_config(num_step)
        with self._lock:
            cp, rra, rrt = self._resolve_prompt(model, ref_audio, ref_text, seed, language)
            return self._render_one(model, text, out_path, cp, rra, rrt,
                                    language, gen_config, seed)

    def _generate_batch(self, model, texts, *, voice_clone_prompt=None,
                        ref_audio="", ref_text="", language="", gen_config=None):
        """One generate() over a LIST of texts. Returns whatever the model gives;
        the caller validates the shape. No retry — any error means 'fall back'."""
        kwargs = {"text": list(texts)}
        if language:
            kwargs["language"] = language
        if gen_config is not None:
            kwargs["generation_config"] = gen_config
        if voice_clone_prompt is not None:
            kwargs["voice_clone_prompt"] = voice_clone_prompt
        elif ref_audio:
            kwargs["ref_audio"] = ref_audio
            if ref_text:
                kwargs["ref_text"] = ref_text
        return model.generate(**kwargs)

    def _render_sub_batch(self, model, sub, clone_prompt, raw_ref_audio,
                          raw_ref_text, language, gen_config, seed):
        """Try to render a sub-batch in ONE batched generate. Returns a list of
        result dicts on success, or None to signal 'fall back to per-item'.

        We accept the batched output ONLY if it is a list/tuple of exactly
        len(sub) real waveforms — anything else (model doesn't batch, returns a
        single concatenated clip, nests differently…) triggers the safe
        per-utterance fallback, so a mismatched API can never produce garbled or
        misaligned audio."""
        import numpy as np
        texts = [(it.get("text", "") or "").strip().lower() for it in sub]
        if any(not t for t in texts):
            return None
        if seed is not None and seed >= 0:
            seed_rng(seed)
        out = self._generate_batch(model, texts, voice_clone_prompt=clone_prompt,
                                   ref_audio=raw_ref_audio, ref_text=raw_ref_text,
                                   language=language, gen_config=gen_config)
        if not isinstance(out, (list, tuple)) or len(out) != len(sub):
            return None
        results = []
        for it, x in zip(sub, out):
            a = np.asarray(x, dtype=np.float32)
            if a.ndim == 0 or a.size < 100:          # not a real waveform
                return None
            wav = strip_lead_blip(a.reshape(-1), sr=SR)
            op = Path(it["out_path"])
            if op.suffix.lower() != ".wav":
                op = op.with_suffix(".wav")
            write_wav_from_float(op, wav, sr=SR)
            if not op.exists() or op.stat().st_size < 256:
                return None
            results.append({"ok": True, "out_path": str(op),
                            "duration": wav_duration(op)})
        return results

    def synthesize_batch(self, items, language="vi", num_step=16, seed=-1,
                         ref_audio="", ref_text="", max_batch=None):
        """Render many chunks per call. Resolves the voice prompt once, then tries
        TRUE GPU batching in sub-batches of ``max_batch`` (fills the L40S), and
        falls back to per-utterance for any sub-batch the model won't batch.
        ``max_batch`` (per-request) overrides the server default so the UI can
        tune VRAM usage without recreating the server. Returns a list of per-item
        dicts (ok/out_path/duration/error)."""
        model = self.load()
        gen_config = self._gen_config(num_step)
        mb = self.max_batch
        if max_batch:
            try:
                mb = max(1, int(max_batch))
            except (TypeError, ValueError):
                pass
        n = len(items)
        results = [None] * n
        with self._lock:
            cp, rra, rrt = self._resolve_prompt(model, ref_audio, ref_text, seed, language)
            i = 0
            while i < n:
                sub = items[i:i + mb]
                batched = None
                if self.use_batch and len(sub) > 1:
                    try:
                        batched = self._render_sub_batch(
                            model, sub, cp, rra, rrt, language, gen_config, seed)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[omnivoice] batch of {len(sub)} failed "
                              f"({exc}); falling back to per-utterance.", flush=True)
                        batched = None
                if batched is not None:
                    print(f"[omnivoice] batched {len(sub)} utterances in one "
                          "generate().", flush=True)
                    for k, r in enumerate(batched):
                        results[i + k] = r
                else:
                    for k, it in enumerate(sub):
                        try:
                            path, dur = self._render_one(
                                model, it.get("text", ""), it.get("out_path", ""),
                                cp, rra, rrt, language, gen_config, seed)
                            results[i + k] = {"ok": True, "out_path": path,
                                              "duration": dur}
                        except Exception as exc:  # noqa: BLE001
                            traceback.print_exc()
                            results[i + k] = {"ok": False, "error": str(exc),
                                              "out_path": it.get("out_path", "")}
                i += len(sub)
        return results


# --------------------------------------------------------------------------- #
# Image engine — lazy diffusers SDXL, shares this GPU process with OmniVoice.
#
# Ported from the desktop "gen-video" project's DiffusersBackend (tuned on an
# RTX 4090): on a big-VRAM card keep the whole pipe RESIDENT (no model_cpu_offload,
# no attention slicing) so the UNet runs at full throughput — ~6× faster than the
# memory-saving path. It is loaded LAZILY (only on the first image request) so a
# voice-only user never pays its VRAM.
# --------------------------------------------------------------------------- #
DEFAULT_IMAGE_MODEL = "stabilityai/sdxl-turbo"
# Distilled models: 1-4 steps, NO classifier-free guidance, negative ignored.
_TURBO_HINTS = ("turbo", "lcm", "lightning", "lightni", "sdxs")


class ImageEngine:
    def __init__(self, model_id=DEFAULT_IMAGE_MODEL):
        self.model_id = (model_id or DEFAULT_IMAGE_MODEL).strip() or DEFAULT_IMAGE_MODEL
        self._pipe = None
        self._device = "cpu"
        self._dtype_str = "float32"
        self._high_vram = False
        self._lock = threading.Lock()   # diffusion isn't reentrant; own lock so
        self.load_error = ""            # image gen never blocks voice synthesis

    @property
    def ready(self):
        return self._pipe is not None

    @property
    def _is_turbo(self):
        return any(h in self.model_id.lower() for h in _TURBO_HINTS)

    def _base_res(self):
        mid = self.model_id.lower()
        if "xl" in mid:
            return 1024
        if self._is_turbo:
            return 512
        return 768

    def _target_size(self, w, h):
        """Scale W×H to the model's native sweet spot (keeps aspect, multiples of
        8). Bigger than native makes diffusion repeat/duplicate content; the
        video renderer upscales to 1080p with lanczos anyway."""
        cap = self._base_res()
        longest = max(w, h)
        if longest <= cap:
            tw, th = w, h
        else:
            scale = cap / longest
            tw, th = int(w * scale), int(h * scale)
        return max(64, (tw // 8) * 8), max(64, (th // 8) * 8)

    def load(self):
        if self._pipe is not None:
            return self._pipe
        with self._lock:
            if self._pipe is not None:
                return self._pipe
            import torch
            from diffusers import AutoPipelineForText2Image
            for _name in ("transformers", "diffusers"):
                try:
                    __import__(_name).logging.set_verbosity_error()
                except Exception:
                    pass
            if torch.cuda.is_available():
                self._device, dtype, self._dtype_str = "cuda", torch.float16, "float16"
                try:
                    total = torch.cuda.get_device_properties(0).total_memory
                    self._high_vram = total >= 16 * (1024 ** 3)
                except Exception:
                    self._high_vram = False
            else:
                self._device, dtype, self._dtype_str = "cpu", torch.float32, "float32"
            print(f"[image] loading '{self.model_id}' on {self._device} "
                  f"({self._dtype_str})…", flush=True)
            try:
                pipe = AutoPipelineForText2Image.from_pretrained(
                    self.model_id, torch_dtype=dtype, safety_checker=None,
                    variant="fp16" if dtype == torch.float16 else None)
            except Exception:
                # No fp16 variant on disk, or safety_checker kwarg rejected.
                try:
                    pipe = AutoPipelineForText2Image.from_pretrained(
                        self.model_id, torch_dtype=dtype, safety_checker=None)
                except TypeError:
                    pipe = AutoPipelineForText2Image.from_pretrained(
                        self.model_id, torch_dtype=dtype)
            pipe = pipe.to(self._device)
            # Big-VRAM card → keep resident (skip the memory-saving brakes). Only
            # on a tight card pay the speed cost of slicing/tiling/offload.
            if not self._high_vram and self._device == "cuda":
                try:
                    pipe.enable_attention_slicing()
                    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
                        pipe.vae.enable_slicing()
                    pipe.enable_vae_tiling()
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pass
            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass
            # Squeeze the GPU: TF32 matmul/conv (Ada/Ampere have fast TF32) + the
            # cudnn autotuner. Inference-only, harmless on CPU. (OmniEngine sets the
            # same; do it here too in case the image model loads first.)
            try:
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            self._pipe = pipe
            print(f"[image] model ready (high_vram={self._high_vram}).", flush=True)
            return pipe

    def _prep(self, item):
        """Normalise one request item into the kwargs the pipe needs."""
        import random
        w, h = self._target_size(int(item.get("width", 1024) or 1024),
                                 int(item.get("height", 576) or 576))
        if self._is_turbo:
            steps, guidance, neg = max(1, min(int(item.get("steps", 4) or 4), 4)), 0.0, None
        else:
            steps = max(1, int(item.get("steps", 24) or 24))
            guidance = float(item.get("cfg", 7.0) or 7.0)
            neg = (item.get("negative") or "") or None
        seed = int(item.get("seed", -1))
        if seed < 0:
            seed = random.randint(0, 2 ** 31 - 1)
        return {"prompt": (item.get("prompt") or "").strip(), "neg": neg, "w": w,
                "h": h, "steps": steps, "guidance": guidance, "seed": seed,
                "out_path": item.get("out_path", "")}

    def _save(self, image, out_path):
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            out = out.with_suffix(".png")
        image.convert("RGB").save(out, "PNG")
        if not out.exists() or out.stat().st_size < 256:
            raise RuntimeError("Diffusers produced an empty image")
        return str(out)

    def _run_group(self, pipe, group):
        """ONE pipe call over a LIST of prompts (true GPU batch) → list of PIL.
        All items in a group share w/h/steps/guidance. Negatives: list, or None
        for turbo. Per-image generators keep each scene's seed reproducible."""
        import torch
        g0 = group[0]
        prompts = [p["prompt"] for p in group]
        negs = None if g0["neg"] is None else [(p["neg"] or "") for p in group]
        gens = [torch.Generator(device=self._device).manual_seed(p["seed"]) for p in group]
        result = pipe(prompt=prompts, negative_prompt=negs,
                      num_inference_steps=g0["steps"], guidance_scale=g0["guidance"],
                      width=g0["w"], height=g0["h"], generator=gens)
        return result.images

    def generate(self, prompt, negative, out_path, width=1024, height=576,
                 steps=24, cfg=7.0, seed=-1):
        """Single image (the /generate_image endpoint)."""
        pipe = self.load()
        p = self._prep({"prompt": prompt, "negative": negative, "out_path": out_path,
                        "width": width, "height": height, "steps": steps,
                        "cfg": cfg, "seed": seed})
        with self._lock:
            image = self._run_group(pipe, [p])[0]
        return self._save(image, out_path)

    def generate_batch(self, items):
        """TRUE GPU batching: run a LIST of prompts through the pipe in ONE call so
        the GPU does them in parallel (fills VRAM, big throughput win on a 40-48GB
        card). DEFENSIVE — items are grouped by shape (w/h/steps/guidance) and any
        error (incl. CUDA OOM) in a group falls back to per-image, so a too-large
        batch never aborts the job; just lower the batch next time. Returns a list
        of {ok, out_path[, error]} aligned with ``items``."""
        pipe = self.load()
        prep = [self._prep(it) for it in items]
        n = len(prep)
        results = [None] * n
        with self._lock:
            i = 0
            while i < n:
                j = i + 1
                while (j < n and prep[j]["w"] == prep[i]["w"]
                       and prep[j]["h"] == prep[i]["h"]
                       and prep[j]["steps"] == prep[i]["steps"]
                       and prep[j]["guidance"] == prep[i]["guidance"]):
                    j += 1
                group = prep[i:j]
                images = None
                if len(group) > 1:
                    try:
                        images = self._run_group(pipe, group)
                        if not isinstance(images, list) or len(images) != len(group):
                            images = None
                    except Exception as exc:  # noqa: BLE001 — OOM etc → per-image
                        print(f"[image] batch of {len(group)} failed ({exc}); "
                              "per-image fallback.", flush=True)
                        images = None
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                if images is not None:
                    print(f"[image] batched {len(group)} images in one call.", flush=True)
                    for k, (p, img) in enumerate(zip(group, images)):
                        try:
                            results[i + k] = {"ok": True,
                                              "out_path": self._save(img, p["out_path"])}
                        except Exception as exc:  # noqa: BLE001
                            results[i + k] = {"ok": False, "error": str(exc),
                                              "out_path": p["out_path"]}
                else:
                    for k, p in enumerate(group):
                        try:
                            img = self._run_group(pipe, [p])[0]
                            results[i + k] = {"ok": True,
                                              "out_path": self._save(img, p["out_path"])}
                        except Exception as exc:  # noqa: BLE001
                            traceback.print_exc()
                            results[i + k] = {"ok": False, "error": str(exc),
                                              "out_path": p["out_path"]}
                i = j
        return results


ENGINE: OmniEngine | None = None
IMAGE_ENGINE: ImageEngine | None = None


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter; only to stdout
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # A poller (the login-node /health monitor) hung up mid-response —
            # harmless. Swallow it so socketserver doesn't dump a traceback per poll.
            pass

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/healthz", ""):
            self._send_json({
                "ok": True,
                "ready": bool(ENGINE and ENGINE.ready),
                "model": ENGINE.model_id if ENGINE else "",
                "device": ENGINE._device if ENGINE else "",
                "dtype": ENGINE._dtype_str if ENGINE else "",
                "batch": (ENGINE.use_batch if ENGINE else False),
                "max_batch": (ENGINE.max_batch if ENGINE else 0),
                "load_error": ENGINE.load_error if ENGINE else "",
                # Image side (lazy — ready only after the first /generate_image):
                "image_model": IMAGE_ENGINE.model_id if IMAGE_ENGINE else "",
                "image_ready": bool(IMAGE_ENGINE and IMAGE_ENGINE.ready),
                "image_device": IMAGE_ENGINE._device if IMAGE_ENGINE else "",
                "image_error": IMAGE_ENGINE.load_error if IMAGE_ENGINE else "",
            })
            return
        self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        route = self.path.rstrip("/")
        try:
            req = self._read_json()
        except Exception as exc:
            self._send_json({"error": f"bad json: {exc}"}, 400)
            return

        if route == "/synthesize_batch":
            items = req.get("items") or []
            if not isinstance(items, list) or not items:
                self._send_json({"error": "items trống"}, 400)
                return
            try:
                results = ENGINE.synthesize_batch(
                    items=items,
                    language=(req.get("language") or "vi"),
                    num_step=int(req.get("num_step", 16) or 16),
                    seed=int(req.get("seed", -1)),
                    ref_audio=(req.get("ref_audio") or ""),
                    ref_text=(req.get("ref_text") or ""),
                    max_batch=req.get("max_batch"),
                )
                self._send_json({"ok": True, "results": results})
            except Exception as exc:
                traceback.print_exc()
                self._send_json({"ok": False, "error": str(exc)}, 500)
            return

        if route == "/generate_image_batch":
            if IMAGE_ENGINE is None:
                self._send_json({"ok": False, "error": "image engine không bật"}, 500)
                return
            items = req.get("items") or []
            if not isinstance(items, list) or not items:
                self._send_json({"error": "items trống"}, 400)
                return
            if any(not (it.get("out_path") or "").strip() for it in items):
                self._send_json({"error": "có item thiếu out_path"}, 400)
                return
            try:
                results = IMAGE_ENGINE.generate_batch(items)
                self._send_json({"ok": True, "results": results})
            except Exception as exc:
                traceback.print_exc()
                self._send_json({"ok": False, "error": str(exc)}, 500)
            return

        if route == "/generate_image":
            if IMAGE_ENGINE is None:
                self._send_json({"ok": False, "error": "image engine không bật"}, 500)
                return
            out_path = (req.get("out_path") or "").strip()
            if not out_path:
                self._send_json({"error": "out_path bắt buộc"}, 400)
                return
            try:
                path = IMAGE_ENGINE.generate(
                    prompt=(req.get("prompt") or "").strip(),
                    negative=(req.get("negative") or ""),
                    out_path=out_path,
                    width=int(req.get("width", 1024) or 1024),
                    height=int(req.get("height", 576) or 576),
                    steps=int(req.get("steps", 24) or 24),
                    cfg=float(req.get("cfg", 7.0) or 7.0),
                    seed=int(req.get("seed", -1)),
                )
                self._send_json({"ok": True, "out_path": path})
            except Exception as exc:
                traceback.print_exc()
                self._send_json({"ok": False, "error": str(exc)}, 500)
            return

        if route == "/synthesize":
            out_path = req.get("out_path") or ""
            if not out_path:
                self._send_json({"error": "out_path bắt buộc"}, 400)
                return
            try:
                path, dur = ENGINE.synthesize(
                    text=req.get("text", ""),
                    out_path=out_path,
                    language=(req.get("language") or "vi"),
                    num_step=int(req.get("num_step", 16) or 16),
                    seed=int(req.get("seed", -1)),
                    ref_audio=(req.get("ref_audio") or ""),
                    ref_text=(req.get("ref_text") or ""),
                )
                self._send_json({"ok": True, "out_path": path, "duration": dur})
            except Exception as exc:
                traceback.print_exc()
                self._send_json({"ok": False, "error": str(exc)}, 500)
            return

        self._send_json({"error": "not found"}, 404)


def _free_port():
    s = socket.socket()
    s.bind(("0.0.0.0", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    global ENGINE, IMAGE_ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("OMNI_MODEL", DEFAULT_MODEL))
    ap.add_argument("--image-model",
                    default=os.environ.get("IMAGE_MODEL", DEFAULT_IMAGE_MODEL),
                    help="Diffusers text-to-image model (lazy-loaded on first "
                         "image request). Empty string disables image gen.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--endpoint-file", default="",
                    help="Write {host,port} JSON here once bound (for host-a100).")
    ap.add_argument("--warmup", action="store_true",
                    help="Load the model before accepting requests.")
    args = ap.parse_args()

    ENGINE = OmniEngine(args.model)
    # Image engine: created but NOT loaded — the diffusers pipe loads lazily on
    # the first /generate_image request so a voice-only server pays no VRAM.
    if (args.image_model or "").strip():
        IMAGE_ENGINE = ImageEngine(args.image_model)

    port = args.port or _free_port()
    host_name = os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".")[0]
    if args.endpoint_file:
        try:
            Path(args.endpoint_file).write_text(
                json.dumps({"host": host_name, "port": port}), encoding="utf-8")
        except OSError as exc:
            print(f"[omnivoice] could not write endpoint file: {exc}", flush=True)

    # Warm up in a background thread so /health responds immediately (showing
    # ready=false) while the multi-GB weights load — the monitor polls /health.
    def _warm():
        try:
            ENGINE.load()
        except Exception as exc:
            ENGINE.load_error = str(exc)
            traceback.print_exc()
    threading.Thread(target=_warm, daemon=True).start()

    httpd = ThreadingHTTPServer((args.host, port), Handler)
    print(f"[omnivoice] serving on {host_name}:{port} (model={args.model}, "
          f"image_model={args.image_model or '(disabled)'})", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
